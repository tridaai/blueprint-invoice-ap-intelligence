"""The AP aging queue — a read model over the awaiting invoices.

The approver's daily question is not "what did the agent conclude?"
but "what is waiting, and what is already late?" This module
answers it from the store's records, in code:

- Only invoices still at the gate (``awaiting_approver``) appear.
- Each is aged against its due date: overdue buckets (1–7, 8–30,
  30+ days), due today, due within a week, due later, no due date.
- Ordering is work order: most overdue first, then nearest due date.
- The triage mix and per-currency totals ride along, so the queue
  reads as a workload, not a list.

It writes nothing and decides nothing — the same drafts wait at the
same gate; this is the order a human works them in.

Open **recovery cases** age here too: a duplicate-payment case
sitting in ``supplier_contacted`` for a month is money leaking, so
each open case carries its total age, the age of its current
status, and an SLA flag (``RECOVERY_CASE_SLA_DAYS``) — the queue
answers "what is stuck, and since when?" for cases as well as
invoices.

**Escalated open postings** join the same worklist: an approved
invoice lives on the approved-not-posted report once it outlasts
the grace window, but past a second, longer threshold
(``POSTING_ESCALATION_DAYS``) it stops being a report line and
becomes an action item here, carrying the ERP's failure reason
when a receipt failed — one list for everything the building
still owes an answer for. The service computes them through
``posting.approved_not_posted`` at the escalation threshold; this
module only seats them in the view.

**Four-eyes payment runs** age here too: a large run confirmed
once is locked out of hand-off export until a second, different
confirmer arrives, so past ``PAYMENT_RUN_FOUR_EYES_AGING_DAYS``
the wait joins the worklist as an action item — the run, its
net total, how long it has waited, and the first confirmer's
name (the person to nudge, and the one person who cannot be
the second). The threshold is forecast-aware: a run whose
confirmation-time drift forecast flags members lapsing before
the expected export has no slack — every day of the wait is a
day the projected lapse gets closer — so its threshold
tightens to 0 days and it is chased from the first day of the
wait, carrying the projected lapse value the wait puts at risk.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

from .matching import _parse_date
from .schemas import (
    FourEyesAgingItem,
    OpenPostingItem,
    PaymentRun,
    QueueItem,
    QueueView,
    RecoveryCaseAgingItem,
    RecoveryPack,
)
from .store import ApprovalRecord

_SEVERITY_RANK = {"high": 0, "moderate": 1, "low": 2}

DEFAULT_RECOVERY_SLA_DAYS = 14
DEFAULT_FOUR_EYES_AGING_DAYS = 2


def _age_bucket(age_days: int | None) -> str:
    if age_days is None:
        return "unknown"
    if age_days > 30:
        return "age_30_plus"
    if age_days >= 8:
        return "age_8_30"
    return "age_0_7"


def recovery_case_aging(
    packs: Iterable[RecoveryPack],
    *,
    as_of: date,
    sla_days: int = DEFAULT_RECOVERY_SLA_DAYS,
) -> tuple[list[RecoveryCaseAgingItem], dict[str, int], int]:
    """Age the open recovery cases: (items, buckets, sla_breaches).

    Closed cases are done and drop out. A case's age runs from the
    pack's creation; its status age runs from the last history
    event (the transition that put it in its current status). The
    SLA is on the *status* age — a freshly-transitioned old case is
    moving, not stuck. Work order: SLA breaches first, then the
    longest-stuck, then the oldest.
    """
    items: list[RecoveryCaseAgingItem] = []
    buckets: dict[str, int] = {}
    breaches = 0
    for pack in packs:
        if pack.case_status == "closed":
            continue
        created = _parse_date(pack.created_at)
        age_days = (as_of - created).days if created is not None else None
        last_at = pack.case_history[-1].at if pack.case_history else pack.created_at
        since = _parse_date(last_at)
        status_age = (as_of - since).days if since is not None else None
        breached = status_age is not None and status_age > sla_days
        if breached:
            breaches += 1
        bucket = _age_bucket(age_days)
        buckets[bucket] = buckets.get(bucket, 0) + 1
        items.append(
            RecoveryCaseAgingItem(
                pack_id=pack.pack_id,
                duplicate_invoice_id=pack.duplicate_invoice_id,
                supplier_name=pack.supplier_name,
                currency=pack.currency,
                amount=pack.amount,
                case_status=pack.case_status,
                age_days=age_days,
                status_age_days=status_age,
                sla_breached=breached,
                bucket=bucket,
            )
        )
    items.sort(
        key=lambda item: (
            0 if item.sla_breached else 1,
            -(item.status_age_days or 0),
            -(item.age_days or 0),
            item.pack_id,
        )
    )
    return items, buckets, breaches


# The tightened aging threshold for a forecast-flagged four-eyes
# run: zero days of patience. The configured threshold is the
# allowance for a run whose discounts are safe; a run the
# confirmation-time forecast already projects to lapse has no
# allowance to spend — the wait itself is the leak (see the rule
# in four_eyes_aging).
FORECAST_TIGHTENED_AGING_DAYS = 0


def four_eyes_aging(
    runs: Iterable[PaymentRun],
    *,
    as_of: date,
    aging_days: int = DEFAULT_FOUR_EYES_AGING_DAYS,
) -> list[FourEyesAgingItem]:
    """The four-eyes runs whose second confirmation is overdue:
    confirmed once, stamped ``four_eyes_required``, no second
    confirmer yet, and the first confirmation older than the
    run's aging threshold. A run inside its threshold is a normal
    in-flight wait, not a stall; a second-confirmed (or exported)
    run is done waiting and drops out.

    The threshold is per run, and the forecast sets it: when the
    run's confirmation-time drift forecast (``run.drift_forecast``)
    flags members lapsing before the expected export, the second
    confirmation is the only thing standing between the run and
    its export — so the threshold tightens from the configured
    ``aging_days`` to ``FORECAST_TIGHTENED_AGING_DAYS`` (0): the
    run joins the worklist on the first day of its wait, and the
    item carries the forecast's lapse count and value at risk.
    A run with an empty forecast keeps the configured threshold —
    the forecast was computed at confirmation and is not
    recomputed here.

    Work order: forecast-tightened items first (value at risk
    outranks raw age), then the longest wait — it has the least
    discount headroom left.
    """
    items: list[FourEyesAgingItem] = []
    for run in runs:
        if run.status != "confirmed" or not run.four_eyes_required:
            continue
        if run.second_confirmed_by:
            continue
        confirmed = _parse_date(run.confirmed_at or "")
        if confirmed is None:
            continue
        age_days = (as_of - confirmed).days
        tightened = bool(run.drift_forecast)
        threshold = FORECAST_TIGHTENED_AGING_DAYS if tightened else aging_days
        if age_days <= threshold:
            continue
        items.append(
            FourEyesAgingItem(
                run_id=run.run_id,
                currency=run.currency,
                total_net=run.total_net,
                four_eyes_threshold=run.four_eyes_threshold,
                confirmed_by=run.confirmed_by or "",
                confirmed_at=run.confirmed_at or "",
                age_days=age_days,
                aging_threshold_days=threshold,
                forecast_tightened=tightened,
                forecast_lapse_count=len(run.drift_forecast),
                forecast_value_at_risk=round(
                    sum(e.discount_amount for e in run.drift_forecast), 2
                ),
            )
        )
    items.sort(
        key=lambda item: (
            0 if item.forecast_tightened else 1,
            -(item.age_days or 0),
            item.run_id,
        )
    )
    return items


def _bucket(days_overdue: int | None, due: date | None, as_of: date) -> str:
    if due is None or days_overdue is None:
        return "no_due_date"
    if days_overdue > 30:
        return "overdue_30_plus"
    if days_overdue >= 8:
        return "overdue_8_30"
    if days_overdue >= 1:
        return "overdue_1_7"
    if days_overdue == 0:
        return "due_today"
    if (due - as_of).days <= 7:
        return "due_within_7"
    return "due_later"


def build_queue(
    records: list[ApprovalRecord],
    *,
    as_of: date | None = None,
    recovery_packs: Iterable[RecoveryPack] = (),
    recovery_sla_days: int = DEFAULT_RECOVERY_SLA_DAYS,
    open_postings: Iterable[OpenPostingItem] = (),
    posting_escalation_days: int = 0,
    payment_runs: Iterable[PaymentRun] = (),
    four_eyes_aging_days: int = DEFAULT_FOUR_EYES_AGING_DAYS,
) -> QueueView:
    """The aging queue over the store's records, as of a date
    (today when not given — tests and API callers pass it
    explicitly for a stable view). Open recovery cases age
    alongside the invoices (see :func:`recovery_case_aging`), and
    open postings past the escalation threshold ride along as
    action items, oldest first (``posting.approved_not_posted``
    already orders them). Four-eyes runs whose second
    confirmation has outlasted the aging threshold join them
    (see :func:`four_eyes_aging`)."""
    as_of = as_of or date.today()
    items: list[QueueItem] = []
    triage_mix: dict[str, int] = {}
    totals: dict[str, float] = {}
    buckets: dict[str, int] = {}
    overdue_count = 0
    for record in records:
        result = record.result
        if result.status != "awaiting_approver":
            continue
        invoice = record.invoice or {}
        due = _parse_date(str(invoice.get("due_date", "")))
        days_overdue = (as_of - due).days if due is not None else None
        bucket = _bucket(days_overdue, due, as_of)
        buckets[bucket] = buckets.get(bucket, 0) + 1
        if days_overdue is not None and days_overdue > 0:
            overdue_count += 1
        outcome = result.triage.outcome
        triage_mix[outcome] = triage_mix.get(outcome, 0) + 1
        total = invoice.get("total_amount")
        currency = str(invoice.get("currency") or result.facts.currency or "USD")
        if total is not None:
            totals[currency] = round(totals.get(currency, 0.0) + float(total), 2)
        submitted = _parse_date(str(invoice.get("submitted_at", ""))) or _parse_date(
            str(invoice.get("invoice_date", ""))
        )
        discount = result.discount
        discount_deadline = discount.discount_deadline if discount else ""
        deadline = _parse_date(discount_deadline)
        items.append(
            QueueItem(
                invoice_id=result.invoice_id,
                supplier_name=result.supplier_name
                or str(invoice.get("supplier_name", "")),
                invoice_number=str(invoice.get("invoice_number", "")),
                outcome=outcome,
                severity=result.triage.severity,
                total_amount=float(total) if total is not None else None,
                currency=currency,
                due_date=due.isoformat() if due else "",
                days_overdue=days_overdue,
                bucket=bucket,
                age_days=(as_of - submitted).days if submitted else None,
                discount_deadline=discount_deadline,
                discount_days_left=(deadline - as_of).days if deadline else None,
            )
        )

    def sort_key(item: QueueItem) -> tuple:
        overdue = item.days_overdue is not None and item.days_overdue > 0
        return (
            0 if overdue else 1,  # overdue first
            -(item.days_overdue or 0),  # most overdue first
            item.due_date or "9999-12-31",  # then nearest due date
            _SEVERITY_RANK.get(item.severity, 3),  # then highest severity
            item.invoice_id,
        )

    items.sort(key=sort_key)
    case_items, case_buckets, case_breaches = recovery_case_aging(
        recovery_packs, as_of=as_of, sla_days=recovery_sla_days
    )
    posting_items = list(open_postings)
    four_eyes_items = four_eyes_aging(
        payment_runs, as_of=as_of, aging_days=four_eyes_aging_days
    )
    return QueueView(
        as_of=as_of.isoformat(),
        count=len(items),
        overdue_count=overdue_count,
        triage_mix=triage_mix,
        totals_by_currency=totals,
        buckets=buckets,
        items=items,
        recovery_cases=case_items,
        recovery_buckets=case_buckets,
        recovery_open_count=len(case_items),
        recovery_sla_breaches=case_breaches,
        recovery_sla_days=recovery_sla_days,
        open_postings=posting_items,
        open_posting_count=len(posting_items),
        posting_escalation_days=posting_escalation_days,
        four_eyes_pending=four_eyes_items,
        four_eyes_pending_count=len(four_eyes_items),
        four_eyes_aging_days=four_eyes_aging_days,
    )
