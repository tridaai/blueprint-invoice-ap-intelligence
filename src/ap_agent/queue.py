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

**Twin clusters** escalate here too. The twin graph
(``twins.twin_clusters``) was an audit read: it named the
clusters, but the approver working this queue never saw one as
a case — a twin could arrive while its sibling waited, or one
copy could be approved while another sat at the gate. A
cluster becomes a work item when it is big enough
(``TWIN_CLUSTER_MIN_SIZE``), valuable enough in one currency
(``TWIN_CLUSTER_VALUE_THRESHOLD``), or spans an approved twin
and a waiting one — the double-payment shape itself, at any
size. The item carries the graph (members + screened edges),
so the case is worked whole, not record by record — and it
can be handed off whole (``service.decide_twin_cluster``):
approved through the cluster in one act, each member's
decision recorded on its own record, or split with a named
reason when the grouping itself is wrong, after which the
members are worked individually and the cluster leaves the
queue.

**Information waits** age here too. An invoice blocked on an
information request waits on someone outside the building —
purchasing owes a PO, the supplier owes a legible copy — and
that wait used to be invisible: the case sat at the gate with
nothing ageing it. Past ``INFORMATION_WAIT_AGING_DAYS`` an
unanswered request (or one a partial reply restarted) joins
the worklist with the still-missing items named, so the chase
has an owner and an age like every other stall. Past twice
that, the wait stops being a list line: the service's sweep
fires the signed ``information_wait_escalation`` webhook, once
per wait episode (a partial reply starts a new episode; a
complete reply is the reset), and the item here reads
``escalated``. The firings are graded in turn
(:func:`information_escalation_follow_through`): a reply that
followed the push, a wait closed without one, or a wait still
open exactly as it was — the escalation's effect, measured.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

from .matching import _parse_date
from .schemas import (
    FourEyesAgingItem,
    InformationWaitItem,
    OpenPostingItem,
    PaymentRun,
    QueueItem,
    QueueView,
    RecoveryCaseAgingItem,
    RecoveryPack,
    TwinEscalationItem,
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


DEFAULT_TWIN_CLUSTER_MIN_SIZE = 3
DEFAULT_TWIN_CLUSTER_VALUE_THRESHOLD = 10000.0


def twin_escalation_items(
    records: list[ApprovalRecord],
    *,
    min_size: int = DEFAULT_TWIN_CLUSTER_MIN_SIZE,
    value_threshold: float = DEFAULT_TWIN_CLUSTER_VALUE_THRESHOLD,
) -> list[TwinEscalationItem]:
    """The twin clusters that belong on the worklist, with their
    graphs attached. A cluster escalates on ANY trigger (see
    ``TwinEscalationItem``): size, combined per-currency value,
    or an approved twin spanning a waiting one. Clusters under
    every trigger stay audit-only — the graph endpoint still
    reads them; the queue is for work, and a small, cheap,
    undecided pair is a glance, not a case.

    Work order: an approved+awaiting span first (money already
    committed on one copy is the live risk), then the largest
    combined value, then the largest cluster.

    Two kinds of cluster never escalate. A cluster with no
    member left at the gate is decided — there is nothing to
    hand off, whatever its size. And a cluster the approver
    split (a ``split`` hand-off decision stamped on every
    member, service.decide_twin_cluster) is no longer one case:
    its members are worked individually, and a later arrival
    that twins with them — unstamped — re-opens the cluster as
    a case in its own right."""
    from .twins import twin_clusters

    items: list[TwinEscalationItem] = []
    for cluster in twin_clusters(records):
        statuses = {str(member["status"]) for member in cluster["members"]}
        if "awaiting_approver" not in statuses:
            continue
        if all(
            any(
                decision.get("cluster_id") == cluster["cluster_id"]
                and decision.get("action") == "split"
                for decision in member["cluster_decisions"]
            )
            for member in cluster["members"]
        ):
            continue
        totals: dict[str, float] = {}
        for member in cluster["members"]:
            if member["total_amount"] is not None:
                currency = str(member["currency"] or "USD")
                totals[currency] = round(
                    totals.get(currency, 0.0) + float(member["total_amount"]), 2
                )
        triggers: list[str] = []
        if cluster["size"] >= min_size:
            triggers.append("cluster_size")
        if any(value >= value_threshold for value in totals.values()):
            triggers.append("combined_value")
        if "approved" in statuses and "awaiting_approver" in statuses:
            triggers.append("approved_and_awaiting")
        if not triggers:
            continue
        items.append(
            TwinEscalationItem(
                cluster_id=cluster["cluster_id"],
                supplier_name=cluster["canonical_supplier"],
                invoice_number=cluster["canonical_invoice_number"],
                size=cluster["size"],
                totals_by_currency=totals,
                triggers=triggers,
                arrival_shapes=cluster["arrival_shapes"],
                members=cluster["members"],
                edges=cluster["edges"],
            )
        )
    items.sort(
        key=lambda item: (
            0 if "approved_and_awaiting" in item.triggers else 1,
            -max(item.totals_by_currency.values(), default=0.0),
            -item.size,
            item.cluster_id,
        )
    )
    return items


DEFAULT_INFORMATION_WAIT_AGING_DAYS = 3


def wait_episode(request, invoice: dict) -> tuple[date | None, list[str]]:
    """A wait's current episode: (the anchor date its age runs
    from, the items still missing in that episode).

    With no reply recorded, the episode runs from the request's
    own stamp (``requested_at``; requests stored before the stamp
    existed fall back to the invoice's submission date). A partial
    reply starts a NEW episode: the supplier answered, the ball
    moved, and the remaining items have been owed only since that
    answer landed (``reply_received_at``). The escalation ledger
    dedupes against the episode — one firing per episode, and a
    partial reply's new episode can escalate in its turn. Shared
    by the queue read and the escalation sweep so the two can
    never disagree about when a wait began."""
    if request.reply_status == "partially_answered" and request.reply_received_at:
        return _parse_date(request.reply_received_at), (
            request.reply_still_missing or request.missing_items
        )
    anchor = _parse_date(request.requested_at) or _parse_date(
        str(invoice.get("submitted_at", ""))
    ) or _parse_date(str(invoice.get("invoice_date", "")))
    still_missing = request.missing_items
    if request.reply_received_at:
        # A reply landed but resolved nothing the request asked
        # for: the remaining items are everything.
        still_missing = request.reply_still_missing or still_missing
    return anchor, still_missing


def information_wait_items(
    records: list[ApprovalRecord],
    *,
    as_of: date,
    aging_days: int = DEFAULT_INFORMATION_WAIT_AGING_DAYS,
) -> list[InformationWaitItem]:
    """The information requests that have outlasted their wait:
    invoices still at the gate whose request is unanswered — or
    answered only in part — past the aging threshold.

    The wait's clock is its episode's (see :func:`wait_episode`).
    A fully answered request closes the wait whatever its age,
    and a decided invoice is not waiting at all. An item whose
    episode the escalation sweep has already fired for reads
    ``escalated`` — the chase was pushed, not just listed. Work
    order: the longest wait first.
    """
    items: list[InformationWaitItem] = []
    for record in records:
        result = record.result
        if result.status != "awaiting_approver":
            continue
        request = result.information_request
        if not result.needs_information or request is None:
            continue
        if request.reply_status == "answered":
            continue
        invoice = record.invoice or {}
        anchor, still_missing = wait_episode(request, invoice)
        if anchor is None:
            continue
        waiting_days = (as_of - anchor).days
        if waiting_days <= aging_days:
            continue
        episode_since = anchor.isoformat()
        items.append(
            InformationWaitItem(
                invoice_id=result.invoice_id,
                supplier_name=result.supplier_name
                or str(invoice.get("supplier_name", "")),
                invoice_number=str(invoice.get("invoice_number", "")),
                outcome=result.triage.outcome,
                requested_at=request.requested_at,
                reply_status=request.reply_status,
                reply_received_at=request.reply_received_at,
                waiting_days=waiting_days,
                aging_threshold_days=aging_days,
                still_missing=list(still_missing),
                episode_since=episode_since,
                escalated=any(
                    entry.episode_since == episode_since
                    for entry in result.information_escalations
                ),
            )
        )
    items.sort(key=lambda item: (-(item.waiting_days or 0), item.invoice_id))
    return items


def information_escalation_follow_through(records: list[ApprovalRecord]) -> dict:
    """Grade the recorded information-wait escalations against
    what happened next: did the push work?

    The escalation ledger (``result.information_escalations``)
    records that a wait was pushed — once per episode, at twice
    the aging threshold. This reads each firing against the
    record as it now stands and grades the episode:

    - ``answered_after_push`` — a reply arrived at or after the
      dispatch (the request's latest reply stamp is the
      evidence): whole or partial, the chase landed. A partial
      reply starts a new episode, so an old episode followed by
      ANY later reply reads as answered — the push was the last
      thing that happened before the supplier moved.
    - ``closed_otherwise`` — no reply followed the push, but the
      wait is over anyway: the case was decided at the gate, or
      the request left the record. The push cannot claim it, and
      the grade does not let it.
    - ``still_open`` — no reply, no decision: the wait is open
      exactly as it was when the webhook fired. A chase failing
      in public, counted.

    Answered episodes also carry their days-to-answer (dispatch
    date → reply date), aggregated as count / median / mean /
    max: the push's price in days, not just its hit rate.
    ``answer_rate`` is answered over graded — None when nothing
    has been graded, never a dressed-up zero. Pure over the
    records.
    """
    graded = 0
    answered = 0
    still_open = 0
    closed_otherwise = 0
    answer_days: list[int] = []
    still_open_ids: list[str] = []
    for record in records:
        result = record.result
        for entry in result.information_escalations:
            graded += 1
            dispatched = _parse_date(entry.dispatched_at)
            request = result.information_request
            reply_at = (
                _parse_date(request.reply_received_at)
                if request is not None
                else None
            )
            if (
                request is not None
                and reply_at is not None
                and dispatched is not None
                and reply_at >= dispatched
                and request.reply_status in ("partially_answered", "answered")
            ):
                answered += 1
                answer_days.append((reply_at - dispatched).days)
            elif (
                result.status != "awaiting_approver"
                or request is None
                or not result.needs_information
            ):
                closed_otherwise += 1
            else:
                still_open += 1
                still_open_ids.append(result.invoice_id)
    days_block = None
    if answer_days:
        ordered = sorted(answer_days)
        days_block = {
            "count": len(ordered),
            "median": ordered[len(ordered) // 2]
            if len(ordered) % 2 == 1
            else round((ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2, 2),
            "mean": round(sum(ordered) / len(ordered), 2),
            "max": ordered[-1],
        }
    return {
        "escalations_graded": graded,
        "answered_after_push": answered,
        "still_open": still_open,
        "closed_otherwise": closed_otherwise,
        "answer_rate": round(answered / graded, 4) if graded else None,
        "days_to_answer": days_block,
        "still_open_invoice_ids": sorted(set(still_open_ids)),
    }


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
    twin_items: Iterable[TwinEscalationItem] = (),
    twin_min_size: int = DEFAULT_TWIN_CLUSTER_MIN_SIZE,
    twin_value_threshold: float = DEFAULT_TWIN_CLUSTER_VALUE_THRESHOLD,
    information_waits: Iterable[InformationWaitItem] = (),
    information_wait_aging_days: int = DEFAULT_INFORMATION_WAIT_AGING_DAYS,
) -> QueueView:
    """The aging queue over the store's records, as of a date
    (today when not given — tests and API callers pass it
    explicitly for a stable view). Open recovery cases age
    alongside the invoices (see :func:`recovery_case_aging`), and
    open postings past the escalation threshold ride along as
    action items, oldest first (``posting.approved_not_posted``
    already orders them). Four-eyes runs whose second
    confirmation has outlasted the aging threshold join them
    (see :func:`four_eyes_aging`), twin clusters past their
    escalation triggers join as cases with their graphs (see
    :func:`twin_escalation_items`), and information waits past
    their aging threshold ride along computed by the service
    (see :func:`information_wait_items`)."""
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
    twin_list = list(twin_items)
    wait_list = list(information_waits)
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
        twin_escalations=twin_list,
        twin_escalation_count=len(twin_list),
        twin_cluster_min_size=twin_min_size,
        twin_cluster_value_threshold=twin_value_threshold,
        information_waits=wait_list,
        information_wait_count=len(wait_list),
        information_wait_aging_days=information_wait_aging_days,
    )
