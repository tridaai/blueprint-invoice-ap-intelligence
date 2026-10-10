"""The escalation digest — the day's escalations, composed.

The queue is a pull surface: an operator opens it and sees what
is stuck. Escalations deserve a push counterpart, but this agent
sends nothing — so the digest is the push, composed and left
where a human (or the customer's own scheduler, polling the
endpoint) picks it up: one deterministic summary of everything
that crossed a line, as of a day.

- **Recovery breaches** — open duplicate-payment cases whose
  current status outlasted ``RECOVERY_CASE_SLA_DAYS`` (the queue's
  SLA flags, with the case amounts summed per currency: money
  awaiting recovery).
- **Open-posting escalations** — approved invoices still unposted
  past ``POSTING_ESCALATION_DAYS`` (the queue's action items).
- **Stuck items** — invoices still at the approver gate past
  their due date (the queue's overdue items).
- **Four-eyes stalls** — large runs confirmed once and waiting on
  their second confirmer past the aging threshold.
- **The day's run news** — discounts forecast to lapse on runs
  confirmed *that day*, and drift realised on runs exported
  *that day*: the forecast and its outcome, one day apart in the
  making, side by side in the digest.

Composition is the reconciliation summary's discipline: computed
in code from the stored records, every exception named, an
explicit all-clear when nothing fired — no free-form drafting,
and the summary says plainly that the agent sent nothing.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

from .matching import _parse_date
from .schemas import (
    DigestDriftEntry,
    DigestForecastLapse,
    EscalationDigest,
    PaymentRun,
    QueueView,
)


def build_digest(
    *,
    as_of: date,
    queue: QueueView,
    runs: Iterable[PaymentRun],
) -> EscalationDigest:
    """Compose the digest for one day from the queue view (built
    for the same day by the service) and the stored payment runs.
    Pure: nothing is stored, sent, or mutated."""
    breaches = [case for case in queue.recovery_cases if case.sla_breached]
    recovery_amounts: dict[str, float] = {}
    for case in breaches:
        recovery_amounts[case.currency] = round(
            recovery_amounts.get(case.currency, 0.0) + case.amount, 2
        )
    stuck = [
        item
        for item in queue.items
        if item.days_overdue is not None and item.days_overdue > 0
    ]
    most_overdue = max(
        (item.days_overdue or 0 for item in stuck), default=None
    )

    forecast_lapses: list[DigestForecastLapse] = []
    forecast_value: dict[str, float] = {}
    drift_today: list[DigestDriftEntry] = []
    drift_value: dict[str, float] = {}
    for run in runs:
        if _parse_date(run.confirmed_at or "") == as_of and run.drift_forecast:
            for entry in run.drift_forecast:
                forecast_lapses.append(
                    DigestForecastLapse(
                        run_id=run.run_id,
                        invoice_id=entry.invoice_id,
                        supplier_name=entry.supplier_name,
                        currency=run.currency,
                        discount_amount=entry.discount_amount,
                        discount_deadline=entry.discount_deadline,
                        expected_export_date=entry.expected_export_date,
                    )
                )
                forecast_value[run.currency] = round(
                    forecast_value.get(run.currency, 0.0)
                    + entry.discount_amount,
                    2,
                )
        if (
            _parse_date(run.handoff_generated_at or "") == as_of
            and run.handoff_drift
            and run.handoff_drift_total > 0
        ):
            drift_today.append(
                DigestDriftEntry(
                    run_id=run.run_id,
                    currency=run.currency,
                    members_drifted=len(run.handoff_drift),
                    value_lost=run.handoff_drift_total,
                )
            )
            drift_value[run.currency] = round(
                drift_value.get(run.currency, 0.0) + run.handoff_drift_total, 2
            )
    forecast_lapses.sort(key=lambda e: (e.discount_deadline, e.invoice_id))
    drift_today.sort(key=lambda e: e.run_id)

    digest = EscalationDigest(
        as_of=as_of.isoformat(),
        recovery_breaches=breaches,
        recovery_breach_count=len(breaches),
        recovery_amount_by_currency=recovery_amounts,
        posting_escalations=list(queue.open_postings),
        posting_escalation_count=queue.open_posting_count,
        stuck_items=stuck,
        stuck_invoice_count=len(stuck),
        most_overdue_days=most_overdue,
        four_eyes_pending=list(queue.four_eyes_pending),
        four_eyes_pending_count=queue.four_eyes_pending_count,
        forecast_lapses=forecast_lapses,
        forecast_value_by_currency=forecast_value,
        drift_today=drift_today,
        drift_value_by_currency=drift_value,
    )
    digest.summary = _compose_summary(digest, queue)
    return digest


def _money(amounts: dict[str, float]) -> str:
    return ", ".join(
        f"{currency} {amount:,.2f}"
        for currency, amount in sorted(amounts.items())
    )


def _compose_summary(digest: EscalationDigest, queue: QueueView) -> str:
    """The operator text — composed from the computed sections,
    every escalation named, in the reconciliation summary's
    discipline. A quiet day says so; it is never padded."""
    parts = [
        f"Escalation digest for {digest.as_of} — composed from the "
        "stored records; the agent sends nothing."
    ]
    if digest.recovery_breach_count:
        named = "; ".join(
            f"{case.pack_id} ({case.supplier_name or 'supplier unknown'}, "
            f"{case.currency} {case.amount:,.2f}) stuck in "
            f"{case.case_status.replace('_', ' ')} for "
            f"{case.status_age_days}d against a {queue.recovery_sla_days}-day SLA"
            for case in digest.recovery_breaches
        )
        parts.append(
            f"Recovery cases past their SLA: {digest.recovery_breach_count} "
            f"— {named}. {_money(digest.recovery_amount_by_currency)} "
            "is awaiting recovery."
        )
    else:
        parts.append("Recovery cases past their SLA: none.")
    if digest.posting_escalation_count:
        named = "; ".join(
            f"{item.invoice_id} ({item.supplier_name or 'supplier unknown'}, "
            f"{item.currency} {(item.total_amount or 0):,.2f}) "
            f"{item.age_days}d since approval, "
            + (
                f"posting failed: {item.receipt_detail}"
                if item.receipt_state == "failed"
                else "no receipt recorded"
            )
            for item in digest.posting_escalations
        )
        parts.append(
            f"Open postings escalated past {queue.posting_escalation_days} "
            f"days: {digest.posting_escalation_count} — {named}."
        )
    else:
        parts.append("Open postings escalated: none.")
    if digest.stuck_invoice_count:
        named = "; ".join(
            f"{item.invoice_id} ({item.supplier_name or 'supplier unknown'}, "
            f"{item.currency} {(item.total_amount or 0):,.2f}) "
            f"{item.days_overdue}d overdue"
            for item in digest.stuck_items
        )
        parts.append(
            f"Invoices stuck at the gate past their due date: "
            f"{digest.stuck_invoice_count} — {named}."
        )
    else:
        parts.append("Invoices stuck at the gate: none overdue.")
    if digest.four_eyes_pending_count:
        named = "; ".join(
            f"{item.run_id} ({item.currency} {item.total_net:,.2f} net) "
            f"confirmed by {item.confirmed_by or 'an unnamed confirmer'}, "
            f"waiting {item.age_days}d for a second confirmer"
            for item in digest.four_eyes_pending
        )
        parts.append(
            f"Four-eyes runs waiting past the "
            f"{queue.four_eyes_aging_days}-day threshold: "
            f"{digest.four_eyes_pending_count} — {named}."
        )
    else:
        parts.append("Four-eyes runs waiting: none past the threshold.")
    if digest.forecast_lapses:
        named = "; ".join(
            f"{entry.invoice_id} ({entry.run_id}, {entry.currency} "
            f"{entry.discount_amount:,.2f}, deadline {entry.discount_deadline})"
            for entry in digest.forecast_lapses
        )
        parts.append(
            f"Forecast on runs confirmed today: {len(digest.forecast_lapses)} "
            f"discount(s) projected to lapse before their runs' expected "
            f"export dates — {named}. {_money(digest.forecast_value_by_currency)} "
            "at risk if the exports wait."
        )
    if digest.drift_today:
        named = "; ".join(
            f"{entry.run_id} lost {entry.currency} {entry.value_lost:,.2f} "
            f"across {entry.members_drifted} member(s)"
            for entry in digest.drift_today
        )
        parts.append(
            f"Drift realised on runs exported today: {named} — "
            f"{_money(digest.drift_value_by_currency)} of discounts lapsed "
            "between confirmation and export."
        )
    if (
        digest.recovery_breach_count == 0
        and digest.posting_escalation_count == 0
        and digest.stuck_invoice_count == 0
        and digest.four_eyes_pending_count == 0
        and not digest.forecast_lapses
        and not digest.drift_today
    ):
        parts.append(
            "All clear: nothing crossed a line today — no breaches, no "
            "escalations, no stalls, no lapses."
        )
    return "\n".join(parts)
