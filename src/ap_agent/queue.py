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
"""

from __future__ import annotations

from datetime import date

from .matching import _parse_date
from .schemas import QueueItem, QueueView
from .store import ApprovalRecord

_SEVERITY_RANK = {"high": 0, "moderate": 1, "low": 2}


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
    records: list[ApprovalRecord], *, as_of: date | None = None
) -> QueueView:
    """The aging queue over the store's records, as of a date
    (today when not given — tests and API callers pass it
    explicitly for a stable view)."""
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
    return QueueView(
        as_of=as_of.isoformat(),
        count=len(items),
        overdue_count=overdue_count,
        triage_mix=triage_mix,
        totals_by_currency=totals,
        buckets=buckets,
        items=items,
    )
