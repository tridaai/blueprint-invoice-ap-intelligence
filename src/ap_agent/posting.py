"""Approved-not-posted reconciliation — the ERP loop's open items.

The posting-receipt loop (``service.record_posting_receipt``)
closes only when a receipt arrives. An approved invoice whose
receipt never comes — or whose only receipt says the ERP refused
the posting — is a silent break: AP believes the invoice is
handled, the ledger disagrees. This read model names those
invoices:

- Only human-approved invoices appear (only approvals post).
- An invoice leaves the report the moment a ``posted`` receipt is
  recorded; a ``failed`` receipt keeps it on the report, carrying
  the ERP's reason — a failure nobody retried is still open.
- Age runs from the gate decision (``decided_at``, stamped by the
  service), falling back to the submission and then the invoice
  date for records decided before the stamp existed; the item
  says which anchor its age uses.
- Only invoices older than the grace window are reported
  (``POSTING_RECEIPT_GRACE_DAYS``, default 7): a receipt that is
  merely in flight is not an open item yet.

It writes nothing, posts nothing, and chases nobody — like the
aging queue, it answers a daily question in work order, oldest
first.
"""

from __future__ import annotations

from datetime import date

from .matching import _parse_date
from .schemas import ApprovedNotPostedReport, OpenPostingItem
from .store import ApprovalRecord

DEFAULT_GRACE_DAYS = 7


def _anchor(record: ApprovalRecord) -> tuple[date | None, str]:
    """The date an approval's age runs from, and its basis."""
    result = record.result
    invoice = record.invoice or {}
    for value, basis in (
        (result.decided_at, "decision"),
        (invoice.get("submitted_at"), "submission"),
        (invoice.get("invoice_date"), "invoice_date"),
    ):
        parsed = _parse_date(str(value)) if value else None
        if parsed is not None:
            return parsed, basis
    return None, ""


def approved_not_posted(
    records: list[ApprovalRecord],
    *,
    as_of: date,
    grace_days: int = DEFAULT_GRACE_DAYS,
) -> ApprovedNotPostedReport:
    """The approved invoices with no posted receipt older than the
    grace window, as of a date. Pure over the records given."""
    items: list[OpenPostingItem] = []
    totals: dict[str, float] = {}
    for record in records:
        result = record.result
        if result.status != "approved":
            continue
        receipt = result.posting_receipt
        if receipt is not None and receipt.status == "posted":
            continue  # the loop closed for this invoice
        anchor, basis = _anchor(record)
        age_days = (as_of - anchor).days if anchor is not None else None
        if age_days is None or age_days <= grace_days:
            continue  # within the grace window — in flight, not open
        invoice = record.invoice or {}
        currency = str(invoice.get("currency") or result.facts.currency or "USD")
        total = invoice.get("total_amount")
        if total is not None:
            totals[currency] = round(totals.get(currency, 0.0) + float(total), 2)
        items.append(
            OpenPostingItem(
                invoice_id=result.invoice_id,
                supplier_name=result.supplier_name
                or str(invoice.get("supplier_name", "")),
                invoice_number=str(invoice.get("invoice_number", "")),
                total_amount=float(total) if total is not None else None,
                currency=currency,
                decided_by=result.decided_by or "",
                decided_at=result.decided_at or "",
                age_days=age_days,
                age_basis=basis,
                receipt_state="failed" if receipt is not None else "none",
                receipt_detail=receipt.detail if receipt is not None else "",
            )
        )
    items.sort(key=lambda item: (-(item.age_days or 0), item.invoice_id))
    return ApprovedNotPostedReport(
        as_of=as_of.isoformat(),
        grace_days=grace_days,
        count=len(items),
        totals_by_currency=totals,
        items=items,
    )
