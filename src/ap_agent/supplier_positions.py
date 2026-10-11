"""Per-supplier open-position roll-up — the supplier-level "what do
we still owe, and what is in dispute" read model.

Computed in code from the same records the per-invoice payable
endpoint reads: each supplier row is the sum of its invoices'
payable positions (:func:`ap_agent.creditnotes.payable_position`),
so the roll-up can never disagree with the per-invoice figures —
a test pins exactly that identity. Credits an approver recorded
are netted because the positions they sum already net them.

The dispute figure comes from the supplier's **latest**
reconciliation (by statement date): the absolute amount differences
on its ``amount_differs`` lines plus the statement amounts of its
``missing_from_ap`` lines — money the two sides' records disagree
about, or that the supplier claims and AP never processed. Earlier
reconciliations are history; the position speaks for the latest.

The recovery side rides as its own lines, summed from the recovery
packs (the packs' case histories are the supplier's recovery
record): open claims — claimed, not yet received, case unclosed —
stand beside the payable figures and are never netted into them,
and the recorded-received total moves only when an operator
records a receipt on a case. Netting a disputed recovery into
what we owe is how money leaks; the operator sees gross payable,
credits applied, and recoveries claimed as separate lines.

Figures never cross currencies: a supplier billing in two
currencies gets a row per currency, and the totals are per
currency too. The roll-up is a read model — it writes nothing,
changes no payment state, and approves nothing.
"""

from __future__ import annotations

from datetime import date

from .creditnotes import payable_position
from .matching import _parse_date
from .recovery import open_claims_for_supplier, recovery_totals
from .schemas import (
    CreditApplication,
    RecoveryPack,
    StatementReconciliation,
    SupplierPosition,
    SupplierPositions,
)
from .store import ApprovalRecord


def _latest_reconciliations(
    reconciliations: list[StatementReconciliation],
) -> dict[tuple[str, str], StatementReconciliation]:
    """The latest non-refused reconciliation per (supplier,
    currency): latest by statement date, list order breaking ties
    (the store's order is the order reports were reconciled in)."""
    latest: dict[tuple[str, str], StatementReconciliation] = {}
    for report in reconciliations:
        if report.refused:
            continue
        key = (report.supplier_name, report.currency)
        current = latest.get(key)
        if current is None or (report.statement_date or "") >= (
            current.statement_date or ""
        ):
            latest[key] = report
    return latest


def _dispute(report: StatementReconciliation | None) -> tuple[float, list[str], str]:
    """(disputed amount, disputed invoice numbers, statement id)
    from one reconciliation: what differs, plus what the supplier
    lists that AP never processed."""
    if report is None:
        return 0.0, [], ""
    amount = 0.0
    numbers: list[str] = []
    for line in report.lines:
        if line.status == "amount_differs":
            amount += abs(line.difference or 0.0)
            numbers.append(line.invoice_number)
        elif line.status == "missing_from_ap":
            amount += float(line.statement_amount or 0.0)
            numbers.append(line.invoice_number)
    return round(amount, 2), numbers, report.statement_id


def supplier_positions(
    records: list[ApprovalRecord],
    applied: list[CreditApplication],
    reconciliations: list[StatementReconciliation],
    as_of: date | None = None,
    packs: list[RecoveryPack] | None = None,
) -> SupplierPositions:
    """Roll the processed invoices, recorded credits and latest
    reconciliations up into one position row per supplier per
    currency. ``as_of`` anchors the oldest-open-invoice age
    (default: today). ``packs`` — the store's recovery packs —
    supply each row's recovery lines (open claims + recorded
    receipts), keyed by the pack's own supplier + currency, so a
    supplier whose only record is a claim still appears."""
    anchor = as_of or date.today()
    latest = _latest_reconciliations(reconciliations)
    packs = packs or []
    rows: dict[tuple[str, str], SupplierPosition] = {}
    oldest: dict[tuple[str, str], tuple[int, str, str]] = {}
    for record in records:
        position = payable_position(record, applied, packs)
        key = (position.supplier_name, position.currency)
        row = rows.setdefault(
            key,
            SupplierPosition(
                supplier_name=position.supplier_name,
                currency=position.currency,
            ),
        )
        row.invoice_count += 1
        row.gross_invoiced = round(row.gross_invoiced + position.gross_amount, 2)
        row.credits_applied = round(
            row.credits_applied + position.total_credits, 2
        )
        row.open_amount = round(row.open_amount + position.open_amount, 2)
        if position.open_amount > 0:
            row.open_invoice_count += 1
            invoice = record.invoice or {}
            invoice_date = _parse_date(
                str(invoice.get("invoice_date") or invoice.get("submitted_at") or "")
            )
            if invoice_date is not None:
                age = (anchor - invoice_date).days
                current = oldest.get(key)
                if current is None or age > current[0]:
                    oldest[key] = (
                        age,
                        position.invoice_id,
                        position.invoice_number,
                    )
    # Recovery lines attach by the pack's own supplier + currency.
    # A claim whose supplier has no row (its records are gone from
    # the list handed in) still gets one — a claim must never be
    # invisible in the position because its invoices aged out.
    for pack in packs:
        key = (pack.supplier_name, pack.currency)
        if key not in rows:
            rows[key] = SupplierPosition(
                supplier_name=pack.supplier_name, currency=pack.currency
            )
    suppliers = []
    for key in sorted(rows):
        row = rows[key]
        disputed, numbers, statement_id = _dispute(latest.get(key))
        row.disputed_amount = disputed
        row.disputed_invoice_numbers = numbers
        row.dispute_statement_id = statement_id
        row.recovery_claims = open_claims_for_supplier(
            packs, row.supplier_name, row.currency
        )
        open_claimed, received = recovery_totals(
            packs, row.supplier_name, row.currency
        )
        row.recovery_open_amount = open_claimed
        row.recovery_received_amount = received
        if key in oldest:
            age, invoice_id, invoice_number = oldest[key]
            row.oldest_open_age_days = age
            row.oldest_open_invoice_id = invoice_id
            row.oldest_open_invoice_number = invoice_number
        suppliers.append(row)
    totals: dict[str, dict[str, float]] = {}
    for row in suppliers:
        bucket = totals.setdefault(
            row.currency,
            {
                "gross_invoiced": 0.0,
                "credits_applied": 0.0,
                "open_amount": 0.0,
                "disputed_amount": 0.0,
                "recovery_open_amount": 0.0,
                "recovery_received_amount": 0.0,
            },
        )
        bucket["gross_invoiced"] = round(
            bucket["gross_invoiced"] + row.gross_invoiced, 2
        )
        bucket["credits_applied"] = round(
            bucket["credits_applied"] + row.credits_applied, 2
        )
        bucket["open_amount"] = round(bucket["open_amount"] + row.open_amount, 2)
        bucket["disputed_amount"] = round(
            bucket["disputed_amount"] + row.disputed_amount, 2
        )
        bucket["recovery_open_amount"] = round(
            bucket["recovery_open_amount"] + row.recovery_open_amount, 2
        )
        bucket["recovery_received_amount"] = round(
            bucket["recovery_received_amount"] + row.recovery_received_amount, 2
        )
    return SupplierPositions(
        as_of=anchor.isoformat(),
        suppliers=suppliers,
        supplier_count=len({row.supplier_name for row in suppliers}),
        totals_by_currency=totals,
    )
