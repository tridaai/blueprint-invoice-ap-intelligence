"""Credit-note matching — computed in code over the processed invoices.

A credit note arrives after the invoice it belongs to: the supplier
concedes a price difference, a short shipment, an over-billing. AP's
question is narrow and factual: *does this credit settle a variance
we actually computed?* This module answers it from the stored match
facts — never from the note's own prose:

- The note is matched to a processed invoice by supplier + the
  supplier invoice number it references.
- The invoice's overbilled amount is recomputed from its line
  matches: price variances at (invoiced − PO) price, quantity
  variances and short receipts at the invoiced price, lines missing
  from the PO in full — grossed up at the invoice's own effective
  tax rate, so the figure is comparable to a tax-inclusive credit.
- The credit settles the variance when it equals that amount (within
  the tolerance family the match itself uses); below it is a partial
  settlement, above it exceeds the variance, and a note against a
  clean invoice has no variance to settle.

Matching is a report. It approves nothing, pays nothing, and changes
no invoice's state — an approver acts on it. Acting has its own step:
an approver records the credit against the invoice (the service's
``apply_credit_note``), and from then on the invoice's payable
position — its processed total less recorded credits, computed by
:func:`payable_position` — is what payment proposals net from.
Evaluation is application-aware: credits already recorded against
an invoice reduce the variance a later note can settle, and a note
that was already recorded reports ``already_applied`` instead of
settling twice. Two caps are enforced in code on the write path:
recorded credits never exceed the invoice's remaining computed
variance, and never exceed its open amount.
"""

from __future__ import annotations

from .schemas import (
    CreditApplication,
    CreditNoteInput,
    CreditNoteMatch,
    PayablePosition,
)
from .store import ApprovalRecord

# Settlement tolerance: the larger of a cent or 1% of the variance.
_SETTLE_ABS = 0.01
_SETTLE_PCT = 0.01


def overbilled_amount(record: ApprovalRecord) -> tuple[float, list[str]]:
    """(Overbilled amount incl. tax effect, per-line basis) for one
    processed invoice, recomputed from its stored match facts."""
    facts = record.result.facts
    lines_total = facts.invoice_lines_total or 0.0
    tax_rate = (
        float(facts.tax_stated) / lines_total
        if facts.tax_stated and lines_total
        else 0.0
    )
    pre_tax = 0.0
    basis: list[str] = []
    for m in facts.line_matches:
        if m.status == "price_variance":
            over = m.invoice_qty * (m.invoice_price - m.po_price)
            why = (
                f"invoiced {m.invoice_qty:g} @ {m.invoice_price} vs PO "
                f"{m.po_price}"
            )
        elif m.status == "quantity_variance":
            over = (m.invoice_qty - m.po_qty) * m.invoice_price
            why = f"invoiced {m.invoice_qty:g} vs ordered {m.po_qty:g}"
        elif m.status == "short_received":
            over = (m.invoice_qty - m.received_qty) * m.invoice_price
            why = f"invoiced {m.invoice_qty:g} vs received {m.received_qty:g}"
        elif m.status == "missing_on_po":
            over = m.invoice_qty * m.invoice_price
            why = "no matching line on the purchase order"
        else:
            continue
        if over <= 0:
            continue
        pre_tax += over
        basis.append(f"L{m.line_no} {m.description}: {m.status} — {why} = {over:.2f}")
    if pre_tax <= 0:
        return 0.0, basis
    total = round(pre_tax * (1.0 + tax_rate), 2)
    if tax_rate:
        basis.append(
            f"tax effect at the invoice's effective rate {tax_rate:.2%}: "
            f"{round(total - pre_tax, 2):.2f}"
        )
    return total, basis


def evaluate_credit_note(
    note: CreditNoteInput,
    records: list[ApprovalRecord],
    applied: list[CreditApplication] | None = None,
) -> CreditNoteMatch:
    """Match one credit note against the processed invoice records.

    ``applied`` is the store's recorded credit applications: credits
    already recorded against the matched invoice reduce the variance
    this note can settle, and a note whose id was already recorded
    reports ``already_applied``. With no applications the evaluation
    is exactly the historical one."""
    report = CreditNoteMatch(
        credit_note_id=note.credit_note_id,
        credit_note_number=note.credit_note_number,
        supplier_name=note.supplier_name,
        referenced_invoice_number=note.referenced_invoice_number,
        credit_amount=float(note.amount or 0.0),
    )
    candidates = [
        r
        for r in records
        if note.referenced_invoice_number
        and r.invoice.get("invoice_number") == note.referenced_invoice_number
        and (
            not note.supplier_name
            or r.result.supplier_name == note.supplier_name
        )
    ]
    if not candidates:
        report.note = (
            "no processed invoice matches the referenced supplier invoice "
            "number — the note cannot be tied to a variance we computed"
        )
        return report
    record = candidates[0]
    report.matched_invoice_id = record.result.invoice_id
    applied = applied or []
    invoice_applications = [
        a for a in applied if a.invoice_id == record.result.invoice_id
    ]
    applied_total = round(sum(a.amount for a in invoice_applications), 2)
    report.applied_credits_total = applied_total
    already = next(
        (a for a in invoice_applications if a.credit_note_id == note.credit_note_id),
        None,
    )
    invoice_currency = str(record.invoice.get("currency") or "").upper()
    if note.currency and invoice_currency and note.currency.upper() != invoice_currency:
        report.status = "currency_mismatch"
        report.note = (
            f"the note is in {note.currency.upper()} but invoice "
            f"{record.result.invoice_id} is in {invoice_currency} — amounts "
            "are not compared across currencies"
        )
        return report
    overbilled, basis = overbilled_amount(record)
    report.overbilled_amount = overbilled
    report.basis = basis
    credit = report.credit_amount
    if already is not None:
        report.status = "already_applied"
        report.remaining_variance = round(max(0.0, overbilled - applied_total), 2)
        report.note = (
            f"this credit note was already recorded against "
            f"{record.result.invoice_id} by {already.applied_by or 'an approver'}"
            f" on {already.applied_at or 'an earlier date'} — it is not "
            "counted twice"
        )
        return report
    # Credits already recorded settle part of the variance first:
    # this note can only settle what remains.
    overbilled = round(overbilled - applied_total, 2)
    if overbilled <= 0.005:
        report.status = "no_variance"
        if applied_total > 0:
            report.note = (
                f"the computed variance on {record.result.invoice_id} is "
                f"already settled by {applied_total:.2f} of recorded "
                "credits — nothing remains for this credit to settle"
            )
        else:
            report.note = (
                f"invoice {record.result.invoice_id} was not overbilled — its "
                "match computed no variance for this credit to settle"
            )
        return report
    tolerance = max(_SETTLE_ABS, round(_SETTLE_PCT * overbilled, 2))
    difference = round(credit - overbilled, 2)
    if abs(difference) <= tolerance:
        report.status = "settles_variance"
        report.settles_variance = True
        report.remaining_variance = 0.0
        report.note = (
            f"the credit settles the computed variance on "
            f"{record.result.invoice_id} in full"
        )
    elif difference < 0:
        report.status = "partial_settlement"
        report.remaining_variance = round(overbilled - credit, 2)
        report.note = (
            f"the credit covers part of the variance on "
            f"{record.result.invoice_id}; {report.remaining_variance:.2f} "
            "remains open"
        )
    else:
        report.status = "exceeds_variance"
        report.note = (
            f"the credit exceeds the computed variance on "
            f"{record.result.invoice_id} by {difference:.2f} — an approver "
            "should ask why before accepting it"
        )
    return report


def payable_position(
    record: ApprovalRecord, applied: list[CreditApplication]
) -> PayablePosition:
    """One invoice's payable position: its processed total less the
    credits recorded against it. Computed, never narrated — this is
    the figure payment proposals net from."""
    invoice = record.invoice or {}
    gross = round(float(invoice.get("total_amount") or 0.0), 2)
    credits = [
        a for a in applied if a.invoice_id == record.result.invoice_id
    ]
    total_credits = round(sum(a.amount for a in credits), 2)
    return PayablePosition(
        invoice_id=record.result.invoice_id,
        invoice_number=str(invoice.get("invoice_number", "")),
        supplier_name=record.result.supplier_name
        or str(invoice.get("supplier_name", "")),
        currency=str(invoice.get("currency") or "USD"),
        gross_amount=gross,
        credits=credits,
        total_credits=total_credits,
        open_amount=round(max(0.0, gross - total_credits), 2),
    )
