"""Multi-invoice bundles: one arrival, several invoices.

Real intake is batched: a supplier emails one combined scan, or a
portal drops a single PDF holding a month of invoices. The pipeline
runs one invoice at a time, so the bundle is split first — in code,
never by the model:

- Only documents typed ``invoice`` are segmented; anything else in
  the bundle (the cover email, a remittance note) is context.
- Segments are cut at invoice-number anchors — a line that opens an
  invoice block (``Invoice NB-4490``). Text before a document's
  first anchor (the supplier's letterhead) rides with its first
  segment.
- Each segment is parsed for the figures the same lines carry in
  the bundled samples — line items (``name xqty @ price = total``),
  subtotal / tax / total, dates, PO reference, currency, terms —
  and becomes a full ``InvoiceInput`` with its own document, its
  own run, and its own gate. The bundle's purchase orders and goods
  receipts attach by PO number, so a bundled invoice can still
  three-way match.
- A document with no anchor is one segment with no invoice number:
  it is processed like any arrival and triaged on its content.

The summary (service layer) counts outcomes and totals in code;
each invoice also lives in the store as its own run.
"""

from __future__ import annotations

import re

from .discounts import parse_payment_terms
from .matching import _norm
from .schemas import (
    DocumentInput,
    InvoiceBundleInput,
    InvoiceInput,
    LineItem,
)

_ANCHOR = re.compile(
    r"(?im)^[ \t]*(?:invoice|tax\s+invoice|bill)\s+"
    r"(?:no\.?|number|#)?\s*[:#]?\s*([A-Z]{1,5}-?\d{2,})\b"
)
_LINE = re.compile(
    r"(?m)^(.+?)\s+x\s*(\d+(?:\.\d+)?)\s*@\s*\$?\s*([\d,]+\.\d{2})"
    r"\s*=\s*\$?\s*([\d,]+\.\d{2})\s*$"
)
_SUBTOTAL = re.compile(r"(?im)^\s*subtotal\s*[:\-]?\s*\$?\s*([\d,]+\.\d{2})")
_TAX = re.compile(r"(?im)^\s*(?:tax|vat|mwst)\s*[:\-]?\s*\$?\s*([\d,]+\.\d{2})")
_TOTAL = re.compile(r"(?im)^\s*(?:total|gesamt)\s*[:\-]?\s*\$?\s*([\d,]+\.\d{2})")
_DATE = re.compile(r"(?im)^\s*(?:date|datum)\s*:\s*(\d{4}-\d{2}-\d{2})")
_DUE = re.compile(r"(?im)^\s*due(?:\s*date)?\s*:\s*(\d{4}-\d{2}-\d{2})")
_PO = re.compile(r"(?im)^\s*(?:po|p\.o\.|purchase\s+order)\s*[:#]?\s*([A-Z]{1,5}-?\d{2,})")
_CURRENCY = re.compile(r"\b(USD|EUR|GBP)\b")
_TERMS = re.compile(r"\d+(?:\.\d+)?\s*(?:%)?\s*/\s*\d+\s*,?\s*net\s*\d+", re.IGNORECASE)


def _money(match: re.Match | None) -> float | None:
    return float(match.group(1).replace(",", "")) if match else None


def _split_document(text: str) -> list[tuple[str, str]]:
    """Cut one document's text into (invoice_number, segment) pairs
    at the anchor lines. No anchor -> one segment, number ""."""
    anchors = list(_ANCHOR.finditer(text))
    if not anchors:
        return [("", text)] if text.strip() else []
    segments: list[tuple[str, str]] = []
    for i, anchor in enumerate(anchors):
        start = 0 if i == 0 else anchor.start()
        end = anchors[i + 1].start() if i + 1 < len(anchors) else len(text)
        segments.append((anchor.group(1), text[start:end].strip()))
    return segments


def _segment_invoice(
    bundle: InvoiceBundleInput,
    *,
    index: int,
    number: str,
    segment: str,
    source: DocumentInput,
    supplier: str,
) -> InvoiceInput:
    lines = [
        LineItem(
            description=m.group(1).strip(),
            quantity=float(m.group(2)),
            unit_price=float(m.group(3).replace(",", "")),
            line_total=float(m.group(4).replace(",", "")),
        )
        for m in _LINE.finditer(segment)
    ]
    subtotal = _money(_SUBTOTAL.search(segment))
    tax = _money(_TAX.search(segment))
    total = _money(_TOTAL.search(segment))
    date_match = _DATE.search(segment)
    due_match = _DUE.search(segment)
    po_match = _PO.search(segment)
    currency_match = _CURRENCY.search(segment)
    terms_match = _TERMS.search(segment)
    po_number = po_match.group(1) if po_match else ""
    currency = currency_match.group(1) if currency_match else bundle.currency

    fields: dict[str, str] = {}
    if number:
        fields["invoice_number"] = number
    if supplier:
        fields["supplier_name"] = supplier
    if date_match:
        fields["invoice_date"] = date_match.group(1)
    if due_match:
        fields["due_date"] = due_match.group(1)
    if po_number:
        fields["po_number"] = po_number
    if subtotal is not None:
        fields["subtotal"] = f"{subtotal:.2f}"
    if tax is not None:
        fields["tax_amount"] = f"{tax:.2f}"
    if total is not None:
        fields["total"] = f"{total:.2f}"
    if currency:
        fields["currency"] = currency

    po = next(
        (p for p in bundle.purchase_orders if p.po_number == po_number),
        None,
    )
    grn = next(
        (g for g in bundle.goods_receipts if g.po_number == po_number),
        None,
    )
    if po is not None:
        # Segment text carries no SKUs, but the matcher keys lines by
        # SKU when the PO has them. Map each segment line to the PO
        # line with the same normalised description and adopt its
        # SKU — a deterministic intake mapping, like a real AP
        # system's supplier-item catalogue.
        sku_by_description = {
            _norm(line.description): line.sku for line in po.lines if line.sku
        }
        for line in lines:
            if not line.sku:
                line.sku = sku_by_description.get(_norm(line.description), "")
    return InvoiceInput(
        invoice_id=f"{bundle.bundle_id}-{index:02d}",
        supplier_name=supplier,
        supplier_email_text=bundle.supplier_email_text,
        notes=bundle.notes,
        invoice_number=number,
        invoice_date=date_match.group(1) if date_match else "",
        due_date=due_match.group(1) if due_match else "",
        payment_terms=(
            terms_match.group(0)
            if terms_match and parse_payment_terms(terms_match.group(0))
            else ""
        ),
        currency=currency,
        tax_amount=tax,
        total_amount=total,
        submitted_at=bundle.submitted_at,
        line_items=lines,
        documents=[
            DocumentInput(
                doc_type="invoice",
                document_id=f"{bundle.bundle_id}-DOC-{index:02d}",
                ocr_text=segment,
                ocr_confidence=source.ocr_confidence,
                scan_quality=source.scan_quality,
                fields=fields,
            )
        ],
        purchase_order=po,
        goods_receipt=grn,
    )


def segment_bundle(
    bundle: InvoiceBundleInput,
) -> tuple[list[InvoiceInput], str]:
    """Split a bundle into per-invoice inputs, in arrival order.

    Returns the invoices and a note describing the split (documents
    segmented vs treated as context) for the bundle summary.
    """
    supplier = bundle.supplier_name
    invoices: list[InvoiceInput] = []
    segmented_docs = 0
    context_docs = 0
    for doc in bundle.documents:
        if doc.doc_type != "invoice":
            context_docs += 1
            continue
        segments = _split_document(doc.text_layer)
        if not segments:
            context_docs += 1
            continue
        segmented_docs += 1
        if not supplier and segments:
            first_line = next(
                (ln.strip() for ln in segments[0][1].splitlines() if ln.strip()),
                "",
            )
            if first_line and not _ANCHOR.match(first_line):
                supplier = first_line
        for number, segment in segments:
            invoices.append(
                _segment_invoice(
                    bundle,
                    index=len(invoices) + 1,
                    number=number,
                    segment=segment,
                    source=doc,
                    supplier=supplier,
                )
            )
    note = (
        f"{segmented_docs} invoice document(s) segmented into "
        f"{len(invoices)} invoice(s)"
        + (
            f"; {context_docs} document(s) treated as context"
            if context_docs
            else ""
        )
        + ". Each invoice runs the full pipeline and waits at its "
        "own approver gate."
    )
    return invoices, note
