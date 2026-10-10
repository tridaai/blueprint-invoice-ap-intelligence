"""The three-way match — computed in code, never by the model.

Invoice vs purchase order vs goods receipt, line by line, with the
tolerance rules an AP department actually works to:

- **Price**: an invoice line passes when its unit price is within
  ``PRICE_TOLERANCE_PCT`` (2%) of the PO price, or within
  ``PRICE_TOLERANCE_ABS`` (1.00 currency unit) — whichever is more
  forgiving, the way most AP tolerance tables are written.
- **Quantity**: an invoice line passes when the invoiced quantity is
  no more than the ordered quantity AND no more than the received
  quantity. Invoicing more than was ordered is a quantity variance;
  invoicing more than arrived is a short-receipt exception.
- **Totals**: the stated invoice total must agree with the computed
  line total plus stated tax within the same tolerance family, and
  the stated tax must equal (stated total − computed subtotal) — a
  tax figure that does not reconcile is a mismatch in its own right.
- **Currency**: invoice, PO and receipt must agree on currency.

The same inputs always produce the same match. The model reads these
facts; it never computes them.
"""

from __future__ import annotations

from datetime import date, datetime

from .schemas import (
    DocumentExtraction,
    InvoiceInput,
    LineItem,
    LineMatch,
    MatchFacts,
)

PRICE_TOLERANCE_PCT = 0.02
PRICE_TOLERANCE_ABS = 1.00
TOTALS_TOLERANCE_PCT = 0.01
TOTALS_TOLERANCE_ABS = 1.00


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _line_key(line: LineItem) -> str:
    return _norm(line.sku) if line.sku else _norm(line.description)


def _index(lines: list[LineItem]) -> dict[str, LineItem]:
    out: dict[str, LineItem] = {}
    for line in lines:
        key = _line_key(line)
        if key and key not in out:
            out[key] = line
    return out


def _price_ok(invoice_price: float, po_price: float) -> tuple[bool, float]:
    if po_price == 0:
        return invoice_price == 0, 0.0
    variance_pct = (invoice_price - po_price) / po_price
    if abs(variance_pct) <= PRICE_TOLERANCE_PCT:
        return True, round(variance_pct * 100, 2)
    if abs(invoice_price - po_price) <= PRICE_TOLERANCE_ABS:
        return True, round(variance_pct * 100, 2)
    return False, round(variance_pct * 100, 2)


def _parse_date(s: str) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(str(s)[:10])
        except ValueError:
            return None


def _line_total(line: LineItem) -> float:
    if line.line_total is not None:
        return float(line.line_total)
    return round(line.quantity * line.unit_price, 2)


def compute_match(
    invoice: InvoiceInput,
    extractions: list[DocumentExtraction] | None = None,
    duplicate_of: str = "",
    today: date | None = None,
) -> MatchFacts:
    """Compute the three-way match facts for one invoice."""
    po = invoice.purchase_order
    grn = invoice.goods_receipt
    po_lines = _index(po.lines) if po else {}
    grn_lines = _index(grn.lines) if grn else {}

    matches: list[LineMatch] = []
    matched = 0
    for i, inv_line in enumerate(invoice.line_items, start=1):
        key = _line_key(inv_line)
        po_line = po_lines.get(key)
        grn_line = grn_lines.get(key)
        base = dict(
            line_no=i,
            description=inv_line.description,
            invoice_qty=inv_line.quantity,
            invoice_price=inv_line.unit_price,
            po_qty=po_line.quantity if po_line else 0.0,
            received_qty=grn_line.quantity if grn_line else 0.0,
            po_price=po_line.unit_price if po_line else 0.0,
        )
        if po is not None and po_line is None:
            matches.append(LineMatch(status="missing_on_po",
                                     detail="no matching line on the purchase order", **base))
            continue
        if po_line is not None:
            price_ok, var_pct = _price_ok(inv_line.unit_price, po_line.unit_price)
            base["price_variance_pct"] = var_pct
            if not price_ok:
                matches.append(LineMatch(
                    status="price_variance",
                    detail=f"unit price {inv_line.unit_price} vs PO {po_line.unit_price} ({var_pct:+.2f}%)",
                    **base))
                continue
            if inv_line.quantity > po_line.quantity:
                matches.append(LineMatch(
                    status="quantity_variance",
                    detail=f"invoiced {inv_line.quantity} vs ordered {po_line.quantity}",
                    **base))
                continue
        if grn is not None:
            if grn_line is None or inv_line.quantity > grn_line.quantity:
                received = grn_line.quantity if grn_line else 0.0
                matches.append(LineMatch(
                    status="short_received",
                    detail=f"invoiced {inv_line.quantity} vs received {received}",
                    **base))
                continue
        matches.append(LineMatch(status="matched", detail="price, quantity and receipt agree", **base))
        matched += 1

    total_lines = len(invoice.line_items)
    lines_total = round(sum(_line_total(l) for l in invoice.line_items), 2)
    stated_total = invoice.total_amount
    tax_stated = invoice.tax_amount
    tax_expected: float | None = None
    tax_variance = 0.0
    if stated_total is not None and tax_stated is not None:
        tax_expected = round(stated_total - lines_total, 2)
        tax_variance = round(tax_stated - tax_expected, 2)
    totals_variance_pct = 0.0
    if stated_total:
        expected_total = lines_total + (tax_stated or 0.0)
        totals_variance_pct = round((stated_total - expected_total) / stated_total * 100, 2) \
            if stated_total else 0.0

    if po is None:
        match_status = "no_po"
    elif grn is None:
        match_status = "no_receipt"
    elif total_lines and matched == total_lines and abs(tax_variance) <= TOTALS_TOLERANCE_ABS \
            and abs(totals_variance_pct) <= TOTALS_TOLERANCE_PCT * 100:
        match_status = "full_match"
    elif matched > 0:
        match_status = "partial_match"
    elif any(
        m.status in {"price_variance", "quantity_variance", "short_received"}
        for m in matches
    ):
        # Lines correspond to the order/receipt but disagree on price
        # or quantity — a partial match, not a non-match.
        match_status = "partial_match"
    else:
        match_status = "no_match"

    currency = (invoice.currency or "USD").upper()
    currencies = {currency}
    if po and po.currency:
        currencies.add(po.currency.upper())
    currency_consistent = len(currencies) == 1

    discrepancies: list[str] = []
    for ext in extractions or []:
        for f in ext.fields:
            if f.status == "mismatch":
                discrepancies.append(
                    f"{ext.document_id or ext.doc_type}: field '{f.field}' differs from the document text"
                )

    invoice_date = _parse_date(invoice.invoice_date)
    age = ((today or date.today()) - invoice_date).days if invoice_date else 0

    warning = ""
    if po is None:
        warning = "no purchase order attached — the three-way match cannot run; PO lookup required"
    elif grn is None:
        warning = "no goods receipt attached — receipt quantities cannot be verified"

    return MatchFacts(
        match_status=match_status,
        line_matches=matches,
        matched_lines=matched,
        total_lines=total_lines,
        match_score=round(matched / total_lines, 3) if total_lines else 0.0,
        invoice_lines_total=lines_total,
        invoice_total_stated=stated_total,
        totals_variance_pct=totals_variance_pct,
        tax_stated=tax_stated,
        tax_expected=tax_expected,
        tax_variance=tax_variance,
        currency=currency,
        currency_consistent=currency_consistent,
        has_po=po is not None,
        has_receipt=grn is not None,
        po_number=po.po_number if po else "",
        duplicate_suspected=bool(duplicate_of),
        duplicate_of=duplicate_of,
        invoice_age_days=max(0, age),
        discrepancy_count=len(discrepancies),
        discrepancies=discrepancies,
        document_pair_warning=warning,
    )


def format_match_summary(facts: MatchFacts) -> str:
    """One-line match summary for prompts, traces, and packets."""
    lines = "; ".join(
        f"L{m.line_no} {m.status}" + (f" ({m.detail})" if m.status != "matched" else "")
        for m in facts.line_matches
    ) or "no line items"
    return (
        f"match {facts.match_status}: {facts.matched_lines}/{facts.total_lines} lines matched · "
        f"lines total {facts.invoice_lines_total} {facts.currency} · "
        f"stated total {facts.invoice_total_stated} · tax variance {facts.tax_variance} · "
        f"lines: {lines}"
    )
