"""Early-payment discount evaluation — parsed and priced in code.

Suppliers offer terms like "2/10 net 30": 2% off when the invoice is
paid within 10 days, the full amount otherwise due in 30. Whether
taking the discount is worth it is arithmetic, not judgement:

    gross saving   = discount % × invoice total
    cost of paying = total × cost-of-capital × (days paid early / 365)
    net saving     = gross saving − cost of paying

This module parses the terms and computes that comparison
deterministically. The model only ever quotes these numbers (in the
packet and the recommended option); it never produces them.

The annual cost of capital is configuration, not a constant baked
into code: ``COST_OF_CAPITAL_ANNUAL_PCT`` (default 6.0). The
evaluation date is the day AP received the invoice (``submitted_at``,
falling back to the invoice date) — the earliest day a pay-early
decision could act on it.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from .config import env_float
from .matching import _parse_date
from .schemas import DiscountEvaluation, InvoiceInput

_TERMS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%)?\s*/\s*(\d+)\s*,?\s*net\s*(\d+)", re.IGNORECASE
)

DEFAULT_COST_OF_CAPITAL_ANNUAL_PCT = 6.0


def parse_payment_terms(text: str | None) -> tuple[float, int, int] | None:
    """Parse "2/10 net 30" -> (2.0, 10, 30); None when the text carries
    no discount terms ("net 30" alone is a due date, not a discount)."""
    if not text:
        return None
    match = _TERMS_RE.search(str(text))
    if not match:
        return None
    return float(match.group(1)), int(match.group(2)), int(match.group(3))


def terms_from_invoice(invoice: InvoiceInput) -> str:
    """The payment-terms text for an invoice: the structured field
    first, then a ``payment_terms`` / ``terms`` document field."""
    if invoice.payment_terms:
        return invoice.payment_terms
    for doc in invoice.documents:
        for key in ("payment_terms", "terms"):
            if doc.fields.get(key):
                return doc.fields[key]
    return ""


def evaluate_discount(
    invoice: InvoiceInput,
    *,
    as_of: date | None = None,
    cost_of_capital_annual_pct: float | None = None,
) -> DiscountEvaluation | None:
    """Price the invoice's early-payment terms, or None when there is
    nothing to price (no discount terms, no total, no invoice date)."""
    parsed = parse_payment_terms(terms_from_invoice(invoice))
    if parsed is None or not invoice.total_amount:
        return None
    invoice_date = _parse_date(invoice.invoice_date)
    if invoice_date is None:
        return None
    pct, discount_days, net_days = parsed
    rate = (
        cost_of_capital_annual_pct
        if cost_of_capital_annual_pct is not None
        else env_float(
            "COST_OF_CAPITAL_ANNUAL_PCT", DEFAULT_COST_OF_CAPITAL_ANNUAL_PCT
        )
    )
    evaluation_date = as_of or _parse_date(invoice.submitted_at) or invoice_date
    deadline = invoice_date + timedelta(days=discount_days)
    due_date = _parse_date(invoice.due_date) or (invoice_date + timedelta(days=net_days))
    days_early = max(0, (due_date - evaluation_date).days)
    total = float(invoice.total_amount)
    window_open = evaluation_date <= deadline
    gross = round(total * pct / 100.0, 2) if window_open else 0.0
    capital_cost = round(total * (rate / 100.0) * days_early / 365.0, 2)
    net_saving = round(gross - capital_cost, 2)
    worthwhile = window_open and net_saving > 0
    if not window_open:
        note = (
            f"the discount window closed on {deadline.isoformat()} — before "
            "the invoice reached AP, so no discount is available"
        )
    elif worthwhile:
        note = (
            f"paying by {deadline.isoformat()} earns {gross:.2f} for a "
            f"{capital_cost:.2f} cost of capital — a net saving"
        )
    else:
        note = (
            "the discount does not cover the cost of paying early — pay "
            "on the due date instead"
        )
    return DiscountEvaluation(
        terms_text=f"{pct:g}/{discount_days} net {net_days}",
        discount_pct=pct,
        discount_days=discount_days,
        net_days=net_days,
        discount_deadline=deadline.isoformat(),
        days_early=days_early,
        gross_discount=gross,
        capital_cost=capital_cost,
        net_saving=net_saving,
        cost_of_capital_annual_pct=rate,
        worthwhile=worthwhile,
        note=note,
    )


def format_discount_line(evaluation: DiscountEvaluation, currency: str = "") -> str:
    """The one-line packet sentence: quantified, and free of any
    payment promise (the guardrails check the drafts for those)."""
    cur = f" {currency}" if currency else ""
    if evaluation.worthwhile:
        return (
            f"Early-payment discount (computed in code): terms "
            f"{evaluation.terms_text} — {evaluation.gross_discount:.2f}{cur} "
            f"discount for payment by {evaluation.discount_deadline}, less "
            f"{evaluation.capital_cost:.2f}{cur} cost of capital "
            f"({evaluation.cost_of_capital_annual_pct:g}% p.a. over "
            f"{evaluation.days_early} days) = {evaluation.net_saving:.2f}{cur} "
            "net saving. Recommendation for the approver: pay early and "
            "take the discount."
        )
    return (
        f"Early-payment discount (computed in code): terms "
        f"{evaluation.terms_text} — {evaluation.note}."
    )
