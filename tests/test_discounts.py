"""Early-payment discount evaluation: terms parsed in code, the
saving priced in code, the approve option's score and wording
carrying the quantified result (options.py + discounts.py)."""

from __future__ import annotations

from datetime import date

from ap_agent.discounts import evaluate_discount, parse_payment_terms
from ap_agent.options import score_option
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import InvoiceInput
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore


def _invoice(**overrides) -> InvoiceInput:
    base = {
        "invoice_id": "T-1",
        "supplier_name": "Terms Supplier",
        "invoice_number": "TS-1",
        "invoice_date": "2026-09-01",
        "due_date": "2026-10-01",
        "submitted_at": "2026-09-02T09:00:00",
        "total_amount": 10000.0,
        "payment_terms": "2/10 net 30",
    }
    base.update(overrides)
    return InvoiceInput(**base)


def test_parse_payment_terms_variants():
    assert parse_payment_terms("2/10 net 30") == (2.0, 10, 30)
    assert parse_payment_terms("Terms: 1.5/15 NET 45") == (1.5, 15, 45)
    assert parse_payment_terms("2/10, net 30") == (2.0, 10, 30)
    assert parse_payment_terms("net 30") is None  # a due date, not a discount
    assert parse_payment_terms("") is None
    assert parse_payment_terms(None) is None


def test_evaluate_discount_prices_the_saving():
    ev = evaluate_discount(_invoice())
    assert ev is not None
    assert ev.terms_text == "2/10 net 30"
    assert ev.discount_deadline == "2026-09-11"  # invoice date + 10 days
    assert ev.days_early == 29  # due 2026-10-01 minus received 2026-09-02
    assert ev.gross_discount == 200.0  # 2% of 10000
    # capital: 10000 * 6% * 29/365 = 47.67
    assert ev.capital_cost == 47.67
    assert ev.net_saving == 152.33
    assert ev.worthwhile is True


def test_evaluate_discount_window_closed():
    ev = evaluate_discount(_invoice(), as_of=date(2026, 9, 20))
    assert ev is not None
    assert ev.gross_discount == 0.0  # deadline was 2026-09-11
    assert ev.worthwhile is False
    assert "closed" in ev.note


def test_evaluate_discount_not_worthwhile_at_high_cost_of_capital():
    ev = evaluate_discount(_invoice(), cost_of_capital_annual_pct=40.0)
    assert ev is not None
    assert ev.net_saving < 0
    assert ev.worthwhile is False


def test_evaluate_discount_absent_without_terms_or_total():
    assert evaluate_discount(_invoice(payment_terms="")) is None
    assert evaluate_discount(_invoice(payment_terms="net 30")) is None
    assert evaluate_discount(_invoice(total_amount=None)) is None


def test_terms_can_come_from_a_document_field():
    inv = _invoice(
        payment_terms="",
        documents=[
            {
                "doc_type": "invoice",
                "document_id": "D-1",
                "fields": {"payment_terms": "1/10 net 30"},
            }
        ],
    )
    ev = evaluate_discount(inv)
    assert ev is not None and ev.discount_pct == 1.0


def test_score_option_discount_term_only_helps_proceeding():
    triage = {"outcome": "clean_match", "severity": "low"}
    facts = {"match_status": "full_match", "total_amount": 1180.0}
    plain = score_option("approve_for_payment_prep", triage, facts)
    with_discount = score_option(
        "approve_for_payment_prep",
        triage,
        {**facts, "discount": {"worthwhile": True, "net_saving": 19.72}},
    )
    assert with_discount == round(plain + 19.72, 2)
    # the term never lifts a non-proceeding kind
    hold = score_option("hold_for_review", triage, facts)
    hold_discounted = score_option(
        "hold_for_review",
        triage,
        {**facts, "discount": {"worthwhile": True, "net_saving": 19.72}},
    )
    assert hold_discounted == hold


def test_sample_inv_1001_carries_a_priced_discount_end_to_end():
    service = InvoiceService(store=InMemoryStore(), checkpointer=False)
    sample = next(s for s in sample_invoice_models() if s.invoice_id == "INV-1001")
    result = service.analyze(sample)
    assert result.triage.outcome == "clean_match"
    assert result.discount is not None
    assert result.discount.terms_text == "2/10 net 30"
    assert result.discount.discount_deadline == "2026-10-08"
    assert result.discount.gross_discount == 23.60  # 2% of 1180
    # capital: received 2026-09-29, due 2026-10-28 -> 29 days early;
    # 1180 * 6% * 29/365 = 5.63; net = 23.60 - 5.63
    assert result.discount.days_early == 29
    assert result.discount.capital_cost == 5.63
    assert result.discount.net_saving == 17.97
    assert result.discount.worthwhile is True
    # the saving is quantified in the packet and in the recommended option
    assert result.drafts is not None
    assert "Early-payment discount" in result.drafts.ap_packet
    assert "17.97" in result.drafts.ap_packet
    recommended = next(o for o in result.options if o.recommended)
    assert recommended.kind == "approve_for_payment_prep"
    assert "net saving" in recommended.description
