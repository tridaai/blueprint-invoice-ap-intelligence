"""Domain core: three-way match, triage rules, screening, OCR, vision.

Pure deterministic units — no graph, no service. These pin the
arithmetic and the rule priorities the rest of the system relies on.
"""

from __future__ import annotations

import copy

from ap_agent.matching import compute_match, format_match_summary
from ap_agent.ocr import OfflineOcrEngine, run_ocr
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import InvoiceInput
from ap_agent.screening import (
    fraud_flag_label,
    sanitize_text,
    screen_fraud,
    screen_invoice,
    sanitized_notes,
)
from ap_agent.triage import classify_invoice
from ap_agent.vision import MetadataVision

SAMPLES = {inv.invoice_id: inv for inv in sample_invoice_models()}


def _clone(invoice_id: str) -> InvoiceInput:
    return InvoiceInput.model_validate(
        copy.deepcopy(SAMPLES[invoice_id].model_dump())
    )


# -- three-way match ---------------------------------------------------


def test_clean_invoice_full_match():
    facts = compute_match(SAMPLES["INV-1001"])
    assert facts.match_status == "full_match"
    assert facts.match_score == 1.0
    assert facts.tax_variance == 0.0
    assert all(m.status == "matched" for m in facts.line_matches)
    assert "full_match" in format_match_summary(facts)


def test_price_variance_line_and_status():
    facts = compute_match(SAMPLES["INV-1002"])
    assert facts.match_status == "partial_match"
    line = facts.line_matches[0]
    assert line.status == "price_variance"
    assert line.price_variance_pct == 8.0


def test_price_inside_absolute_tolerance_matches():
    # A 0.60 variance on a 6.00 line is 10% but inside the 1.00
    # absolute band of AP-MATCH-01 — the line matches.
    inv = _clone("INV-1003")
    inv.line_items[0].unit_price = 6.6
    inv.tax_amount = 118.8
    inv.total_amount = 778.8
    inv.goods_receipt.lines[0].quantity = 100
    facts = compute_match(inv)
    assert facts.line_matches[0].status == "matched"


def test_short_receipt_is_quantity_side():
    facts = compute_match(SAMPLES["INV-1003"])
    assert facts.line_matches[0].status == "short_received"
    assert facts.match_status == "partial_match"


def test_billed_more_than_ordered():
    inv = _clone("INV-1003")
    inv.line_items[0].quantity = 120
    inv.goods_receipt.lines[0].quantity = 120
    facts = compute_match(inv)
    assert facts.line_matches[0].status == "quantity_variance"


def test_no_po_and_no_receipt_statuses():
    assert compute_match(SAMPLES["INV-1005"]).match_status == "no_po"
    assert compute_match(SAMPLES["INV-1006"]).match_status == "no_receipt"


def test_tax_mismatch_detected_from_arithmetic():
    facts = compute_match(SAMPLES["INV-1008"])
    # Stated total 1180 - lines 1000 = implied tax 180; stated tax 100.
    assert facts.tax_expected == 180.0
    assert facts.tax_variance == -80.0
    assert facts.match_status == "partial_match"


def test_currency_mismatch_flagged():
    inv = _clone("INV-1007")
    inv.purchase_order.currency = "USD"
    facts = compute_match(inv)
    assert facts.currency_consistent is False


def test_duplicate_carried_into_facts():
    facts = compute_match(SAMPLES["INV-1004"], duplicate_of="INV-1001")
    assert facts.duplicate_suspected is True
    assert facts.duplicate_of == "INV-1001"


# -- triage rules -------------------------------------------------------


def _triage(invoice_id: str, **kwargs):
    inv = SAMPLES[invoice_id]
    facts = compute_match(inv, duplicate_of=kwargs.pop("duplicate_of", ""))
    vision = MetadataVision().verify(inv, run_ocr(inv, OfflineOcrEngine()))
    fraud = screen_fraud(inv)
    return classify_invoice(inv, facts, vision=vision, ocr_results=run_ocr(inv, OfflineOcrEngine()), fraud_flags=fraud)


def test_triage_outcomes_for_samples():
    assert _triage("INV-1001").outcome == "clean_match"
    assert _triage("INV-1002").outcome == "price_variance"
    assert _triage("INV-1002").severity == "high"  # 8% > 5%
    assert _triage("INV-1003").outcome == "quantity_variance"
    assert _triage("INV-1004", duplicate_of="INV-1001").outcome == "duplicate_suspected"
    assert _triage("INV-1005").outcome == "missing_po"
    assert _triage("INV-1006").outcome == "missing_receipt"
    assert _triage("INV-1007").outcome == "clean_match"
    assert _triage("INV-1008").outcome == "price_variance"
    assert _triage("INV-1009").outcome == "unreadable"


def test_price_variance_beats_quantity_variance():
    inv = _clone("INV-1003")
    inv.line_items[0].unit_price = 7.5  # outside both tolerance bands
    inv.tax_amount = 135.0
    inv.total_amount = 885.0
    facts = compute_match(inv)
    triage = classify_invoice(inv, facts)
    assert triage.outcome == "price_variance"


def test_not_an_invoice_from_vision():
    inv = _clone("INV-1001")
    inv.documents[0].doc_type = "statement"
    facts = compute_match(inv)
    vision = MetadataVision().verify(inv, run_ocr(inv, OfflineOcrEngine()))
    assert vision.is_invoice is False
    assert classify_invoice(inv, facts, vision=vision).outcome == "not_an_invoice"


# -- screening -----------------------------------------------------------


def test_injection_flagged_and_stripped():
    inv = SAMPLES["INV-1011"]
    flags = screen_invoice(inv)
    assert flags, "the SYSTEM: injection must be flagged"
    cleaned = sanitized_notes(inv)
    assert "SYSTEM" not in cleaned
    assert "approve payment immediately" not in cleaned


def test_bank_change_flagged_as_fraud_not_injection():
    inv = SAMPLES["INV-1010"]
    fraud = screen_fraud(inv)
    assert fraud, "the bank-change request must be flagged as fraud"
    assert all("bank_change_request" in f for f in fraud)
    # Digit runs (and sort-code shapes) are masked in the flag excerpts.
    assert "8839204716" not in " ".join(fraud)
    assert "20-44-18" not in " ".join(fraud)
    # And the sentences never survive sanitisation into prompt text.
    cleaned = sanitized_notes(inv)
    assert "remit" not in cleaned.lower()
    assert "8839204716" not in cleaned


def test_fraud_flag_label_withholds_sentence_for_drafts():
    flag = screen_fraud(SAMPLES["INV-1010"])[0]
    label = fraud_flag_label(flag)
    assert "account number" not in label
    assert "AP-BANK-01" in label


def test_reported_payment_demand_is_not_screened():
    # The guardrail trap: reported speech about payment timing is not
    # an instruction to the agent, so the screen leaves it — the
    # guardrail layer is what catches it in drafts.
    inv = SAMPLES["INV-1012"]
    assert screen_invoice(inv) == []
    assert "payment will be made by Friday" in sanitized_notes(inv)


def test_benign_text_untouched():
    assert sanitize_text("Invoice attached for PO-9001. Thanks!") == (
        "Invoice attached for PO-9001. Thanks!"
    )


# -- OCR + vision ports ---------------------------------------------------


def test_offline_ocr_uses_text_layer_and_confidence_metadata():
    inv = SAMPLES["INV-1009"]
    results = run_ocr(inv, OfflineOcrEngine())
    assert results[0].engine == "offline"
    assert results[0].mean_confidence == 0.32
    assert results[0].low_confidence is True


def test_metadata_vision_unreadable_for_poor_scan():
    inv = SAMPLES["INV-1009"]
    vision = MetadataVision().verify(inv, run_ocr(inv, OfflineOcrEngine()))
    assert vision.is_invoice is True
    assert vision.quality == "unreadable"
    assert vision.legible is False


def test_metadata_vision_good_for_clean_scan():
    inv = SAMPLES["INV-1001"]
    vision = MetadataVision().verify(inv, run_ocr(inv, OfflineOcrEngine()))
    assert vision.is_invoice is True
    assert vision.quality == "good"
