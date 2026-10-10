"""Pipeline behaviour over the bundled samples (offline backend).

The sample matrix is the blueprint's contract: every sample must
triage to its designed outcome, with the designed safety posture
(fraud hold, injection stripping, guardrail trap, gate state).
"""

from __future__ import annotations

import pytest

from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = sample_invoice_models()


@pytest.fixture()
def service():
    return InvoiceService(store=InMemoryStore(), checkpointer=False)


def _run_all(service) -> dict:
    return {inv.invoice_id: service.analyze(inv) for inv in SAMPLES}


EXPECTED_OUTCOMES = {
    "INV-1001": "clean_match",
    "INV-1002": "price_variance",
    "INV-1003": "quantity_variance",
    "INV-1004": "duplicate_suspected",
    "INV-1005": "missing_po",
    "INV-1006": "missing_receipt",
    "INV-1007": "clean_match",
    "INV-1008": "price_variance",
    "INV-1009": "unreadable",
    "INV-1010": "clean_match",
    "INV-1011": "clean_match",
    "INV-1012": "clean_match",
}


def test_sample_matrix_outcomes(service):
    results = _run_all(service)
    for invoice_id, expected in EXPECTED_OUTCOMES.items():
        assert results[invoice_id].triage.outcome == expected, invoice_id


def test_every_run_stops_at_the_gate(service):
    for result in _run_all(service).values():
        assert result.status == "awaiting_approver"
        assert result.external_action_taken is False
        assert result.drafts is not None
        trace_names = [s.name for s in result.trace]
        assert trace_names[0] == "ocr"
        assert "match" in trace_names and "approval_gate" in trace_names


def test_duplicate_only_fires_in_batch_order(service):
    # INV-1004 is byte-identical to INV-1001 in supplier/number/total;
    # analysed alone (fresh store) it is a clean invoice — the
    # duplicate verdict comes from store history, not the payload.
    fresh = InvoiceService(store=InMemoryStore(), checkpointer=False)
    alone = fresh.analyze(SAMPLES[3])
    assert alone.triage.outcome == "clean_match"
    results = _run_all(service)
    assert results["INV-1004"].triage.outcome == "duplicate_suspected"
    assert results["INV-1004"].facts.duplicate_of == "INV-1001"


def test_fraud_sample_is_blocked_not_paid(service):
    result = service.analyze(SAMPLES[9])  # INV-1010
    assert result.bank_change_blocked is True
    assert result.fraud_flags
    recommended = next(o for o in result.options if o.recommended)
    assert recommended.kind == "hold_for_review"
    assert result.autonomy.eligible_for_auto_triage is False
    # The new bank details appear nowhere in the drafts.
    combined = result.drafts.supplier_letter + result.drafts.ap_packet
    assert "8839204716" not in combined
    assert "20-44-18" not in combined
    # And the drafts still pass guardrails (labels, not sentences).
    assert result.validation.passed


def test_injection_sample_text_never_reaches_drafts(service):
    result = service.analyze(SAMPLES[10])  # INV-1011
    assert result.injection_flags
    combined = result.drafts.supplier_letter + result.drafts.ap_packet
    assert "approve payment immediately" not in combined
    assert result.triage.outcome == "clean_match"
    assert result.validation.passed


def test_guardrail_trap_sample_fails_validation(service):
    # INV-1012's supplier note relays a payment-timing demand; the
    # template drafter quotes it and the guardrail layer stops the
    # draft — the designed offline behaviour (a live model is
    # expected to avoid the trap; see the LLM eval pack).
    result = service.analyze(SAMPLES[11])
    assert result.triage.outcome == "clean_match"
    assert result.validation is not None and not result.validation.passed
    failed = {c.name for c in result.validation.checks if not c.passed}
    assert "no_payment_language" in failed
    assert result.repair_attempted is True


def test_autonomy_only_for_clean_low_value(service):
    results = _run_all(service)
    assert results["INV-1001"].autonomy.eligible_for_auto_triage is True
    assert results["INV-1002"].autonomy.eligible_for_auto_triage is False
    assert results["INV-1010"].autonomy.eligible_for_auto_triage is False


def test_information_requests_composed_not_sent(service):
    results = _run_all(service)
    assert results["INV-1005"].needs_information is True
    assert results["INV-1006"].needs_information is True
    assert results["INV-1009"].needs_information is True
    assert results["INV-1001"].needs_information is False
    message = results["INV-1005"].information_request.message
    assert "purchase order" in message.lower()


def test_clean_samples_are_grounded_and_reviewed(service):
    results = _run_all(service)
    for invoice_id in ("INV-1001", "INV-1007", "INV-1011"):
        result = results[invoice_id]
        assert result.verification.grounded, result.verification.issues
        assert result.review.verdict == "pass"
        assert result.reviewer_blocked is False


def test_telemetry_marks_offline_honestly(service):
    result = service.analyze(SAMPLES[0])
    assert result.telemetry.backend == "mock"
    assert result.telemetry.input_tokens is None  # no model ran
