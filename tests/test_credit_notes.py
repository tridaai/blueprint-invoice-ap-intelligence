"""Credit-note matching: a supplier credit note is tied to the
processed invoice it references, and the report says — from the
invoice's own computed match facts — whether the credit settles the
variance (creditnotes.py + service.match_credit_note)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import (
    load_sample_credit_notes,
    sample_credit_note_models,
    sample_invoice_models,
)
from ap_agent.schemas import CreditNoteInput
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


@pytest.fixture()
def service() -> InvoiceService:
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    # INV-1001 clean, INV-1002 price variance, INV-1003 short receipt.
    for invoice_id in ("INV-1001", "INV-1002", "INV-1003"):
        svc.analyze(SAMPLES[invoice_id])
    return svc


def _note(**overrides) -> CreditNoteInput:
    base = {
        "credit_note_id": "CN-T1",
        "supplier_name": "Cascade IT Hardware",
        "credit_note_number": "CN-T1",
        "currency": "USD",
        "amount": 94.40,
        "referenced_invoice_number": "CI-2210",
    }
    base.update(overrides)
    return CreditNoteInput(**base)


def test_credit_note_settles_price_variance(service):
    match = service.match_credit_note(_note())
    assert match.matched_invoice_id == "INV-1002"
    # 4 units x (270 - 250) = 80.00, grossed at the invoice's 18% tax
    # rate = 94.40 — the credit equals it exactly.
    assert match.overbilled_amount == 94.40
    assert match.status == "settles_variance"
    assert match.settles_variance is True
    assert match.remaining_variance == 0.0
    assert any("price_variance" in line for line in match.basis)


def test_credit_note_partial_settlement(service):
    match = service.match_credit_note(_note(amount=50.0))
    assert match.status == "partial_settlement"
    assert match.settles_variance is False
    assert match.remaining_variance == 44.40


def test_credit_note_exceeding_variance_is_flagged(service):
    match = service.match_credit_note(_note(amount=200.0))
    assert match.status == "exceeds_variance"
    assert match.settles_variance is False


def test_credit_note_settles_short_receipt_variance(service):
    match = service.match_credit_note(
        _note(
            credit_note_id="CN-T2",
            supplier_name="Harborline Packaging",
            amount=141.60,
            referenced_invoice_number="HP-881",
        )
    )
    assert match.matched_invoice_id == "INV-1003"
    # 20 units short x 6.00 = 120.00, grossed at 18% = 141.60.
    assert match.overbilled_amount == 141.60
    assert match.status == "settles_variance"


def test_credit_note_against_clean_invoice_has_no_variance(service):
    match = service.match_credit_note(
        _note(
            credit_note_id="CN-T3",
            supplier_name="Northbeam Office Supplies",
            amount=10.0,
            referenced_invoice_number="NB-4471",
        )
    )
    assert match.matched_invoice_id == "INV-1001"
    assert match.status == "no_variance"
    assert match.settles_variance is False


def test_credit_note_unmatched_reference(service):
    match = service.match_credit_note(_note(referenced_invoice_number="XX-000"))
    assert match.status == "unmatched"
    assert match.matched_invoice_id == ""


def test_credit_note_wrong_supplier_does_not_match(service):
    match = service.match_credit_note(_note(supplier_name="Somebody Else Ltd"))
    assert match.status == "unmatched"


def test_credit_note_currency_mismatch(service):
    match = service.match_credit_note(_note(currency="EUR"))
    assert match.status == "currency_mismatch"
    assert match.settles_variance is False


def test_credit_note_matching_changes_no_invoice_state(service):
    before = service.get("INV-1002")
    service.match_credit_note(_note())
    after = service.get("INV-1002")
    assert before is not None and after is not None
    assert after.status == before.status == "awaiting_approver"
    assert after.decision is None


def test_bundled_sample_credit_note_settles(service):
    notes = {n.credit_note_id: n for n in sample_credit_note_models()}
    match = service.match_credit_note(notes["CN-3001"])
    assert match.status == "settles_variance"
    assert match.matched_invoice_id == "INV-1002"


def test_credit_note_endpoint():
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1002"])
    client = TestClient(create_app(svc))
    response = client.post("/credit-notes/match", json=load_sample_credit_notes()[0])
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "settles_variance"
    assert body["matched_invoice_id"] == "INV-1002"
    assert body["overbilled_amount"] == 94.40
