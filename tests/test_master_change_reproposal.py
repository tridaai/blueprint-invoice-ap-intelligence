"""Rejected-change re-proposal assist.

A rejected master change used to be a dead end with a note on it:
the second attempt started from memory, and nothing forced it to
answer the reason the first one failed. The assist carries the
refusal into the new proposal — decider, decision note, and the
proposer's response ride in its reason, ``reproposes`` links the
lineage — requires the response (a re-proposal that answers
nothing is a resubmission), and re-verifies the evidence instead
of copying it: a document-sourced original is re-derived from
the stored invoice against the current master.
"""

from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import (
    HandoffCreditorEntry,
    MasterChangeRequest,
    PaymentRun,
    PaymentRunLine,
    SupplierMasterRecord,
)
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore
from ap_agent.supplier_master import InMemorySupplierMaster, sample_supplier_master
from ap_agent.worklist import marker_worklist

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
NEW_ACCOUNT = "8839204716"
REJECTION_NOTE = "call-back failed — the supplier does not recognise the change"
RESPONSE = (
    "Re-called the number on file with the supplier's AP lead on "
    "the line; she confirmed the account move and the letter is "
    "countersigned this time."
)


@pytest.fixture()
def master() -> InMemorySupplierMaster:
    return InMemorySupplierMaster(sample_supplier_master())


@pytest.fixture()
def service(master) -> InvoiceService:
    return InvoiceService(
        store=InMemoryStore(), checkpointer=False, supplier_master=master
    )


def _rejected_operator_change(service) -> MasterChangeRequest:
    change = service.propose_master_change(
        SupplierMasterRecord(
            supplier_name="Newco Paper Co",
            account_number="12345678",
            sort_code="040004",
        ),
        actor="Priya Nair",
        reason="new supplier onboarding",
    )
    return service.decide_master_change(
        change.request_id,
        actor="Tomas Reid",
        approve=False,
        note=REJECTION_NOTE,
    )


def test_reproposal_carries_the_refusal_and_the_answer(service):
    original = _rejected_operator_change(service)
    again = service.repropose_master_change(
        original.request_id, actor="Priya Nair", response=RESPONSE
    )
    assert again.request_id != original.request_id
    assert again.status == "pending"
    assert again.reproposes == original.request_id
    assert again.proposed_by == "Priya Nair"
    # The refusal and the answer are both in the new attempt's
    # stated context, where the next decider reads them.
    assert f"Re-proposal of {original.request_id}" in again.reason
    assert "rejected by Tomas Reid" in again.reason
    assert REJECTION_NOTE in again.reason
    assert f"Response: {RESPONSE}" in again.reason
    # The operator's proposed record is restated; the lineage is
    # the only thing inherited from the refused attempt.
    assert again.proposed.account_number == "12345678"


def test_reproposal_requires_a_response(service):
    original = _rejected_operator_change(service)
    with pytest.raises(ValueError, match="must answer the rejection"):
        service.repropose_master_change(
            original.request_id, actor="Priya Nair", response="  "
        )


def test_only_a_rejected_change_can_be_reproposed(service):
    change = service.propose_master_change(
        SupplierMasterRecord(supplier_name="Newco Paper Co", account_number="1"),
        actor="Priya Nair",
    )
    with pytest.raises(ValueError, match="is pending"):
        service.repropose_master_change(
            change.request_id, actor="Priya Nair", response=RESPONSE
        )
    with pytest.raises(KeyError, match="Unknown master change"):
        service.repropose_master_change(
            "MCR-9999", actor="Priya Nair", response=RESPONSE
        )


def test_four_eyes_still_applies_to_the_reproposal(service):
    original = _rejected_operator_change(service)
    again = service.repropose_master_change(
        original.request_id, actor="Priya Nair", response=RESPONSE
    )
    with pytest.raises(ValueError, match="cannot also decide it"):
        service.decide_master_change(
            again.request_id,
            actor="Priya Nair",
            approve=True,
            verification_ref="call-back VR-2081",
        )
    decided = service.decide_master_change(
        again.request_id,
        actor="Tomas Reid",
        approve=True,
        verification_ref="call-back VR-2081",
    )
    assert decided.status == "approved"


def test_document_sourced_reproposal_rederives_its_evidence(service, master):
    service.analyze(SAMPLES["INV-1010"])
    original = service.propose_master_change_from_invoice(
        "INV-1010", actor="Priya Nair", reason="supplier says their bank changed"
    )
    service.decide_master_change(
        original.request_id, actor="Tomas Reid", approve=False, note=REJECTION_NOTE
    )
    # The master moves after the refusal: Harborline's terms are
    # renegotiated. A copied proposal would restate the old
    # snapshot; a re-derived one overlays the document on the
    # master as it now stands.
    record = master.get_supplier("Harborline Packaging")
    master.upsert_supplier(record.model_copy(update={"payment_terms": "net 45"}))
    again = service.repropose_master_change(
        original.request_id, actor="Priya Nair", response=RESPONSE
    )
    assert again.source == "document"
    assert again.source_invoice_id == "INV-1010"
    assert again.reproposes == original.request_id
    assert again.proposed.account_number == NEW_ACCOUNT
    assert again.proposed.payment_terms == "net 45"
    assert again.current is not None
    assert again.current.payment_terms == "net 45"
    assert REJECTION_NOTE in again.reason


def test_repropose_endpoint(service):
    original = _rejected_operator_change(service)
    client = TestClient(create_app(service))
    ok = client.post(
        f"/suppliers/master-changes/{original.request_id}/repropose",
        json={"actor": "Priya Nair", "response": RESPONSE},
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["reproposes"] == original.request_id
    assert body["status"] == "pending"
    refused = client.post(
        f"/suppliers/master-changes/{original.request_id}/repropose",
        json={"actor": "Priya Nair", "response": ""},
    )
    # The original is still the refused one, but a blank response
    # is refused first.
    assert refused.status_code == 422
    missing = client.post(
        "/suppliers/master-changes/MCR-9999/repropose",
        json={"actor": "Priya Nair", "response": RESPONSE},
    )
    assert missing.status_code == 404


def test_worklist_row_shows_the_reproposal_lineage():
    supplier = "Northbeam Office Supplies"
    run = PaymentRun(
        run_id="RUN-WL",
        currency="USD",
        status="confirmed",
        handoff_generated_at="2026-10-19T09:00:00+00:00",
        handoff_format="csv",
        invoice_count=1,
        lines=[
            PaymentRunLine(
                invoice_id="INV-WL-1",
                supplier_name=supplier,
                gross_amount=100.0,
                net_amount=100.0,
            )
        ],
        handoff_creditor_entries=[
            HandoffCreditorEntry(
                invoice_id="INV-WL-1", supplier_name=supplier, source="at_import"
            )
        ],
    )
    rejected = MasterChangeRequest(
        request_id="MCR-0001",
        supplier_name=supplier,
        proposed=SupplierMasterRecord(supplier_name=supplier, account_number="1"),
        status="rejected",
        proposed_by="Priya Nair",
        proposed_at="2026-10-18T09:00:00+00:00",
        decided_by="Tomas Reid",
        decision_note=REJECTION_NOTE,
    )
    reproposal = MasterChangeRequest(
        request_id="MCR-0002",
        supplier_name=supplier,
        proposed=SupplierMasterRecord(supplier_name=supplier, account_number="1"),
        status="pending",
        proposed_by="Priya Nair",
        proposed_at="2026-10-19T09:00:00+00:00",
        reproposes="MCR-0001",
    )
    worklist = marker_worklist(
        [run], changes=[rejected, reproposal]
    )
    item = worklist.items[0]
    assert item.onboarding_state == "in_flight"
    assert item.in_flight_change_id == "MCR-0002"
    assert item.in_flight_reproposes == "MCR-0001"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
