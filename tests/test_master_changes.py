"""Governed supplier-master changes: a bank-detail change is
proposed as a request object, decided by a second operator with an
out-of-band verification reference recorded, and only then applied
to the master — effective-dated and audited. Until it is applied,
the fraud screen keeps returning a mismatch for the new details."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import SupplierMasterRecord
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore
from ap_agent.supplier_master import InMemorySupplierMaster, sample_supplier_master

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}

# The fraud sample's note asks AP to remit to this account; the
# seeded master has 5520109873 on file for Harborline.
NEW_ACCOUNT = "8839204716"


@pytest.fixture()
def master() -> InMemorySupplierMaster:
    return InMemorySupplierMaster(sample_supplier_master())


@pytest.fixture()
def service(master) -> InvoiceService:
    return InvoiceService(
        store=InMemoryStore(), checkpointer=False, supplier_master=master
    )


def _propose_from_fraud_invoice(service) -> object:
    service.analyze(SAMPLES["INV-1010"])
    return service.propose_master_change_from_invoice(
        "INV-1010", actor="Priya Nair", reason="supplier says their bank changed"
    )


# --- proposal ------------------------------------------------------------

def test_propose_from_document_builds_the_delta(service, master):
    change = _propose_from_fraud_invoice(service)
    assert change.request_id == "MCR-0001"
    assert change.status == "pending"
    assert change.source == "document"
    assert change.source_invoice_id == "INV-1010"
    assert change.proposed_by == "Priya Nair"
    # The document's account, overlaid on the current record: terms
    # and tax id carry over, the on-file account is the snapshot.
    assert change.proposed.account_number == NEW_ACCOUNT
    assert change.proposed.sort_code == "204418"  # as stated, normalised
    assert change.proposed.payment_terms == "net 30"
    assert change.current is not None
    assert change.current.account_number == "5520109873"
    assert [e.status for e in change.history] == ["proposed"]
    # Proposing applied nothing.
    assert master.get_supplier("Harborline Packaging").account_number == "5520109873"


def test_propose_operator_change_for_a_new_supplier(service):
    change = service.propose_master_change(
        SupplierMasterRecord(
            supplier_name="Cobalt Fixtures",
            account_number="4455667788",
            sort_code="11-22-33",
        ),
        actor="Priya Nair",
        reason="new supplier onboarding",
    )
    assert change.current is None  # no master record yet
    assert change.source == "operator"
    second = service.propose_master_change(
        SupplierMasterRecord(supplier_name="Cobalt Fixtures", iban="GB29NWBK60161331926819"),
        actor="Priya Nair",
    )
    assert second.request_id == "MCR-0002"  # ids are sequential


def test_propose_validates_the_request(service):
    with pytest.raises(ValueError, match="at least one bank identifier"):
        service.propose_master_change(
            SupplierMasterRecord(supplier_name="Cobalt Fixtures"),
            actor="Priya Nair",
        )
    with pytest.raises(ValueError, match="source_invoice_id"):
        service.propose_master_change(
            SupplierMasterRecord(
                supplier_name="Cobalt Fixtures", account_number="4455667788"
            ),
            actor="Priya Nair",
            source="document",
        )
    with pytest.raises(ValueError, match="not an ISO date"):
        service.propose_master_change(
            SupplierMasterRecord(
                supplier_name="Cobalt Fixtures", account_number="4455667788"
            ),
            actor="Priya Nair",
            effective_from="next tuesday",
        )


def test_propose_from_invoice_without_bank_details_is_refused(service):
    service.analyze(SAMPLES["INV-1001"])  # states no bank details
    with pytest.raises(ValueError, match="no bank identifiers"):
        service.propose_master_change_from_invoice("INV-1001", actor="Priya Nair")
    with pytest.raises(KeyError):
        service.propose_master_change_from_invoice("INV-9999", actor="Priya Nair")


# --- the screen keeps returning mismatch until approval -------------------

def test_pending_change_keeps_the_mismatch(service):
    change = _propose_from_fraud_invoice(service)
    assert change.status == "pending"
    result = service.analyze(SAMPLES["INV-1010"])
    assert result.bank_master_check.status == "checked_mismatch"
    assert result.bank_change_blocked is True


# --- decision: four-eyes + verification -----------------------------------

def test_proposer_cannot_decide_their_own_change(service):
    change = _propose_from_fraud_invoice(service)
    with pytest.raises(ValueError, match="four-eyes"):
        service.decide_master_change(
            change.request_id,
            actor="Priya Nair",
            approve=True,
            verification_ref="call-back done",
        )


def test_approval_requires_the_verification_reference(service):
    change = _propose_from_fraud_invoice(service)
    with pytest.raises(ValueError, match="verification reference"):
        service.decide_master_change(
            change.request_id, actor="Tomas Reid", approve=True
        )
    # The failed attempt decided nothing.
    assert service.get_master_change(change.request_id).status == "pending"


def test_approval_applies_and_the_screen_turns_to_match(service, master):
    change = _propose_from_fraud_invoice(service)
    decided = service.decide_master_change(
        change.request_id,
        actor="Tomas Reid",
        approve=True,
        verification_ref="call-back to the number on file; spoke to Harborline AP",
    )
    assert decided.status == "approved"
    assert decided.decided_by == "Tomas Reid"
    assert decided.verification_ref.startswith("call-back")
    assert decided.applied_at is not None  # effective today: applied at once
    assert [e.status for e in decided.history] == ["proposed", "approved", "applied"]
    assert master.get_supplier("Harborline Packaging").account_number == NEW_ACCOUNT
    result = service.analyze(SAMPLES["INV-1010"])
    assert result.bank_master_check.status == "checked_match"
    assert result.bank_change_blocked is False
    assert result.fraud_flags == []


def test_rejection_leaves_the_master_untouched(service, master):
    change = _propose_from_fraud_invoice(service)
    decided = service.decide_master_change(
        change.request_id,
        actor="Tomas Reid",
        approve=False,
        note="call-back failed — the supplier does not recognise the change",
    )
    assert decided.status == "rejected"
    assert decided.applied_at is None
    assert master.get_supplier("Harborline Packaging").account_number == "5520109873"
    with pytest.raises(ValueError, match="already rejected"):
        service.decide_master_change(
            change.request_id, actor="Priya Nair", approve=True,
            verification_ref="call-back done",
        )


# --- effective dating ------------------------------------------------------

def test_future_dated_approval_waits_for_its_effective_date(service, master):
    change = service.propose_master_change(
        SupplierMasterRecord(
            supplier_name="Harborline Packaging",
            account_number=NEW_ACCOUNT,
            sort_code="20-44-18",
            payment_terms="net 30",
        ),
        actor="Priya Nair",
        effective_from="2099-01-01",
        reason="supplier's bank migration completes at year end",
    )
    decided = service.decide_master_change(
        change.request_id,
        actor="Tomas Reid",
        approve=True,
        verification_ref="call-back to the number on file",
    )
    assert decided.status == "approved"
    assert decided.applied_at is None  # approved, not yet in force
    assert master.get_supplier("Harborline Packaging").account_number == "5520109873"

    assert service.apply_due_master_changes(as_of="2098-12-31") == []
    applied = service.apply_due_master_changes(as_of="2099-01-01")
    assert [c.request_id for c in applied] == [change.request_id]
    assert applied[0].applied_at is not None
    assert applied[0].history[-1].actor == "system"
    assert master.get_supplier("Harborline Packaging").account_number == NEW_ACCOUNT
    # The sweep is idempotent.
    assert service.apply_due_master_changes(as_of="2099-01-02") == []


# --- API -------------------------------------------------------------------

def test_master_change_endpoints(service):
    client = TestClient(create_app(service))
    service.analyze(SAMPLES["INV-1010"])
    proposed = client.post(
        "/invoices/INV-1010/master-change",
        json={"actor": "Priya Nair", "reason": "supplier bank change note"},
    )
    assert proposed.status_code == 200
    change = proposed.json()
    assert change["status"] == "pending"
    assert change["proposed"]["account_number"] == NEW_ACCOUNT

    listed = client.get("/suppliers/master-changes", params={"status": "pending"})
    assert [c["request_id"] for c in listed.json()] == [change["request_id"]]

    self_decide = client.post(
        f"/suppliers/master-changes/{change['request_id']}/decide",
        json={"actor": "Priya Nair", "approve": True, "verification_ref": "x"},
    )
    assert self_decide.status_code == 422
    assert "four-eyes" in self_decide.json()["detail"]

    decided = client.post(
        f"/suppliers/master-changes/{change['request_id']}/decide",
        json={
            "actor": "Tomas Reid",
            "approve": True,
            "verification_ref": "call-back to the number on file",
        },
    )
    assert decided.status_code == 200
    assert decided.json()["applied_at"] is not None

    missing = client.get("/suppliers/master-changes/MCR-9999")
    assert missing.status_code == 404
    sweep = client.post("/suppliers/master-changes/apply-due")
    assert sweep.status_code == 200 and sweep.json() == {"applied": []}
