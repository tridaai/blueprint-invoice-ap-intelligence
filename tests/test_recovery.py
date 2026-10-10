"""Duplicate-payment recovery packs: when duplicate detection fires
on an invoice whose original was already approved, the agent drafts
the recovery case — computed evidence, a supplier recovery letter,
an internal memo — through the standard verify + guardrail path.
It recovers nothing itself (recovery.py + service)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


@pytest.fixture()
def service() -> InvoiceService:
    """INV-1001 approved; its portal resend INV-1013 analysed after
    (duplicate detection ties it to INV-1001)."""
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    svc.analyze(SAMPLES["INV-1013"])
    return svc


def test_recovery_pack_is_drafted_for_approved_twin(service):
    pack = service.draft_duplicate_recovery("INV-1013")
    assert pack.pack_id == "REC-INV-1013"
    assert pack.original_invoice_id == "INV-1001"
    assert pack.supplier_name == "Northbeam Office Supplies"
    assert pack.invoice_number == "NB-4471"
    assert pack.amount == 1180.0
    assert pack.exposure_status == "twin_approved"
    assert pack.status == "ready"


def test_recovery_evidence_is_computed_from_both_records(service):
    pack = service.draft_duplicate_recovery("INV-1013")
    joined = "\n".join(pack.evidence)
    assert "NB-4471" in joined
    assert "1180.00" in joined
    assert "approved by Priya Nair" in joined
    assert "one payment in flight" in joined


def test_recovery_drafts_pass_the_standard_guardrails(service):
    pack = service.draft_duplicate_recovery("INV-1013")
    assert pack.validation is not None and pack.validation.passed
    assert pack.verification is not None and pack.verification.grounded
    letter = pack.drafts.supplier_letter
    assert "credit note" in letter
    assert "INV-1013" in letter and "INV-1001" in letter
    memo = pack.drafts.ap_packet
    assert "reject" in memo.lower()  # the stop instruction for the copy
    assert "[AP-DUP-01]" in letter + memo


# --- the case lifecycle: opened → supplier_contacted → credit_received → closed ---

def test_case_opens_with_the_pack(service):
    pack = service.draft_duplicate_recovery("INV-1013")
    assert pack.case_status == "opened"
    assert len(pack.case_history) == 1
    opened = pack.case_history[0]
    assert opened.status == "opened" and opened.at == pack.created_at


def test_case_walks_the_lifecycle_with_an_audit_trail(service):
    service.draft_duplicate_recovery("INV-1013")
    pack = service.transition_recovery_case(
        "INV-1013", to_status="supplier_contacted",
        actor="Priya Nair", note="letter sent via supplier portal",
    )
    assert pack.case_status == "supplier_contacted"
    pack = service.transition_recovery_case(
        "INV-1013", to_status="credit_received", actor="Priya Nair"
    )
    pack = service.transition_recovery_case(
        "INV-1013", to_status="closed", actor="Tomas Reid", note="credit CN-88 on file"
    )
    assert pack.case_status == "closed"
    history = service.get_recovery_pack("INV-1013").case_history
    assert [h.status for h in history] == [
        "opened", "supplier_contacted", "credit_received", "closed",
    ]
    assert history[1].actor == "Priya Nair"
    assert "supplier portal" in history[1].note
    assert history[3].actor == "Tomas Reid"


def test_case_cannot_skip_or_go_backwards(service):
    service.draft_duplicate_recovery("INV-1013")
    with pytest.raises(ValueError, match="next step is supplier_contacted"):
        service.transition_recovery_case(
            "INV-1013", to_status="credit_received", actor="Priya Nair"
        )
    with pytest.raises(ValueError, match="Unknown case status"):
        service.transition_recovery_case(
            "INV-1013", to_status="refunded", actor="Priya Nair"
        )
    for step in ("supplier_contacted", "credit_received", "closed"):
        service.transition_recovery_case("INV-1013", to_status=step, actor="P")
    with pytest.raises(ValueError, match=r"none \(the case is closed\)"):
        service.transition_recovery_case(
            "INV-1013", to_status="closed", actor="Priya Nair"
        )


def test_case_transition_requires_a_pack(service):
    with pytest.raises(KeyError):
        service.transition_recovery_case(
            "INV-1001", to_status="supplier_contacted", actor="Priya Nair"
        )


def test_case_transition_endpoints(service):
    client = TestClient(create_app(service))
    client.post("/invoices/INV-1013/recovery-pack")
    moved = client.post(
        "/invoices/INV-1013/recovery-case/transition",
        json={"actor": "Priya Nair", "to_status": "supplier_contacted"},
    )
    assert moved.status_code == 200
    assert moved.json()["case_status"] == "supplier_contacted"
    skip = client.post(
        "/invoices/INV-1013/recovery-case/transition",
        json={"actor": "Priya Nair", "to_status": "closed"},
    )
    assert skip.status_code == 422
    unknown = client.post(
        "/invoices/INV-9999/recovery-case/transition",
        json={"actor": "Priya Nair", "to_status": "supplier_contacted"},
    )
    assert unknown.status_code == 404
    no_actor = client.post(
        "/invoices/INV-1013/recovery-case/transition",
        json={"to_status": "credit_received"},
    )
    assert no_actor.status_code == 422


def test_recovery_is_idempotent_and_changes_no_invoice_state(service):
    first = service.draft_duplicate_recovery("INV-1013")
    second = service.draft_duplicate_recovery("INV-1013")
    assert second.pack_id == first.pack_id
    assert second.created_at == first.created_at
    duplicate = service.get("INV-1013")
    original = service.get("INV-1001")
    assert duplicate is not None and duplicate.status == "awaiting_approver"
    assert original is not None and original.status == "approved"
    assert original.external_action_taken is False


def test_recovery_refused_when_twin_not_approved():
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])  # never approved
    svc.analyze(SAMPLES["INV-1013"])
    with pytest.raises(ValueError, match="not approved"):
        svc.draft_duplicate_recovery("INV-1013")


def test_recovery_refused_for_non_duplicate(service):
    with pytest.raises(ValueError, match="not a suspected duplicate"):
        service.draft_duplicate_recovery("INV-1001")
    with pytest.raises(KeyError):
        service.draft_duplicate_recovery("INV-9999")


def test_both_approved_is_the_worst_case_exposure(service):
    service.approve("INV-1013", approver="Tomas Reid", reason="missed the flag")
    pack = service.draft_duplicate_recovery("INV-1013")
    assert pack.exposure_status == "both_approved"
    assert "two payments are in flight" in "\n".join(pack.evidence)
    assert "escalate" in pack.drafts.ap_packet.lower()
    assert pack.validation is not None and pack.validation.passed


def test_recovery_pack_endpoints():
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair")
    svc.analyze(SAMPLES["INV-1013"])
    client = TestClient(create_app(svc))
    missing = client.get("/invoices/INV-1013/recovery-pack")
    assert missing.status_code == 404
    drafted = client.post("/invoices/INV-1013/recovery-pack")
    assert drafted.status_code == 200
    body = drafted.json()
    assert body["pack_id"] == "REC-INV-1013"
    assert body["status"] == "ready"
    fetched = client.get("/invoices/INV-1013/recovery-pack")
    assert fetched.status_code == 200
    refused = client.post("/invoices/INV-1001/recovery-pack")
    assert refused.status_code == 422
