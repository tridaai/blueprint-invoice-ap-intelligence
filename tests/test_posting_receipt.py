"""The ERP posting receipt loop: the ERP's outcome — posted with
its document number, or failed with its reason — is recorded back
onto the approved invoice's record (and into the audit export),
closing the output seam the example adapter opens. The agent
posted nothing itself; a posted receipt cannot be replaced by a
different posting."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
KEY = "ap-posting-deadbeefcafe"


@pytest.fixture()
def service() -> InvoiceService:
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    return svc


def test_posted_receipt_lands_on_the_record_and_the_audit(service):
    result = service.record_posting_receipt(
        "INV-1001",
        status="posted",
        actor="Tomas Reid",
        external_reference="ERP-DOC-1042",
        idempotency_key=KEY,
    )
    receipt = result.posting_receipt
    assert receipt is not None
    assert receipt.status == "posted"
    assert receipt.external_reference == "ERP-DOC-1042"
    assert receipt.idempotency_key == KEY
    assert receipt.recorded_by == "Tomas Reid"
    # Bookkeeping only: the decision stands, no action was taken.
    assert result.status == "approved"
    assert result.external_action_taken is False
    stored = service.get("INV-1001")
    assert stored is not None and stored.posting_receipt is not None
    row = {r["invoice_id"]: r for r in service.audit_rows()}["INV-1001"]
    assert row["posting_status"] == "posted"
    assert row["posting_reference"] == "ERP-DOC-1042"


def test_receipt_requires_an_approved_invoice():
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])  # still at the gate
    with pytest.raises(ValueError, match="only an approved invoice"):
        svc.record_posting_receipt(
            "INV-1001", status="posted", actor="Tomas Reid",
            external_reference="ERP-DOC-1",
        )
    with pytest.raises(KeyError):
        svc.record_posting_receipt(
            "INV-9999", status="posted", actor="Tomas Reid",
            external_reference="ERP-DOC-1",
        )


def test_receipt_shape_guards(service):
    with pytest.raises(ValueError, match="external reference"):
        service.record_posting_receipt(
            "INV-1001", status="posted", actor="Tomas Reid"
        )
    with pytest.raises(ValueError, match="reason"):
        service.record_posting_receipt(
            "INV-1001", status="failed", actor="Tomas Reid"
        )
    with pytest.raises(ValueError, match="'posted' or 'failed'"):
        service.record_posting_receipt(
            "INV-1001", status="queued", actor="Tomas Reid",
            external_reference="ERP-DOC-1",
        )
    assert service.get("INV-1001").posting_receipt is None


def test_failed_receipt_can_be_superseded_by_the_retry(service):
    failed = service.record_posting_receipt(
        "INV-1001", status="failed", actor="Tomas Reid",
        idempotency_key=KEY, detail="period 2026-09 is closed",
    )
    assert failed.posting_receipt.status == "failed"
    posted = service.record_posting_receipt(
        "INV-1001", status="posted", actor="Tomas Reid",
        external_reference="ERP-DOC-1042", idempotency_key=KEY,
    )
    assert posted.posting_receipt.status == "posted"
    assert posted.posting_receipt.external_reference == "ERP-DOC-1042"


def test_posted_receipt_replays_by_key_but_never_double_posts(service):
    first = service.record_posting_receipt(
        "INV-1001", status="posted", actor="Tomas Reid",
        external_reference="ERP-DOC-1042", idempotency_key=KEY,
    )
    replay = service.record_posting_receipt(
        "INV-1001", status="posted", actor="Tomas Reid",
        external_reference="ERP-DOC-1042", idempotency_key=KEY,
    )
    assert replay.posting_receipt.recorded_at == first.posting_receipt.recorded_at
    with pytest.raises(ValueError, match="already recorded as posted"):
        service.record_posting_receipt(
            "INV-1001", status="posted", actor="Tomas Reid",
            external_reference="ERP-DOC-9999",
            idempotency_key="ap-posting-othervalue",
        )
    with pytest.raises(ValueError, match="already recorded as posted"):
        service.record_posting_receipt(
            "INV-1001", status="failed", actor="Tomas Reid",
            detail="anything",
        )


def test_posting_receipt_endpoint(service):
    client = TestClient(create_app(service))
    recorded = client.post(
        "/invoices/INV-1001/posting-receipt",
        json={
            "actor": "Tomas Reid",
            "status": "posted",
            "external_reference": "ERP-DOC-1042",
            "idempotency_key": KEY,
        },
    )
    assert recorded.status_code == 200
    assert recorded.json()["posting_receipt"]["external_reference"] == "ERP-DOC-1042"
    export = client.get("/audit/export").json()
    row = {r["invoice_id"]: r for r in export["records"]}["INV-1001"]
    assert row["posting_status"] == "posted"
    missing = client.post(
        "/invoices/INV-9999/posting-receipt",
        json={"actor": "Tomas Reid", "status": "posted", "external_reference": "X"},
    )
    assert missing.status_code == 404
    invalid = client.post(
        "/invoices/INV-1001/posting-receipt",
        json={"actor": "Tomas Reid", "status": "posted"},
    )
    assert invalid.status_code == 422  # a different posting, and no reference
