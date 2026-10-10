"""Open-posting escalation into the queue: an approved invoice
with no posted receipt is a report line past the grace window
(7 days); past the escalation threshold (POSTING_ESCALATION_DAYS,
default 21) it joins the approver queue as an action item, with
the ERP failure reason attached when a receipt failed — one
worklist for everything the building still owes an answer for."""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.matching import _parse_date
from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


@pytest.fixture()
def service() -> InvoiceService:
    return InvoiceService(store=InMemoryStore(), checkpointer=False)


def _approved(service, invoice_id="INV-1001"):
    service.analyze(SAMPLES[invoice_id])
    return service.approve(invoice_id, approver="Priya Nair", reason="clean match")


def _plus(iso: str, days: int) -> str:
    return (_parse_date(iso) + timedelta(days=days)).isoformat()


def test_open_posting_escalates_only_past_the_second_threshold(service):
    result = _approved(service)
    # Past grace (7) but not escalation (21): on the report, not in the queue.
    queue = service.queue_view(as_of=_plus(result.decided_at, 10))
    assert service.approved_not_posted_report(as_of=_plus(result.decided_at, 10)).count == 1
    assert queue.open_posting_count == 0
    assert queue.posting_escalation_days == 21
    # Past escalation: an action item in the queue.
    queue = service.queue_view(as_of=_plus(result.decided_at, 22))
    assert queue.open_posting_count == 1
    item = queue.open_postings[0]
    assert item.invoice_id == "INV-1001"
    assert item.age_days == 22
    assert item.receipt_state == "none"


def test_escalated_item_carries_the_erp_failure_reason(service):
    result = _approved(service)
    service.record_posting_receipt(
        "INV-1001",
        status="failed",
        actor="Tomas Reid",
        detail="ERP refused: cost centre blocked for period close",
        idempotency_key="post-INV-1001",
    )
    queue = service.queue_view(as_of=_plus(result.decided_at, 30))
    assert queue.open_posting_count == 1
    item = queue.open_postings[0]
    assert item.receipt_state == "failed"
    assert item.receipt_detail == "ERP refused: cost centre blocked for period close"


def test_a_posted_receipt_never_escalates(service):
    result = _approved(service)
    service.record_posting_receipt(
        "INV-1001",
        status="posted",
        actor="Tomas Reid",
        external_reference="ERP-DOC-1001",
        idempotency_key="post-INV-1001",
    )
    queue = service.queue_view(as_of=_plus(result.decided_at, 90))
    assert queue.open_posting_count == 0


def test_escalation_threshold_is_configurable_and_never_below_grace(service, monkeypatch):
    result = _approved(service)
    monkeypatch.setenv("POSTING_ESCALATION_DAYS", "10")
    queue = service.queue_view(as_of=_plus(result.decided_at, 11))
    assert queue.posting_escalation_days == 10
    assert queue.open_posting_count == 1
    # An escalation below the grace window is clamped to it.
    monkeypatch.setenv("POSTING_ESCALATION_DAYS", "3")
    queue = service.queue_view(as_of=_plus(result.decided_at, 5))
    assert queue.posting_escalation_days == 7
    assert queue.open_posting_count == 0


def test_metrics_count_the_escalated_items(service, monkeypatch):
    _approved(service)
    metrics = service.metrics_snapshot()
    assert metrics["posting_open_items"]["escalated_count"] == 0
    assert metrics["posting_open_items"]["escalation_days"] == 21


def test_queue_api_exposes_the_escalated_postings(service):
    result = _approved(service)
    client = TestClient(create_app(service))
    resp = client.get("/queue", params={"as_of": _plus(result.decided_at, 25)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["open_posting_count"] == 1
    assert body["open_postings"][0]["invoice_id"] == "INV-1001"
    assert body["posting_escalation_days"] == 21
