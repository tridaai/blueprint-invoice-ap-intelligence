"""API surface: health/readiness, analysis, decisions, auth, SSE."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import load_sample_invoices
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = load_sample_invoices()


@pytest.fixture()
def client():
    service = InvoiceService(store=InMemoryStore(), checkpointer=False)
    return TestClient(create_app(service))


def test_health_and_readiness(client):
    assert client.get("/health").json()["status"] == "ok"
    readiness = client.get("/readiness")
    assert readiness.status_code == 200
    body = readiness.json()
    assert body["ready"] is True
    assert body["store"] == "memory"


def test_request_id_header_present(client):
    response = client.get("/health", headers={"X-Request-ID": "test-req-1"})
    assert response.headers.get("x-request-id") == "test-req-1"


def test_console_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Invoice" in response.text


def test_analyze_then_get(client):
    response = client.post("/invoices/analyze", json=SAMPLES[0])
    assert response.status_code == 200
    body = response.json()
    assert body["invoice_id"] == "INV-1001"
    assert body["triage"]["outcome"] == "clean_match"
    assert body["status"] == "awaiting_approver"
    fetched = client.get("/invoices/INV-1001")
    assert fetched.status_code == 200
    assert fetched.json()["invoice_id"] == "INV-1001"


def test_approve_and_reject_flow(client):
    client.post("/invoices/analyze", json=SAMPLES[0])
    decided = client.post(
        "/invoices/INV-1001/approve",
        json={"approver": "priya.finance", "reason": "match verified"},
    )
    assert decided.status_code == 200
    assert decided.json()["status"] == "approved"
    assert decided.json()["decided_by"] == "priya.finance"
    again = client.post("/invoices/INV-1001/approve", json={"approver": "priya"})
    assert again.status_code == 422


def test_decision_requires_actor(client):
    client.post("/invoices/analyze", json=SAMPLES[0])
    assert client.post("/invoices/INV-1001/approve", json={}).status_code == 422


def test_unknown_invoice_404(client):
    assert client.get("/invoices/INV-9999").status_code == 404
    assert client.post("/invoices/INV-9999/approve", json={"approver": "x"}).status_code == 404


def test_guardrail_failed_draft_cannot_be_approved_via_api(client):
    trap = next(s for s in SAMPLES if s["invoice_id"] == "INV-1012")
    client.post("/invoices/analyze", json=trap)
    response = client.post("/invoices/INV-1012/approve", json={"approver": "priya"})
    assert response.status_code == 422
    assert "guardrail" in response.text.lower()


def test_stream_emits_events_and_result(client):
    response = client.post("/invoices/analyze/stream", json=SAMPLES[0])
    assert response.status_code == 200
    assert "run_completed" in response.text
    assert "INV-1001" in response.text


def test_policies_endpoint(client):
    body = client.get("/policies").json()
    ids = [p["id"] for p in body]
    assert "AP-MATCH-01" in ids
    assert "AP-BANK-01" in ids


def test_api_key_gate(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    service = InvoiceService(store=InMemoryStore(), checkpointer=False)
    gated = TestClient(create_app(service), raise_server_exceptions=False)
    assert gated.get("/health").status_code == 200  # liveness stays open
    assert gated.post("/invoices/analyze", json=SAMPLES[0]).status_code == 401
    ok = gated.post(
        "/invoices/analyze", json=SAMPLES[0], headers={"X-API-Key": "test-key"}
    )
    assert ok.status_code == 200
