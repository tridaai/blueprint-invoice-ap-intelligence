"""Cross-cutting surfaces: the HMAC-signed approval webhook,
/metrics, and the audit export (JSON + CSV)."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import load_sample_invoices
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = load_sample_invoices()


@pytest.fixture()
def service() -> InvoiceService:
    return InvoiceService(store=InMemoryStore(), checkpointer=False)


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _capture(monkeypatch):
    import ap_agent.service as service_module

    captured = {}

    def fake_urlopen(request, timeout=0):
        captured["request"] = request
        return _Response()

    monkeypatch.setattr(service_module.urllib.request, "urlopen", fake_urlopen)
    return captured


def test_webhook_signed_when_secret_configured(service, monkeypatch):
    captured = _capture(monkeypatch)
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    monkeypatch.setenv("ACTION_WEBHOOK_SECRET", "test-secret")
    result = service.analyze(SAMPLES[0])
    decided = service.approve(result.invoice_id, approver="a")
    assert decided.dispatch_status == "sent"
    request = captured["request"]
    body = request.data
    expected = hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
    assert request.headers.get("X-trida-signature") == f"sha256={expected}"
    # the signed body is the packet payload itself
    assert json.loads(body)["invoice_id"] == result.invoice_id


def test_webhook_unsigned_without_secret(service, monkeypatch):
    captured = _capture(monkeypatch)
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    result = service.analyze(SAMPLES[0])
    service.approve(result.invoice_id, approver="a")
    assert captured["request"].headers.get("X-trida-signature") is None


def _populated(service: InvoiceService) -> InvoiceService:
    for sample in SAMPLES[:3]:  # INV-1001 clean, 1002 variance, 1003 quantity
        service.analyze(sample)
    service.approve("INV-1001", approver="Priya", reason="match verified")
    service.reject("INV-1002", reviewer="Priya", reason="price not agreed")
    return service


def test_metrics_snapshot_counts(service):
    _populated(service)
    metrics = service.metrics_snapshot()
    assert metrics["runs_total"] == 3
    assert metrics["decisions"] == {
        "approved": 1,
        "rejected": 1,
        "awaiting_approver": 1,
    }
    assert metrics["by_outcome"]["clean_match"] == 1
    assert metrics["by_outcome"]["price_variance"] == 1
    assert metrics["mean_latency_seconds"] is not None
    # mock runs report no tokens — the totals say so honestly
    assert metrics["total_input_tokens"] is None
    assert metrics["total_estimated_cost_usd"] is None


def test_audit_rows_carry_the_decision_trail(service):
    _populated(service)
    rows = {row["invoice_id"]: row for row in service.audit_rows()}
    assert rows["INV-1001"]["decision"] == "approved"
    assert rows["INV-1001"]["decided_by"] == "Priya"
    assert rows["INV-1001"]["decision_reason"] == "match verified"
    assert rows["INV-1002"]["outcome"] == "price_variance"
    assert rows["INV-1003"]["status"] == "awaiting_approver"
    assert rows["INV-1003"]["decision"] is None


def test_metrics_and_audit_endpoints():
    svc = _populated(InvoiceService(store=InMemoryStore(), checkpointer=False))
    client = TestClient(create_app(svc))

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["runs_total"] == 3

    export = client.get("/audit/export")
    assert export.status_code == 200
    body = export.json()
    assert body["count"] == 3
    assert body["records"][0]["invoice_id"] == "INV-1001"

    csv_export = client.get("/audit/export", params={"format": "csv"})
    assert csv_export.status_code == 200
    assert "text/csv" in csv_export.headers["content-type"]
    assert "attachment" in csv_export.headers["content-disposition"]
    lines = csv_export.text.strip().splitlines()
    assert lines[0].startswith("invoice_id,supplier_name,invoice_number")
    assert len(lines) == 4  # header + 3 rows
    assert any("INV-1002" in line and "price not agreed" in line for line in lines)

    assert client.get("/audit/export", params={"format": "xml"}).status_code == 422
