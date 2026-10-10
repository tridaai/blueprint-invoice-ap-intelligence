"""Open-posting escalation webhook: the queue has always *shown*
an approved invoice still unposted past POSTING_ESCALATION_DAYS;
the sweep is the push counterpart — a signed
``posting_escalation`` webhook per open episode, recorded in the
invoice record's own escalation ledger beside the receipt fields
and deduped there, so a scheduler can run the sweep as often as
it likes. The recovery cases set the pattern; an unposted
approval is the same shape of silence."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

import ap_agent.service as service_module
from ap_agent.api import create_app
from ap_agent.matching import _parse_date
from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _capture(monkeypatch, status: int = 200):
    captured: list = []

    def fake_urlopen(request, timeout=0):
        captured.append(request)
        response = _Response()
        response.status = status
        return response

    monkeypatch.setattr(service_module.urllib.request, "urlopen", fake_urlopen)
    return captured


def _plus(iso: str, days: int) -> str:
    return (_parse_date(iso) + timedelta(days=days)).isoformat()


@pytest.fixture()
def service() -> InvoiceService:
    return InvoiceService(store=InMemoryStore(), checkpointer=False)


def _approved(service, invoice_id="INV-1001"):
    service.analyze(SAMPLES[invoice_id])
    return service.approve(invoice_id, approver="Priya Nair", reason="clean match")


def test_a_crossed_posting_fires_the_signed_escalation(service, monkeypatch):
    captured = _capture(monkeypatch)
    monkeypatch.delenv("POSTING_ESCALATION_DAYS", raising=False)
    # Approved before the webhook is armed: the approval's own
    # dispatch is a different event and not this test's count.
    result = _approved(service)
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    monkeypatch.setenv("ACTION_WEBHOOK_SECRET", "test-secret")
    # 20 days in: open, but inside the 21-day threshold — silence.
    assert service.escalate_open_postings(as_of=_plus(result.decided_at, 20)) == []
    escalated = service.escalate_open_postings(as_of=_plus(result.decided_at, 30))
    assert [item.invoice_id for item in escalated] == ["INV-1001"]
    assert len(captured) == 1
    request = captured[0]
    body = request.data
    payload = json.loads(body)
    assert payload["event"] == "posting_escalation"
    assert payload["invoice_id"] == "INV-1001"
    assert payload["age_days"] == 30 and payload["age_basis"] == "decision"
    assert payload["receipt_state"] == "none" and payload["receipt_detail"] == ""
    assert payload["escalation_days"] == 21 and payload["escalated"] is True
    assert payload["decided_by"] == "Priya Nair"
    # Signed like the approval webhook: hex HMAC-SHA256 over the body.
    expected = hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
    assert request.headers["X-trida-signature"] == f"sha256={expected}"
    # The ledger entry sits on the record, beside the receipt fields.
    record = service.get("INV-1001")
    assert len(record.posting_escalations) == 1
    entry = record.posting_escalations[0]
    assert entry.receipt_state == "none"
    assert entry.episode_since == _plus(result.decided_at, 0)
    assert entry.age_days == 30 and entry.escalation_days == 21
    assert entry.dispatch_status == "sent"


def test_sweep_is_deduped_per_episode_and_a_posted_receipt_resets(service, monkeypatch):
    _capture(monkeypatch)
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    result = _approved(service)
    first = service.escalate_open_postings(as_of=_plus(result.decided_at, 30))
    second = service.escalate_open_postings(as_of=_plus(result.decided_at, 45))
    assert len(first) == 1 and second == []  # once per episode
    # The receipt arrives: the item closes, the story resets —
    # later sweeps find nothing open to escalate.
    service.record_posting_receipt(
        "INV-1001",
        status="posted",
        actor="Tomas Reid",
        external_reference="ERP-DOC-1001",
        idempotency_key="post-INV-1001",
    )
    assert service.escalate_open_postings(as_of=_plus(result.decided_at, 90)) == []
    assert len(service.get("INV-1001").posting_escalations) == 1  # history kept


def test_a_failed_receipt_starts_a_new_episode_that_escalates_again(service, monkeypatch):
    captured = _capture(monkeypatch)
    result = _approved(service)
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    assert len(service.escalate_open_postings(as_of=_plus(result.decided_at, 30))) == 1
    service.record_posting_receipt(
        "INV-1001",
        status="failed",
        actor="Tomas Reid",
        detail="ERP refused: cost centre closed",
        idempotency_key="post-INV-1001",
    )
    escalated = service.escalate_open_postings(as_of=_plus(result.decided_at, 31))
    assert len(escalated) == 1 and len(captured) == 2
    payload = json.loads(captured[1].data)
    assert payload["receipt_state"] == "failed"
    assert payload["receipt_detail"] == "ERP refused: cost centre closed"
    entries = service.get("INV-1001").posting_escalations
    assert len(entries) == 2
    assert entries[1].receipt_state == "failed"
    receipt = service.get("INV-1001").posting_receipt
    assert entries[1].episode_since == receipt.recorded_at[:10]
    # And the new episode dedupes like the old one.
    assert service.escalate_open_postings(as_of=_plus(result.decided_at, 60)) == []


def test_no_webhook_configured_records_nothing(service, monkeypatch):
    _capture(monkeypatch)
    monkeypatch.delenv("ACTION_WEBHOOK_URL", raising=False)
    result = _approved(service)
    assert service.escalate_open_postings(as_of=_plus(result.decided_at, 30)) == []
    assert service.get("INV-1001").posting_escalations == []
    # Configure it later and the same crossing still fires.
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    assert len(service.escalate_open_postings(as_of=_plus(result.decided_at, 30))) == 1


def test_failed_dispatch_is_ledgered_not_raised(service, monkeypatch):
    _capture(monkeypatch, status=500)
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    result = _approved(service)
    escalated = service.escalate_open_postings(as_of=_plus(result.decided_at, 30))
    assert len(escalated) == 1
    entry = service.get("INV-1001").posting_escalations[0]
    assert entry.dispatch_status == "failed"
    assert service.escalate_open_postings(as_of=_plus(result.decided_at, 31)) == []


def test_metrics_count_the_recorded_escalations(service, monkeypatch):
    _capture(monkeypatch)
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    result = _approved(service)
    assert service.metrics_snapshot()["posting_open_items"]["escalations_recorded"] == 0
    service.escalate_open_postings(as_of=_plus(result.decided_at, 30))
    section = service.metrics_snapshot()["posting_open_items"]
    assert section["escalations_recorded"] == 1
    # escalated_count is anchored on today — the approval is
    # today's, so nothing is past the line yet; the recorded push
    # is the ledger's count, not the threshold's.


def test_escalate_endpoint(service, monkeypatch):
    captured = _capture(monkeypatch)
    result = _approved(service)
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    client = TestClient(create_app(service))
    res = client.post(
        "/postings/escalate", params={"as_of": _plus(result.decided_at, 30)}
    )
    assert res.status_code == 200
    body = res.json()
    assert [item["invoice_id"] for item in body] == ["INV-1001"]
    assert len(captured) == 1
    bad = client.post("/postings/escalate", params={"as_of": "not-a-date"})
    assert bad.status_code == 422
