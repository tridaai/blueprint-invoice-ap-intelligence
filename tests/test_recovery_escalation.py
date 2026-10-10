"""Recovery-case escalation webhook: the queue has always *shown*
a case whose status outlasted the SLA; the sweep is the push
counterpart — a signed ``recovery_escalation`` webhook per status
crossing, recorded in the pack's own escalation ledger and deduped
there, so a scheduler can run the sweep as often as it likes."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

import ap_agent.service as service_module
from ap_agent.api import create_app
from ap_agent.schemas import RecoveryCaseEvent, RecoveryPack
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore


def _pack(pack_id: str, created: str, events: list[tuple[str, str]], status: str):
    return RecoveryPack(
        pack_id=pack_id,
        duplicate_invoice_id=f"INV-{pack_id[-4:]}",
        original_invoice_id="INV-1001",
        supplier_name="Northbeam Office Supplies",
        invoice_number="NB-4471",
        amount=1180.0,
        created_at=created,
        case_status=status,
        case_history=[
            RecoveryCaseEvent(status=s, actor="operator", at=at) for s, at in events
        ],
    )


# Opened Sep 1, contacted Sep 20: at Oct 10 it has sat in
# supplier_contacted for 20 days — past the default 14-day SLA.
BREACHED = _pack(
    "REC-INV-9001",
    "2026-09-01T09:00:00+00:00",
    [("opened", "2026-09-01T09:00:00+00:00"), ("supplier_contacted", "2026-09-20T09:00:00+00:00")],
    "supplier_contacted",
)
# Opened Oct 1: 9 days in opened — within the SLA.
FRESH = _pack(
    "REC-INV-9002",
    "2026-10-01T09:00:00+00:00",
    [("opened", "2026-10-01T09:00:00+00:00")],
    "opened",
)
# Closed cases never escalate, however stale.
CLOSED = _pack(
    "REC-INV-9003",
    "2026-08-01T09:00:00+00:00",
    [("opened", "2026-08-01T09:00:00+00:00"), ("closed", "2026-08-20T09:00:00+00:00")],
    "closed",
)


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


def _service(*packs: RecoveryPack) -> InvoiceService:
    store = InMemoryStore()
    for pack in packs:
        store.save_recovery_pack(pack.model_copy(deep=True))
    return InvoiceService(store=store, checkpointer=False)


def test_breached_case_fires_the_signed_escalation(monkeypatch):
    captured = _capture(monkeypatch)
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    monkeypatch.setenv("ACTION_WEBHOOK_SECRET", "test-secret")
    monkeypatch.delenv("RECOVERY_CASE_SLA_DAYS", raising=False)
    svc = _service(BREACHED, FRESH, CLOSED)
    escalated = svc.escalate_breached_recovery_cases(as_of="2026-10-10")
    assert [p.pack_id for p in escalated] == ["REC-INV-9001"]
    assert len(captured) == 1
    request = captured[0]
    assert request.full_url == "https://example.invalid/hook"
    body = request.data
    payload = json.loads(body)
    assert payload["event"] == "recovery_escalation"
    assert payload["pack_id"] == "REC-INV-9001"
    assert payload["case_status"] == "supplier_contacted"
    assert payload["status_age_days"] == 20 and payload["age_days"] == 39
    assert payload["sla_days"] == 14 and payload["sla_breached"] is True
    assert payload["amount"] == 1180.0
    # Signed like the approval webhook: hex HMAC-SHA256 over the body.
    expected = hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
    assert request.headers["X-trida-signature"] == f"sha256={expected}"
    # The ledger entry is on the pack, and it is not a case transition:
    # the status-age clock (last history event) is untouched.
    pack = svc.get_recovery_pack("INV-9001")
    assert len(pack.escalations) == 1
    entry = pack.escalations[0]
    assert entry.case_status == "supplier_contacted"
    assert entry.status_since == "2026-09-20"
    assert entry.dispatch_status == "sent"
    assert len(pack.case_history) == 2
    # The fresh and the closed case carry no ledger.
    assert svc.get_recovery_pack("INV-9002").escalations == []
    assert svc.get_recovery_pack("INV-9003").escalations == []


def test_sweep_is_deduped_per_crossing(monkeypatch):
    captured = _capture(monkeypatch)
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    store = InMemoryStore()
    store.save_recovery_pack(BREACHED.model_copy(deep=True))
    svc = InvoiceService(store=store, checkpointer=False)
    first = svc.escalate_breached_recovery_cases(as_of="2026-10-10")
    second = svc.escalate_breached_recovery_cases(as_of="2026-10-12")
    assert len(first) == 1 and second == []
    assert len(captured) == 1  # once per crossing, not once per sweep
    # The case moves and stalls again: a new crossing, a new escalation.
    svc.transition_recovery_case("INV-9001", "credit_received", actor="op")
    pack = svc.get_recovery_pack("INV-9001")
    pack.case_history[-1].at = "2026-10-12T09:00:00+00:00"
    store.save_recovery_pack(pack)
    third = svc.escalate_breached_recovery_cases(as_of="2026-10-30")
    assert len(third) == 1 and len(captured) == 2
    entry = svc.get_recovery_pack("INV-9001").escalations[-1]
    assert entry.case_status == "credit_received"
    assert entry.status_since == "2026-10-12"


def test_no_webhook_configured_records_nothing(monkeypatch):
    _capture(monkeypatch)
    monkeypatch.delenv("ACTION_WEBHOOK_URL", raising=False)
    svc = _service(BREACHED)
    assert svc.escalate_breached_recovery_cases(as_of="2026-10-10") == []
    assert svc.get_recovery_pack("INV-9001").escalations == []
    # Configure it later and the same crossing still fires — the
    # no-op sweep consumed nothing.
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    assert len(svc.escalate_breached_recovery_cases(as_of="2026-10-10")) == 1


def test_failed_dispatch_is_ledgered_not_raised(monkeypatch):
    _capture(monkeypatch, status=500)
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    svc = _service(BREACHED)
    escalated = svc.escalate_breached_recovery_cases(as_of="2026-10-10")
    assert len(escalated) == 1
    entry = svc.get_recovery_pack("INV-9001").escalations[0]
    assert entry.dispatch_status == "failed"
    # And the failure is the record: the sweep does not spam retries.
    assert svc.escalate_breached_recovery_cases(as_of="2026-10-11") == []


def test_escalate_endpoint(monkeypatch):
    captured = _capture(monkeypatch)
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    svc = _service(BREACHED, FRESH)
    client = TestClient(create_app(svc))
    res = client.post("/recovery-cases/escalate", params={"as_of": "2026-10-10"})
    assert res.status_code == 200
    body = res.json()
    assert [p["pack_id"] for p in body] == ["REC-INV-9001"]
    assert body[0]["escalations"][0]["dispatch_status"] == "sent"
    assert len(captured) == 1
    bad = client.post("/recovery-cases/escalate", params={"as_of": "not-a-date"})
    assert bad.status_code == 422
