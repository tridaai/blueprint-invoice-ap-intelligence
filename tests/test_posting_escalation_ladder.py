"""The posting-escalation ladder: one shout per episode expired
quietly however long the posting stayed open. Now the episode
climbs — level 1 (``posting_escalation``) past the threshold,
level 2 (``posting_escalation_urgent``) past twice the threshold
— each rung announced once per episode, in the same payload
shape with the rung named, and the posted receipt still the
only reset. A sweep climbs at most one rung per posting, so a
posting first seen past the second threshold hears level 1
first and level 2 on the next sweep."""

from __future__ import annotations

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
THRESHOLD = 21  # POSTING_ESCALATION_DAYS default; rung 2 sits at 42


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


def _armed(service, monkeypatch):
    result = _approved(service)
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    monkeypatch.setenv("ACTION_WEBHOOK_SECRET", "test-secret")
    return result


def test_the_ladder_climbs_level_1_then_level_2(service, monkeypatch):
    captured = _capture(monkeypatch)
    result = _armed(service, monkeypatch)

    # Past the threshold: level 1, in the original payload shape.
    escalated = service.escalate_open_postings(as_of=_plus(result.decided_at, 30))
    assert [i.invoice_id for i in escalated] == ["INV-1001"]
    payload = json.loads(captured[0].data)
    assert payload["event"] == "posting_escalation"
    assert payload["escalation_days"] == THRESHOLD
    assert "level" not in payload  # level 1 keeps the round-10 shape

    # Still open, still inside the second rung: silence.
    assert service.escalate_open_postings(as_of=_plus(result.decided_at, 41)) == []
    assert len(captured) == 1

    # Past twice the threshold: level 2, the urgent event, rung named.
    escalated = service.escalate_open_postings(as_of=_plus(result.decided_at, 50))
    assert [i.invoice_id for i in escalated] == ["INV-1001"]
    payload = json.loads(captured[1].data)
    assert payload["event"] == "posting_escalation_urgent"
    assert payload["level"] == 2 and payload["urgent"] is True
    assert payload["escalation_days"] == 2 * THRESHOLD
    assert payload["age_days"] == 50
    assert payload["escalated"] is True

    # The ledger holds both rungs of the one episode, in order.
    entries = service.get("INV-1001").posting_escalations
    assert [e.level for e in entries] == [1, 2]
    assert entries[0].escalation_days == THRESHOLD
    assert entries[1].escalation_days == 2 * THRESHOLD
    assert entries[0].episode_since == entries[1].episode_since

    # Both rungs fired: the ladder is climbed out — silence again.
    assert service.escalate_open_postings(as_of=_plus(result.decided_at, 90)) == []
    assert len(captured) == 2


def test_a_posting_first_seen_past_rung_2_hears_level_1_first(service, monkeypatch):
    captured = _capture(monkeypatch)
    result = _armed(service, monkeypatch)
    # Nobody swept until the posting was 50 days open: the sweep
    # climbs one rung per pass, so level 1 goes first...
    escalated = service.escalate_open_postings(as_of=_plus(result.decided_at, 50))
    assert len(escalated) == 1 and len(captured) == 1
    assert json.loads(captured[0].data)["event"] == "posting_escalation"
    # ...and level 2 on the next sweep, not in the same breath.
    escalated = service.escalate_open_postings(as_of=_plus(result.decided_at, 50))
    assert len(escalated) == 1 and len(captured) == 2
    assert json.loads(captured[1].data)["event"] == "posting_escalation_urgent"
    assert [e.level for e in service.get("INV-1001").posting_escalations] == [1, 2]


def test_a_posted_receipt_still_resets_the_whole_ladder(service, monkeypatch):
    captured = _capture(monkeypatch)
    result = _armed(service, monkeypatch)
    service.escalate_open_postings(as_of=_plus(result.decided_at, 30))
    service.escalate_open_postings(as_of=_plus(result.decided_at, 50))
    assert len(captured) == 2
    service.record_posting_receipt(
        "INV-1001",
        status="posted",
        actor="Tomas Reid",
        external_reference="ERP-DOC-1001",
        idempotency_key="post-INV-1001",
    )
    assert service.escalate_open_postings(as_of=_plus(result.decided_at, 120)) == []
    assert len(captured) == 2  # the reset holds; the history stays on the record
    assert len(service.get("INV-1001").posting_escalations) == 2


def test_no_webhook_records_no_rung(service, monkeypatch):
    _capture(monkeypatch)
    monkeypatch.delenv("ACTION_WEBHOOK_URL", raising=False)
    result = _approved(service)
    assert service.escalate_open_postings(as_of=_plus(result.decided_at, 50)) == []
    assert service.get("INV-1001").posting_escalations == []
    # Arm it late: the ladder still starts at level 1.
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    assert len(service.escalate_open_postings(as_of=_plus(result.decided_at, 50))) == 1
    entries = service.get("INV-1001").posting_escalations
    assert [e.level for e in entries] == [1]


def test_metrics_split_the_rungs(service, monkeypatch):
    _capture(monkeypatch)
    result = _armed(service, monkeypatch)
    service.escalate_open_postings(as_of=_plus(result.decided_at, 30))
    section = service.metrics_snapshot()["posting_open_items"]
    assert section["escalations_recorded"] == 1
    assert section["escalations_recorded_urgent"] == 0
    service.escalate_open_postings(as_of=_plus(result.decided_at, 50))
    section = service.metrics_snapshot()["posting_open_items"]
    assert section["escalations_recorded"] == 2
    assert section["escalations_recorded_urgent"] == 1


def test_the_endpoint_climbs_too(service, monkeypatch):
    captured = _capture(monkeypatch)
    result = _armed(service, monkeypatch)
    client = TestClient(create_app(service))
    res = client.post("/postings/escalate", params={"as_of": _plus(result.decided_at, 30)})
    assert res.status_code == 200 and len(captured) == 1
    res = client.post("/postings/escalate", params={"as_of": _plus(result.decided_at, 50)})
    assert res.status_code == 200
    assert [i["invoice_id"] for i in res.json()] == ["INV-1001"]
    assert json.loads(captured[1].data)["event"] == "posting_escalation_urgent"
