"""Information-wait escalation: the queue has always *shown* a
wait past INFORMATION_WAIT_AGING_DAYS; the sweep is the push
counterpart — a signed ``information_wait_escalation`` webhook
per wait episode once the episode runs past twice the aging
threshold, recorded in the invoice record's own escalation
ledger beside the request and deduped there. A partial reply
starts a new episode (the remaining items have been owed only
since that answer landed) with an escalation of its own; a
complete reply closes the wait — the reset. The recovery cases
and the posting ladder set the pattern; a blocked case waiting
on someone outside the building is the same shape of silence.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

import ap_agent.service as service_module
from ap_agent.api import create_app
from ap_agent.clarify import humanize_item
from ap_agent.matching import _parse_date
from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
PO_PHRASE = humanize_item("purchase_order")
LEGIBLE_PHRASE = humanize_item("legible_invoice")


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


def _blocked(service: InvoiceService) -> object:
    result = service.analyze(SAMPLES["INV-1005"])
    assert result.needs_information
    return result


def test_a_crossed_wait_fires_the_signed_escalation(service, monkeypatch):
    captured = _capture(monkeypatch)
    result = _blocked(service)
    requested = result.information_request.requested_at
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    monkeypatch.setenv("ACTION_WEBHOOK_SECRET", "test-secret")
    # Aging threshold 3d puts the wait on the worklist; the
    # escalation threshold is twice that — at +6 the wait is old
    # but unpushed, at +7 it fires.
    assert service.escalate_information_waits(as_of=_plus(requested, 6)) == []
    escalated = service.escalate_information_waits(as_of=_plus(requested, 7))
    assert [item.invoice_id for item in escalated] == ["INV-1005"]
    assert len(captured) == 1
    request_sent = captured[0]
    body = request_sent.data
    payload = json.loads(body)
    assert payload["event"] == "information_wait_escalation"
    assert payload["invoice_id"] == "INV-1005"
    assert payload["outcome"] == "missing_po"
    assert payload["reply_status"] == "awaiting_reply"
    assert payload["still_missing"] == [PO_PHRASE]
    assert payload["waiting_days"] == 7
    assert payload["escalation_days"] == 6
    assert payload["escalated"] is True
    # The signature is the HMAC over the exact body bytes.
    signature = request_sent.headers["X-trida-signature"]
    expected = hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
    assert signature == f"sha256={expected}"
    # The dispatch is on the record's ledger.
    record = service._get_store().get("INV-1005")
    (entry,) = record.result.information_escalations
    assert entry.dispatch_status == "sent"
    assert entry.escalation_days == 6
    assert entry.waiting_days == 7
    assert entry.episode_since == _parse_date(requested).isoformat()


def test_the_sweep_does_not_refire_an_episode(service, monkeypatch):
    _capture(monkeypatch)
    result = _blocked(service)
    requested = result.information_request.requested_at
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    first = service.escalate_information_waits(as_of=_plus(requested, 10))
    assert len(first) == 1
    assert service.escalate_information_waits(as_of=_plus(requested, 10)) == []
    assert service.escalate_information_waits(as_of=_plus(requested, 40)) == []
    record = service._get_store().get("INV-1005")
    assert len(record.result.information_escalations) == 1


def test_no_webhook_configured_records_nothing(service, monkeypatch):
    monkeypatch.delenv("ACTION_WEBHOOK_URL", raising=False)
    result = _blocked(service)
    requested = result.information_request.requested_at
    assert service.escalate_information_waits(as_of=_plus(requested, 30)) == []
    record = service._get_store().get("INV-1005")
    assert record.result.information_escalations == []


def test_a_failed_dispatch_is_still_recorded_once(service, monkeypatch):
    _capture(monkeypatch, status=500)
    result = _blocked(service)
    requested = result.information_request.requested_at
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    escalated = service.escalate_information_waits(as_of=_plus(requested, 9))
    assert len(escalated) == 1
    record = service._get_store().get("INV-1005")
    (entry,) = record.result.information_escalations
    assert entry.dispatch_status == "failed"
    # A failed firing is still the episode's one firing.
    assert service.escalate_information_waits(as_of=_plus(requested, 12)) == []


def test_a_partial_reply_starts_a_new_episode_that_escalates_in_turn(
    service, monkeypatch
):
    captured = _capture(monkeypatch)
    result = _blocked(service)
    requested = result.information_request.requested_at
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    assert len(service.escalate_information_waits(as_of=_plus(requested, 8))) == 1
    # Shape the request into a two-item one and land a partial
    # reply: the legible copy arrived on day +10; the PO is still
    # owed, and the wait's episode restarts from the answer.
    store = service._get_store()
    record = store.get("INV-1005")
    request = record.result.information_request
    request.missing_items = [PO_PHRASE, LEGIBLE_PHRASE]
    request.missing_item_keys = ["purchase_order", "legible_invoice"]
    store.save(record)
    partial_at = _plus(requested, 10) + "T09:00:00+00:00"
    service.record_information_reply(
        "INV-1005",
        actor="Sam (AP)",
        supplied_items=["legible_invoice"],
        received_at=partial_at,
    )
    # Five days into the new episode: the crossing has not come.
    assert service.escalate_information_waits(as_of=_plus(partial_at, 5)) == []
    escalated = service.escalate_information_waits(as_of=_plus(partial_at, 7))
    assert [item.invoice_id for item in escalated] == ["INV-1005"]
    assert len(captured) == 2
    payload = json.loads(captured[1].data)
    assert payload["reply_status"] == "partially_answered"
    assert payload["still_missing"] == [PO_PHRASE]
    assert payload["episode_since"] == _plus(requested, 10)
    record = store.get("INV-1005")
    episodes = [e.episode_since for e in record.result.information_escalations]
    assert episodes == [_parse_date(requested).isoformat(), _plus(requested, 10)]


def test_a_complete_reply_is_the_reset(service, monkeypatch):
    _capture(monkeypatch)
    result = _blocked(service)
    requested = result.information_request.requested_at
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    service.record_information_reply(
        "INV-1005",
        actor="Sam (AP)",
        supplied_items=["purchase_order"],
        received_at=_plus(requested, 2) + "T09:00:00+00:00",
    )
    assert service.escalate_information_waits(as_of=_plus(requested, 60)) == []


def test_the_queue_marks_an_escalated_wait_and_metrics_count_it(
    service, monkeypatch
):
    _capture(monkeypatch)
    result = _blocked(service)
    requested = result.information_request.requested_at
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    before = service.queue_view(as_of=_plus(requested, 8))
    assert before.information_waits[0].escalated is False
    service.escalate_information_waits(as_of=_plus(requested, 8))
    after = service.queue_view(as_of=_plus(requested, 8))
    (item,) = after.information_waits
    assert item.escalated is True
    assert item.episode_since == _parse_date(requested).isoformat()
    metrics = service.metrics_snapshot()
    assert metrics["information_waits"]["aging_days"] == 3
    assert metrics["information_waits"]["escalation_days"] == 6
    assert metrics["information_waits"]["escalations_recorded"] == 1


def test_escalation_endpoint(service, monkeypatch):
    _capture(monkeypatch)
    result = _blocked(service)
    requested = result.information_request.requested_at
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    client = TestClient(create_app(service))
    response = client.post(
        "/information-waits/escalate", params={"as_of": _plus(requested, 9)}
    )
    assert response.status_code == 200
    (item,) = response.json()
    assert item["invoice_id"] == "INV-1005"
    assert item["waiting_days"] == 9
    again = client.post(
        "/information-waits/escalate", params={"as_of": _plus(requested, 9)}
    )
    assert again.json() == []
    bad = client.post("/information-waits/escalate", params={"as_of": "junk"})
    assert bad.status_code == 422
