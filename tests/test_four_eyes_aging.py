"""Four-eyes aging in the queue: a large run confirmed once is
locked out of hand-off export until a second, different confirmer
arrives — and a wait nobody ages is a stall nobody sees. Past
PAYMENT_RUN_FOUR_EYES_AGING_DAYS (default 2) the run joins the
approver queue as an action item, with the first confirmer named:
they are the person to nudge, and the one person who cannot be
the second."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.queue import four_eyes_aging
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import PaymentRun
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
AS_OF = date(2026, 10, 10)


def _run(**overrides) -> PaymentRun:
    base = {
        "run_id": "RUN-USD-2026-10-05-2026-10-11",
        "currency": "USD",
        "status": "confirmed",
        "total_net": 30000.0,
        "four_eyes_required": True,
        "four_eyes_threshold": 25000.0,
        "confirmed_by": "Priya Nair",
        "confirmed_at": "2026-10-05T09:30:00+00:00",
    }
    base.update(overrides)
    return PaymentRun(**base)


def test_stale_pending_run_joins_oldest_first():
    older = _run(run_id="RUN-USD-OLDER", confirmed_at="2026-10-01T09:30:00+00:00")
    newer = _run(run_id="RUN-USD-NEWER")
    items = four_eyes_aging([newer, older], as_of=AS_OF, aging_days=2)
    assert [item.run_id for item in items] == ["RUN-USD-OLDER", "RUN-USD-NEWER"]
    item = items[0]
    assert item.age_days == 9
    assert item.confirmed_by == "Priya Nair"  # the first confirmer, named
    assert item.total_net == 30000.0
    assert item.four_eyes_threshold == 25000.0


def test_fresh_waits_and_finished_runs_stay_out():
    fresh = _run(run_id="RUN-FRESH", confirmed_at="2026-10-09T09:30:00+00:00")
    done = _run(run_id="RUN-DONE", second_confirmed_by="Tomas Reid",
                second_confirmed_at="2026-10-06T09:30:00+00:00")
    small = _run(run_id="RUN-SMALL", four_eyes_required=False)
    proposed = _run(run_id="RUN-PROPOSED", status="proposed", confirmed_at=None,
                    confirmed_by=None)
    items = four_eyes_aging(
        [fresh, done, small, proposed], as_of=AS_OF, aging_days=2
    )
    assert items == []
    # The boundary is strict: exactly at the threshold is a wait
    # still inside its allowance, one day past it is a stall.
    edge = _run(run_id="RUN-EDGE", confirmed_at="2026-10-08T09:30:00+00:00")
    assert four_eyes_aging([edge], as_of=AS_OF, aging_days=2) == []
    assert len(four_eyes_aging([edge], as_of=date(2026, 10, 11), aging_days=2)) == 1


@pytest.fixture()
def service(monkeypatch) -> InvoiceService:
    monkeypatch.setenv("PAYMENT_RUN_FOUR_EYES_THRESHOLD", "100")
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    run = svc.propose_payment_runs(window_days=7, as_of="2026-10-05")[0]
    assert run.four_eyes_required  # net 1156.40 > the test threshold
    svc.confirm_payment_run(run.run_id, actor="Tomas Reid")
    svc._test_run_id = run.run_id
    return svc


def _plus(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def test_queue_ages_the_wait_and_names_the_first_confirmer(service):
    assert service.queue_view(as_of=_plus(1)).four_eyes_pending_count == 0
    queue = service.queue_view(as_of=_plus(5))
    assert queue.four_eyes_aging_days == 2
    assert queue.four_eyes_pending_count == 1
    item = queue.four_eyes_pending[0]
    assert item.run_id == service._test_run_id
    assert item.confirmed_by == "Tomas Reid"
    assert item.age_days == 5


def test_second_confirmation_clears_the_action_item(service):
    service.confirm_payment_run_second(service._test_run_id, actor="Priya Nair")
    queue = service.queue_view(as_of=_plus(30))
    assert queue.four_eyes_pending_count == 0


def test_aging_threshold_is_configurable(service, monkeypatch):
    monkeypatch.setenv("PAYMENT_RUN_FOUR_EYES_AGING_DAYS", "10")
    queue = service.queue_view(as_of=_plus(5))
    assert queue.four_eyes_aging_days == 10
    assert queue.four_eyes_pending_count == 0
    queue = service.queue_view(as_of=_plus(11))
    assert queue.four_eyes_pending_count == 1


def test_queue_api_exposes_the_pending_runs(service):
    client = TestClient(create_app(service))
    resp = client.get("/queue", params={"as_of": _plus(4)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["four_eyes_pending_count"] == 1
    assert body["four_eyes_pending"][0]["confirmed_by"] == "Tomas Reid"
    assert body["four_eyes_aging_days"] == 2
