"""Pricing the four-eyes wait: the second-confirmation wait used
to be visible only as elapsed days (the queue's aging) and lost
discounts (the drift check at export, which prices the whole
confirm-to-export span). Now the wait itself carries a price:
when the second confirmation lands, the run record is stamped
with the discounts that were capturable at the first
confirmation and lapsed before the second — charged to the wait
by the same capturable-on-the-deadline-day rule everything else
uses — and /metrics aggregates the stamps beside the forecast
accuracy. A discount already gone before the first confirmation
is never the wait's fault; a same-day second confirmation costs
nothing."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.paymentruns import handoff_metrics, price_four_eyes_wait
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import PaymentRun, PaymentRunLine
from ap_agent.service import InvoiceService
from ap_agent.storage import InMemoryObjectStore
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
# INV-1001's stored discount: 23.60, capturable until 2026-10-08
# (pinned by test_discounts / the run fixtures). The pure-function
# fixtures below use their own 2026-10-05 deadline.
DISCOUNT = 23.60
DEADLINE = "2026-10-08"


def _run(**overrides) -> PaymentRun:
    fields = dict(
        run_id="RUN-USD-X",
        currency="USD",
        status="confirmed",
        four_eyes_required=True,
        confirmed_by="Tomas Reid",
        confirmed_at="2026-10-03T09:00:00+00:00",
        second_confirmed_by="Amara Chen",
        second_confirmed_at="2026-10-08T09:00:00+00:00",
        lines=[
            PaymentRunLine(
                invoice_id="INV-1",
                supplier_name="Northbeam Office Supplies",
                gross_amount=1180.0,
                discount_amount=DISCOUNT,
                discount_deadline="2026-10-05",
                net_amount=1156.40,
            )
        ],
    )
    fields.update(overrides)
    return PaymentRun(**fields)


def test_a_discount_lapsing_inside_the_wait_is_charged_to_it():
    cost, lapses = price_four_eyes_wait(_run())
    # Capturable on the first confirmation's day (10-03 <= 10-05),
    # gone by the second's (10-05 < 10-08): the wait cost 23.60.
    assert cost == DISCOUNT
    assert len(lapses) == 1
    assert lapses[0].invoice_id == "INV-1"
    assert lapses[0].discount_deadline == "2026-10-05"


def test_a_discount_still_alive_at_the_second_confirmation_costs_nothing():
    cost, lapses = price_four_eyes_wait(
        _run(second_confirmed_at="2026-10-05T18:00:00+00:00")
    )
    # The deadline day itself is still capturable — the second
    # confirmer signed in time.
    assert cost == 0.0 and lapses == []


def test_a_discount_gone_before_the_wait_is_not_the_waits_fault():
    cost, lapses = price_four_eyes_wait(
        _run(confirmed_at="2026-10-06T09:00:00+00:00")
    )
    # The window closed 10-05, before the first confirmation on
    # 10-06: the calendar killed it, not the second confirmer.
    assert cost == 0.0 and lapses == []


def test_a_same_day_second_confirmation_costs_nothing():
    cost, lapses = price_four_eyes_wait(
        _run(second_confirmed_at="2026-10-03T17:30:00+00:00")
    )
    assert cost == 0.0 and lapses == []


def test_an_unseconded_run_has_no_price():
    assert price_four_eyes_wait(
        _run(second_confirmed_by="", second_confirmed_at="")
    ) == (0.0, [])


def _service_with_run(monkeypatch) -> InvoiceService:
    monkeypatch.setenv("PAYMENT_RUN_FOUR_EYES_THRESHOLD", "1000")
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        object_store=InMemoryObjectStore(),
    )
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    svc.propose_payment_runs(window_days=7, as_of="2026-10-01")
    return svc


def test_the_second_confirmation_stamps_the_wait_price(monkeypatch):
    svc = _service_with_run(monkeypatch)
    run = svc.list_payment_runs()[0]
    assert run.four_eyes_required is True
    line = run.lines[0]
    assert line.discount_amount == DISCOUNT and line.discount_deadline == DEADLINE
    svc.confirm_payment_run(run.run_id, actor="Tomas Reid")
    # The first confirmation really landed on 2026-10-03, while
    # the discount was still capturable; the second lands today.
    store = svc._get_store()
    stamped = store.get_payment_run(run.run_id)
    stamped.confirmed_at = "2026-10-03T09:00:00+00:00"
    store.save_payment_run(stamped)

    second = svc.confirm_payment_run_second(run.run_id, actor="Amara Chen")
    assert second.four_eyes_wait_cost == DISCOUNT
    assert [lapse.invoice_id for lapse in second.four_eyes_wait_lapses] == [
        "INV-1001"
    ]
    assert second.four_eyes_wait_lapses[0].discount_deadline == DEADLINE
    # The stamp is on the stored record, not just the response.
    assert store.get_payment_run(run.run_id).four_eyes_wait_cost == DISCOUNT


def test_a_prompt_second_confirmation_stamps_zero(monkeypatch):
    svc = _service_with_run(monkeypatch)
    run = svc.list_payment_runs()[0]
    svc.confirm_payment_run(run.run_id, actor="Tomas Reid")
    second = svc.confirm_payment_run_second(run.run_id, actor="Amara Chen")
    # Both confirmations landed today — after the 10-08 deadline,
    # so the discount was already gone before the wait began and
    # none of the loss is charged to it.
    assert second.four_eyes_wait_cost == 0.0
    assert second.four_eyes_wait_lapses == []


def test_metrics_aggregate_the_wait_prices(monkeypatch):
    svc = _service_with_run(monkeypatch)
    run = svc.list_payment_runs()[0]
    svc.confirm_payment_run(run.run_id, actor="Tomas Reid")
    store = svc._get_store()
    stamped = store.get_payment_run(run.run_id)
    stamped.confirmed_at = "2026-10-03T09:00:00+00:00"
    store.save_payment_run(stamped)
    svc.confirm_payment_run_second(run.run_id, actor="Amara Chen")

    section = svc.metrics_snapshot()["payment_runs"]["four_eyes_wait"]
    assert section == {
        "runs_second_confirmed": 1,
        "cost_by_currency": {"USD": DISCOUNT},
        "discounts_lapsed": 1,
    }
    # The pure aggregation agrees with the service's section.
    assert handoff_metrics(svc.list_payment_runs())["four_eyes_wait"] == section


def test_the_api_run_carries_the_price(monkeypatch):
    svc = _service_with_run(monkeypatch)
    run = svc.list_payment_runs()[0]
    client = TestClient(create_app(svc))
    client.post(f"/payment-runs/{run.run_id}/confirm", json={"actor": "Tomas Reid"})
    store = svc._get_store()
    stamped = store.get_payment_run(run.run_id)
    stamped.confirmed_at = "2026-10-03T09:00:00+00:00"
    store.save_payment_run(stamped)
    res = client.post(
        f"/payment-runs/{run.run_id}/confirm-second", json={"actor": "Amara Chen"}
    )
    assert res.status_code == 200
    assert res.json()["four_eyes_wait_cost"] == DISCOUNT
