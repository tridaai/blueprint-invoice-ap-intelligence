"""Forecast-aware four-eyes waits: the configured aging threshold
is the patience for a run whose discounts are safe. A run whose
confirmation-time drift forecast already flags members lapsing
before the expected export has no patience to spare — the second
confirmation is the only thing standing between the run and its
export — so its threshold tightens to 0 days, it joins the queue
on the first day of the wait, and the item carries the projected
lapse value the wait puts at risk. The digest says the same thing
in its own sections: a lapse on an unseconded run is *blocked on
the wait*."""

from __future__ import annotations

from datetime import date

from ap_agent.digest import build_digest
from ap_agent.queue import four_eyes_aging
from ap_agent.schemas import PaymentRun, PaymentRunForecast, QueueView
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

AS_OF = date(2026, 10, 10)


def _forecast(*amounts: float) -> list[PaymentRunForecast]:
    return [
        PaymentRunForecast(
            invoice_id=f"INV-F{i}",
            supplier_name="Northbeam Office Supplies",
            discount_amount=amount,
            discount_deadline="2026-10-12",
            expected_export_date="2026-10-15",
        )
        for i, amount in enumerate(amounts)
    ]


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


def test_a_forecast_flagged_run_is_chased_from_the_first_day():
    flagged = _run(
        run_id="RUN-FLAGGED",
        confirmed_at="2026-10-09T09:30:00+00:00",  # one day of wait
        drift_forecast=_forecast(23.60, 10.00),
        drift_forecast_basis="forecast from the measured proposal→export lag",
    )
    plain = _run(run_id="RUN-PLAIN", confirmed_at="2026-10-09T09:30:00+00:00")
    items = four_eyes_aging([flagged, plain], as_of=AS_OF, aging_days=2)
    # Same wait, different treatment: the flagged run is already
    # an action item; the plain run is still inside its allowance.
    assert [item.run_id for item in items] == ["RUN-FLAGGED"]
    item = items[0]
    assert item.forecast_tightened is True
    assert item.aging_threshold_days == 0
    assert item.forecast_lapse_count == 2
    assert item.forecast_value_at_risk == 33.60
    assert item.age_days == 1


def test_an_unflagged_run_keeps_the_configured_threshold():
    plain = _run(run_id="RUN-PLAIN", confirmed_at="2026-10-09T09:30:00+00:00")
    assert four_eyes_aging([plain], as_of=AS_OF, aging_days=2) == []
    items = four_eyes_aging([plain], as_of=date(2026, 10, 12), aging_days=2)
    assert len(items) == 1
    assert items[0].forecast_tightened is False
    assert items[0].aging_threshold_days == 2
    assert items[0].forecast_value_at_risk == 0.0


def test_tightening_still_respects_the_wait_itself():
    # Confirmed today: age 0 is not past even a 0-day threshold —
    # the chase starts on the first *day* of waiting, not the
    # first minute. And a second-confirmed run is done waiting
    # however dire its forecast was.
    today = _run(
        run_id="RUN-TODAY",
        confirmed_at="2026-10-10T09:30:00+00:00",
        drift_forecast=_forecast(5.0),
    )
    done = _run(
        run_id="RUN-DONE",
        confirmed_at="2026-10-01T09:30:00+00:00",
        second_confirmed_by="Tomas Reid",
        second_confirmed_at="2026-10-03T09:30:00+00:00",
        drift_forecast=_forecast(5.0),
    )
    assert four_eyes_aging([today, done], as_of=AS_OF, aging_days=2) == []


def test_value_at_risk_outranks_raw_age_in_the_work_order():
    old_plain = _run(run_id="RUN-OLD-PLAIN", confirmed_at="2026-10-01T09:30:00+00:00")
    young_flagged = _run(
        run_id="RUN-YOUNG-FLAGGED",
        confirmed_at="2026-10-09T09:30:00+00:00",
        drift_forecast=_forecast(5.0),
    )
    items = four_eyes_aging(
        [old_plain, young_flagged], as_of=AS_OF, aging_days=2
    )
    assert [item.run_id for item in items] == ["RUN-YOUNG-FLAGGED", "RUN-OLD-PLAIN"]


def test_the_digest_names_the_wait_as_the_blocker():
    run = _run(
        run_id="RUN-BLOCKED",
        confirmed_at="2026-10-10T09:00:00+00:00",
        drift_forecast=_forecast(23.60),
        drift_forecast_basis="forecast from the measured proposal→export lag",
    )
    digest = build_digest(
        as_of=AS_OF, queue=QueueView(as_of=AS_OF.isoformat()), runs=[run]
    )
    assert len(digest.forecast_lapses) == 1
    assert digest.forecast_lapses[0].blocked_on_four_eyes is True
    assert "blocked on the wait" in digest.summary
    # Second-confirmed, the same forecast is a plain projection.
    run.second_confirmed_by = "Tomas Reid"
    digest = build_digest(
        as_of=AS_OF, queue=QueueView(as_of=AS_OF.isoformat()), runs=[run]
    )
    assert digest.forecast_lapses[0].blocked_on_four_eyes is False
    assert "blocked on the wait" not in digest.summary


def test_service_queue_tightens_only_the_flagged_run():
    store = InMemoryStore()
    store.save_payment_run(
        _run(
            run_id="RUN-FLAGGED",
            confirmed_at="2026-10-09T09:30:00+00:00",
            drift_forecast=_forecast(23.60),
        )
    )
    store.save_payment_run(
        _run(run_id="RUN-PLAIN", confirmed_at="2026-10-09T09:30:00+00:00")
    )
    svc = InvoiceService(store=store, checkpointer=False)
    queue = svc.queue_view(as_of=AS_OF.isoformat())
    assert queue.four_eyes_aging_days == 2  # the configured threshold stands
    assert queue.four_eyes_pending_count == 1
    item = queue.four_eyes_pending[0]
    assert item.run_id == "RUN-FLAGGED"
    assert item.forecast_tightened is True
    assert item.forecast_value_at_risk == 23.60
