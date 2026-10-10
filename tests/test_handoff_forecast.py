"""Drift forecasting at confirmation: the drift metrics showed
the leak after the fact — runs sit between proposal and export,
and members whose discount windows close in that sit lose money.
At confirmation the same measured lag is projected forward: a
discounted member whose deadline falls before the run's expected
export date (proposal date + the measured mean proposal→export
lag) is flagged on the run, labelled a forecast. The export-time
drift check remains the authority."""

from __future__ import annotations

from datetime import date

from ap_agent.paymentruns import drift_forecast
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import PaymentRun, PaymentRunLine
from ap_agent.service import InvoiceService
from ap_agent.storage import InMemoryObjectStore
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
LAG = {"count": 4, "mean_days": 6.0, "max_days": 9.0}


def _run(**overrides) -> PaymentRun:
    base = {
        "run_id": "RUN-USD-2026-10-05-2026-10-11",
        "currency": "USD",
        "status": "proposed",
        "proposed_at": "2026-10-05T09:00:00+00:00",
        "lines": [
            PaymentRunLine(
                invoice_id="INV-1001",
                supplier_name="Northbeam Office Supplies",
                discount_amount=23.60,
                discount_deadline="2026-10-08",
                gross_amount=1180.0,
                net_amount=1156.40,
            ),
            PaymentRunLine(
                invoice_id="INV-1008",
                supplier_name="Northbeam Office Supplies",
                discount_amount=10.0,
                discount_deadline="2026-10-20",
                gross_amount=500.0,
                net_amount=490.0,
            ),
            PaymentRunLine(
                invoice_id="INV-1009",
                supplier_name="Cascade IT Hardware",
                discount_amount=0.0,
                gross_amount=100.0,
                net_amount=100.0,
            ),
        ],
        "invoice_count": 3,
    }
    base.update(overrides)
    return PaymentRun(**base)


def test_forecast_flags_only_members_inside_the_expected_export():
    entries, basis = drift_forecast(
        _run(), as_of=date(2026, 10, 5), lag_stats=LAG
    )
    # Expected export = Oct 5 + 6d = Oct 11. INV-1001's deadline
    # (Oct 8) falls inside it; INV-1008's (Oct 20) does not; the
    # undiscounted member is not forecast material.
    assert [e.invoice_id for e in entries] == ["INV-1001"]
    entry = entries[0]
    assert entry.discount_amount == 23.60
    assert entry.discount_deadline == "2026-10-08"
    assert entry.expected_export_date == "2026-10-11"
    assert entry.headroom_days == 3
    assert "projected to lapse" in entry.message
    assert "not a promise" in entry.message
    assert "mean 6.00 days over 4 exported run(s)" in basis


def test_forecast_needs_measured_lag_and_a_proposal_date():
    entries, basis = drift_forecast(_run(), as_of=date(2026, 10, 5), lag_stats=None)
    assert entries == [] and "no exported runs" in basis
    entries, basis = drift_forecast(
        _run(proposed_at=""), as_of=date(2026, 10, 5), lag_stats=LAG
    )
    assert entries == [] and "no proposal date" in basis


def test_confirmation_stamps_the_forecast_from_stored_history():
    store = InMemoryStore()
    # One exported run in the store's history: proposed Oct 1,
    # exported Oct 11 — a measured proposal→export lag of 10 days.
    store.save_payment_run(
        PaymentRun(
            run_id="RUN-USD-HISTORY",
            currency="USD",
            status="confirmed",
            proposed_at="2026-10-01T09:00:00+00:00",
            confirmed_at="2026-10-01T10:00:00+00:00",
            handoff_generated_at="2026-10-11T09:00:00+00:00",
            handoff_format="csv",
        )
    )
    svc = InvoiceService(
        store=store, checkpointer=False, object_store=InMemoryObjectStore()
    )
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    run = svc.propose_payment_runs(window_days=7, as_of="2026-10-05")[0]
    # Pin the proposal stamp so the projection is deterministic:
    # proposed Oct 5 + 10d lag → expected export Oct 15, past
    # INV-1001's Oct 8 discount deadline.
    run.proposed_at = "2026-10-05T09:00:00+00:00"
    store.save_payment_run(run)
    confirmed = svc.confirm_payment_run(run.run_id, actor="Tomas Reid")
    assert [e.invoice_id for e in confirmed.drift_forecast] == ["INV-1001"]
    entry = confirmed.drift_forecast[0]
    assert entry.expected_export_date == "2026-10-15"
    assert entry.discount_deadline == "2026-10-08"
    assert "mean 10.00 days over 1 exported run(s)" in confirmed.drift_forecast_basis
    assert "Drift forecast at confirmation" in confirmed.note
    assert "projection" in confirmed.note
    # The forecast changed no figures.
    assert confirmed.total_discount == 23.60
    assert confirmed.lines[0].discount_amount == 23.60


def test_confirmation_without_history_forecasts_nothing():
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        object_store=InMemoryObjectStore(),
    )
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    run = svc.propose_payment_runs(window_days=7, as_of="2026-10-05")[0]
    confirmed = svc.confirm_payment_run(run.run_id, actor="Tomas Reid")
    assert confirmed.drift_forecast == []
    assert "no exported runs" in confirmed.drift_forecast_basis
    assert "Drift forecast" not in confirmed.note
