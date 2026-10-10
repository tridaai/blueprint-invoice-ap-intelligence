"""Hand-off metrics in /metrics: the drift value captured at
export and the proposal→export / confirm→export lag stats, so the
discount discipline is measurable instead of anecdotal. The lag
is the leak that causes the drift, so the two are reported side
by side (paymentruns.handoff_metrics, surfaced by the service)."""

from __future__ import annotations

from ap_agent.paymentruns import handoff_metrics
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import PaymentRun, PaymentRunDrift
from ap_agent.service import InvoiceService
from ap_agent.storage import InMemoryObjectStore
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


def _run(**overrides) -> PaymentRun:
    base = {
        "run_id": "RUN-USD-2026-10-05-2026-10-11",
        "currency": "USD",
        "status": "confirmed",
    }
    base.update(overrides)
    return PaymentRun(**base)


def test_lag_stats_are_computed_from_the_receipt_timestamps():
    run = _run(
        proposed_at="2026-10-05T09:00:00+00:00",
        confirmed_at="2026-10-05T09:30:00+00:00",
        handoff_generated_at="2026-10-07T09:30:00+00:00",
        handoff_format="csv",
    )
    metrics = handoff_metrics([run])
    assert metrics["exported_runs"] == 1
    assert metrics["exports_by_format"] == {"csv": 1}
    assert metrics["export_lag"]["confirm_to_export"] == {
        "count": 1,
        "mean_days": 2.0,
        "max_days": 2.0,
    }
    assert metrics["export_lag"]["proposal_to_export"]["mean_days"] == 2.02


def test_unexported_and_unstamped_runs_contribute_nothing_to_lags():
    metrics = handoff_metrics([_run(status="proposed"), _run()])
    assert metrics["runs_total"] == 2
    assert metrics["by_status"] == {"proposed": 1, "confirmed": 1}
    assert metrics["exported_runs"] == 0
    assert metrics["export_lag"]["confirm_to_export"] is None
    assert metrics["export_lag"]["proposal_to_export"] is None


def test_drift_value_aggregates_per_currency():
    drift = PaymentRunDrift(
        invoice_id="INV-1001",
        discount_deadline="2026-10-08",
        discount_at_proposal=23.60,
        discount_at_export=0.0,
        discount_lost=23.60,
        reason="window closed",
    )
    runs = [
        _run(handoff_drift=[drift], handoff_drift_total=23.60),
        _run(
            run_id="RUN-EUR-2026-10-05-2026-10-11",
            currency="EUR",
            handoff_drift=[drift],
            handoff_drift_total=10.00,
        ),
        _run(run_id="RUN-USD-CLEAN"),
    ]
    metrics = handoff_metrics(runs)
    assert metrics["drift"] == {
        "runs_with_drift": 2,
        "members_drifted": 2,
        "value_lost_by_currency": {"USD": 23.60, "EUR": 10.00},
    }


def test_service_metrics_carry_the_handoff_section():
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        object_store=InMemoryObjectStore(),
    )
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    runs = svc.propose_payment_runs(window_days=7, as_of="2026-10-05")
    svc.confirm_payment_run(runs[0].run_id, actor="Tomas Reid")
    # Exported after the discount deadline: drift is captured.
    svc.export_payment_run_handoff(
        runs[0].run_id, actor="Tomas Reid", as_of="2026-10-20", format="pain001"
    )
    metrics = svc.metrics_snapshot()
    section = metrics["payment_runs"]
    assert section["runs_total"] == 1
    assert section["exported_runs"] == 1
    assert section["exports_by_format"] == {"pain001": 1}
    assert section["drift"]["runs_with_drift"] == 1
    assert section["drift"]["members_drifted"] == 1
    assert section["drift"]["value_lost_by_currency"] == {"USD": 23.60}
    assert section["export_lag"]["confirm_to_export"]["count"] == 1
    assert section["export_lag"]["proposal_to_export"]["count"] == 1
