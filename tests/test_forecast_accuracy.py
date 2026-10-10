"""Forecast-accuracy tracking: the confirmation-time drift
forecast was a projection nobody graded. /metrics now scores it
against the export-time drift the runs actually measured —
flagged members that drifted, flagged members the warning saved,
and the misses no forecast named — so the lag statistic the
forecast rests on can be tuned, or the forecast honestly retired,
from evidence instead of impressions."""

from __future__ import annotations

from ap_agent.paymentruns import forecast_accuracy, handoff_metrics
from ap_agent.schemas import (
    PaymentRun,
    PaymentRunDrift,
    PaymentRunForecast,
    PaymentRunLine,
)
from ap_agent.service import InvoiceService
from ap_agent.storage import InMemoryObjectStore
from ap_agent.store import InMemoryStore

BASIS = (
    "forecast from the measured proposal→export lag: mean 6.00 "
    "days over 4 exported run(s); this run proposed 2026-10-05, "
    "so its expected export date is 2026-10-11"
)


def _forecast(invoice_id: str, amount: float) -> PaymentRunForecast:
    return PaymentRunForecast(
        invoice_id=invoice_id,
        discount_amount=amount,
        discount_deadline="2026-10-08",
        expected_export_date="2026-10-11",
    )


def _drift(invoice_id: str, lost: float) -> PaymentRunDrift:
    return PaymentRunDrift(
        invoice_id=invoice_id,
        discount_deadline="2026-10-08",
        discount_at_proposal=lost,
        discount_at_export=0.0,
        discount_lost=lost,
        reason="window closed",
    )


def _scored_run(**overrides) -> PaymentRun:
    """One exported run with one member in each cell of the
    two-by-two: INV-1 flagged+drifted, INV-2 flagged+saved,
    INV-3 unflagged+drifted, INV-4 unflagged+saved."""
    base = {
        "run_id": "RUN-USD-2026-10-05-2026-10-11",
        "currency": "USD",
        "status": "confirmed",
        "proposed_at": "2026-10-05T09:00:00+00:00",
        "confirmed_at": "2026-10-05T10:00:00+00:00",
        "handoff_generated_at": "2026-10-12T09:00:00+00:00",
        "handoff_format": "csv",
        "drift_forecast_basis": BASIS,
        "drift_forecast": [_forecast("INV-1", 20.0), _forecast("INV-2", 10.0)],
        "handoff_drift": [_drift("INV-1", 20.0), _drift("INV-3", 5.0)],
        "handoff_drift_total": 25.0,
        "lines": [
            # Export-time figures: INV-1 and INV-3 were re-priced
            # to full gross (discount 0); INV-2 and INV-4 kept theirs.
            PaymentRunLine(invoice_id="INV-1", gross_amount=200.0, discount_amount=0.0, net_amount=200.0),
            PaymentRunLine(invoice_id="INV-2", gross_amount=100.0, discount_amount=10.0, net_amount=90.0),
            PaymentRunLine(invoice_id="INV-3", gross_amount=50.0, discount_amount=0.0, net_amount=50.0),
            PaymentRunLine(invoice_id="INV-4", gross_amount=80.0, discount_amount=8.0, net_amount=72.0),
            PaymentRunLine(invoice_id="INV-5", gross_amount=60.0, discount_amount=0.0, net_amount=60.0),
        ],
        "invoice_count": 5,
    }
    base.update(overrides)
    return PaymentRun(**base)


def test_the_two_by_two_counts_every_discounted_member_once():
    accuracy = forecast_accuracy([_scored_run()])
    assert accuracy["scored_runs"] == 1
    assert accuracy["unscored_runs"] == 0
    # INV-5 carried no discount at confirmation: not forecast
    # material, in no cell.
    assert accuracy["members"] == {
        "flagged_drifted": 1,
        "flagged_saved": 1,
        "unflagged_drifted": 1,
        "unflagged_saved": 1,
    }
    assert accuracy["precision"] == 0.5  # 1 of 2 flags lapsed
    assert accuracy["recall"] == 0.5  # 1 of 2 lapses was flagged
    assert accuracy["value_flagged"] == 30.0
    assert accuracy["value_lost_flagged"] == 20.0
    assert accuracy["value_lost_unflagged"] == 5.0


def test_unscored_runs_report_their_drift_separately():
    # Confirmed before any lag existed: the basis honestly says no
    # forecast was made, so its drift cannot be scored as a miss.
    legacy = _scored_run(
        run_id="RUN-USD-LEGACY",
        drift_forecast=[],
        drift_forecast_basis=(
            "no forecast: no exported runs yet, so there is no "
            "measured proposal→export lag to project from"
        ),
    )
    # Never confirmed through the forecast path at all (pre-round-8
    # export): empty basis, no forecast entries.
    ancient = _scored_run(
        run_id="RUN-USD-ANCIENT", drift_forecast=[], drift_forecast_basis=""
    )
    accuracy = forecast_accuracy([legacy, ancient])
    assert accuracy["scored_runs"] == 0
    assert accuracy["unscored_runs"] == 2
    assert accuracy["members"] == {
        "flagged_drifted": 0,
        "flagged_saved": 0,
        "unflagged_drifted": 0,
        "unflagged_saved": 0,
    }
    assert accuracy["precision"] is None
    assert accuracy["recall"] is None
    # Both runs drifted INV-1 (20.0) and INV-3 (5.0).
    assert accuracy["unscored_drift"] == {"members": 4, "value_lost": 50.0}


def test_unexported_runs_are_not_scored():
    pending = _scored_run(handoff_generated_at=None, handoff_format="")
    accuracy = forecast_accuracy([pending])
    assert accuracy["scored_runs"] == 0
    assert accuracy["unscored_runs"] == 0
    assert accuracy["members"]["flagged_drifted"] == 0


def test_partial_repricing_counts_as_drift_and_currencies_split():
    partial = PaymentRunDrift(
        invoice_id="INV-9",
        discount_deadline="2026-10-08",
        discount_at_proposal=12.0,
        discount_at_export=4.0,  # re-priced, not zeroed — still drift
        discount_lost=8.0,
        reason="re-priced at export",
    )
    eur = _scored_run(
        run_id="RUN-EUR-2026-10-05-2026-10-11",
        currency="EUR",
        drift_forecast=[],
        handoff_drift=[partial],
        handoff_drift_total=8.0,
        lines=[
            PaymentRunLine(invoice_id="INV-9", gross_amount=120.0, discount_amount=4.0, net_amount=116.0)
        ],
        invoice_count=1,
    )
    accuracy = forecast_accuracy([_scored_run(), eur])
    assert accuracy["by_currency"]["USD"]["flagged_drifted"] == 1
    eur_cell = accuracy["by_currency"]["EUR"]
    assert eur_cell["unflagged_drifted"] == 1
    assert eur_cell["value_lost_unflagged"] == 8.0
    assert eur_cell["precision"] is None  # nothing was flagged in EUR
    assert eur_cell["recall"] == 0.0  # the one lapse was missed
    assert accuracy["members"]["unflagged_drifted"] == 2


def test_a_zero_lost_drift_entry_is_not_drift():
    # Defensive: an entry that lost nothing (should not occur —
    # discount_drift only itemises real movement) must not poison
    # the recall denominator.
    run = _scored_run(
        handoff_drift=[_drift("INV-1", 20.0), _drift("INV-2", 0.0)],
        drift_forecast=[_forecast("INV-1", 20.0), _forecast("INV-2", 10.0)],
    )
    accuracy = forecast_accuracy([run])
    assert accuracy["members"]["flagged_drifted"] == 1
    assert accuracy["members"]["flagged_saved"] == 1


def test_metrics_snapshot_carries_the_accuracy_section():
    store = InMemoryStore()
    store.save_payment_run(_scored_run())
    svc = InvoiceService(
        store=store, checkpointer=False, object_store=InMemoryObjectStore()
    )
    section = svc.metrics_snapshot()["payment_runs"]
    assert section["forecast_accuracy"]["scored_runs"] == 1
    assert section["forecast_accuracy"]["members"]["unflagged_drifted"] == 1
    # And it rides along inside handoff_metrics itself.
    assert handoff_metrics([_scored_run()])["forecast_accuracy"]["recall"] == 0.5
