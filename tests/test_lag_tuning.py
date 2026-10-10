"""Lag-statistic tuning: the drift forecast's lag is selected and
graded, not static.

The forecast always projected from the global mean proposal→export
lag. That pool mixes unlike work: in the scenario below, EUR runs
take twenty days to export while USD runs take two — the global
mean (11) understates exactly the runs most likely to drift, and
the EUR members' discounts lapsed unflagged, export after export.
These tests pin the tuning: lag statistics segmented by currency
and governing profile, each candidate graded against the accuracy
ledger by replaying its flag rule leave-one-out, and the forecast
selecting the statistic with the best recorded recall — its basis
stating which statistic it used and why.
"""

from __future__ import annotations

import sys
from datetime import date

import pytest

from ap_agent.paymentruns import (
    drift_forecast,
    grade_lag_statistics,
    handoff_metrics,
    lag_statistics,
)
from ap_agent.schemas import PaymentRun, PaymentRunDrift, PaymentRunLine
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

PROPOSED = "2026-09-01T09:00:00+00:00"


def _exported(
    run_id: str,
    *,
    currency: str,
    lag_days: int,
    deadline: str,
    drifted: bool,
    profile_id: str = "",
) -> PaymentRun:
    exported = f"2026-09-{1 + lag_days:02d}T09:00:00+00:00"
    return PaymentRun(
        run_id=run_id,
        currency=currency,
        status="confirmed",
        proposed_at=PROPOSED,
        confirmed_at=PROPOSED,
        handoff_generated_at=exported,
        handoff_profile_id=profile_id,
        invoice_count=1,
        lines=[
            PaymentRunLine(
                invoice_id=f"{run_id}-M1",
                supplier_name="Supplier",
                discount_amount=50.0,
                discount_deadline=deadline,
                gross_amount=1000.0,
                net_amount=950.0,
            )
        ],
        handoff_drift=(
            [
                PaymentRunDrift(
                    invoice_id=f"{run_id}-M1",
                    discount_deadline=deadline,
                    discount_lost=50.0,
                )
            ]
            if drifted
            else []
        ),
    )


def _history() -> list[PaymentRun]:
    """Four slow EUR exports (20d, members drifted) and four fast
    USD exports (2d, members saved) — the mixed pool the global
    mean misreads."""
    runs = []
    for i in range(4):
        runs.append(
            _exported(
                f"EUR-{i}",
                currency="EUR",
                lag_days=20,
                deadline="2026-09-11",
                drifted=True,
                profile_id="HP-CUR-EUR",
            )
        )
        runs.append(
            _exported(
                f"USD-{i}",
                currency="USD",
                lag_days=2,
                deadline="2026-10-01",
                drifted=False,
            )
        )
    return runs


def _new_eur_run() -> PaymentRun:
    return PaymentRun(
        run_id="EUR-NEW",
        currency="EUR",
        status="proposed",
        proposed_at="2026-10-01T09:00:00+00:00",
        invoice_count=1,
        lines=[
            PaymentRunLine(
                invoice_id="INV-NEW-1",
                supplier_name="Supplier",
                discount_amount=40.0,
                discount_deadline="2026-10-15",
                gross_amount=800.0,
                net_amount=760.0,
            )
        ],
    )


# -- segmentation -------------------------------------------------------


def test_lag_statistics_segment_by_currency_and_profile():
    stats = lag_statistics(_history())
    assert stats["global"]["count"] == 8
    assert stats["global"]["mean_days"] == 11.0
    assert stats["by_currency"]["EUR"]["mean_days"] == 20.0
    assert stats["by_currency"]["USD"]["mean_days"] == 2.0
    assert stats["by_profile"]["HP-CUR-EUR"]["count"] == 4
    # Ungoverned exports join no profile segment.
    assert set(stats["by_profile"]) == {"HP-CUR-EUR"}


def test_metrics_carry_segments_and_grades():
    metrics = handoff_metrics(_history())
    lag = metrics["export_lag"]
    assert lag["proposal_to_export_by_currency"]["EUR"]["mean_days"] == 20.0
    assert lag["proposal_to_export_by_profile"]["HP-CUR-EUR"]["count"] == 4
    assert metrics["lag_statistic_grades"]["currency:EUR"]["recall"] == 1.0


# -- grading ------------------------------------------------------------


def test_grades_replay_each_statistic_against_the_ledger():
    grades = grade_lag_statistics(_history())
    eur = grades["currency:EUR"]
    # Under its own segment's lag (20d), every EUR member's
    # deadline fell inside the expected export — all four flagged,
    # all four drifted.
    assert eur["flagged_drifted"] == 4
    assert eur["recall"] == 1.0 and eur["precision"] == 1.0
    assert eur["graded_members"] == 4
    glob = grades["global"]
    # Under the global mean (leave-one-out ≈ 9.7d for an EUR run),
    # the same members were never flagged — the misses the tuning
    # exists to catch.
    assert glob["unflagged_drifted"] == 4
    assert glob["recall"] == 0.0
    assert glob["graded_members"] == 8


def test_grading_is_leave_one_out():
    single = [
        _exported(
            "EUR-0",
            currency="EUR",
            lag_days=20,
            deadline="2026-09-11",
            drifted=True,
        )
    ]
    # One exported run grades nothing: every candidate segment
    # would be the run grading itself.
    assert grade_lag_statistics(single) == {}


# -- selection inside the forecast --------------------------------------


def test_forecast_selects_the_currency_segment_the_ledger_vouches_for():
    runs = _history()
    entries, basis = drift_forecast(
        _new_eur_run(),
        as_of=date(2026, 10, 1),
        lag_segments=lag_statistics(runs),
        lag_grades=grade_lag_statistics(runs),
    )
    # The global mean (11d) would expect export on Oct 12 and miss
    # the Oct 15 deadline entirely; the EUR segment (20d) flags it.
    assert [e.invoice_id for e in entries] == ["INV-NEW-1"]
    assert entries[0].expected_export_date == "2026-10-21"
    assert "per-currency (EUR)" in basis
    assert "selected on the accuracy ledger" in basis
    assert "recall 1.00" in basis
    # The forecast_accuracy gate keys on this prefix.
    assert basis.startswith("forecast from")


def test_forecast_without_grades_picks_the_most_specific_segment():
    segments = {
        "global": {"count": 10, "mean_days": 5.0, "max_days": 9.0},
        "by_currency": {"EUR": {"count": 4, "mean_days": 20.0, "max_days": 22.0}},
        "by_profile": {"HP-CUR-EUR": {"count": 4, "mean_days": 25.0, "max_days": 26.0}},
    }
    entries, basis = drift_forecast(
        _new_eur_run(),
        as_of=date(2026, 10, 1),
        lag_segments=segments,
        lag_grades={},
        profile_id="HP-CUR-EUR",
    )
    assert "per-profile (HP-CUR-EUR)" in basis
    assert "most specific adequately-measured segment" in basis
    assert entries and entries[0].expected_export_date == "2026-10-26"


def test_forecast_on_a_young_ledger_falls_back_to_the_global_pool():
    segments = {
        "global": {"count": 2, "mean_days": 6.0, "max_days": 7.0},
        "by_currency": {"EUR": {"count": 2, "mean_days": 20.0, "max_days": 20.0}},
        "by_profile": {},
    }
    entries, basis = drift_forecast(
        _new_eur_run(),
        as_of=date(2026, 10, 1),
        lag_segments=segments,
        lag_grades={},
    )
    assert "global" in basis and "only segment with a measured lag" in basis
    # Global mean 6d: expected export Oct 7, before the deadline.
    assert entries == []


def test_forecast_with_no_measured_lag_stays_honest():
    segments = {"global": None, "by_currency": {}, "by_profile": {}}
    entries, basis = drift_forecast(
        _new_eur_run(),
        as_of=date(2026, 10, 1),
        lag_segments=segments,
        lag_grades={},
    )
    assert entries == [] and "no exported runs" in basis


# -- the service wires the selection in at confirmation -----------------


def test_confirmation_selects_from_the_stored_history():
    store = InMemoryStore()
    for run in _history():
        store.save_payment_run(run)
    store.save_payment_run(_new_eur_run())
    service = InvoiceService(store=store)
    confirmed = service.confirm_payment_run("EUR-NEW", actor="finance-lead")
    assert [e.invoice_id for e in confirmed.drift_forecast] == ["INV-NEW-1"]
    assert "per-currency (EUR)" in confirmed.drift_forecast_basis
    assert "accuracy ledger" in confirmed.drift_forecast_basis


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
