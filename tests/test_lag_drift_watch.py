"""Lag-grade drift watch: a statistic's grade is a claim about
the present, and the present moves.

The forecast's selection trusted one pooled grade per statistic.
A segment that caught every lapse for a quarter can start
missing them — the bank slows, the profile's desk changes — and
the pooled recall dilutes the decay for months while the
selection keeps crowning the old record. The watch windows the
same leave-one-out replay by export order: recent recall
against the earlier record. A fall of 0.25 or more names the
statistic degrading in /metrics and demotes it in the
selection: the forecast falls to the best standing graded
candidate (or the most specific measured segment when none
stands), with the demotion in the basis.

Scenario: EUR exports ran 20 days with deadlines at +10 — the
EUR segment (and the global pool) flagged every one, and every
one drifted. The last five EUR exports carry deadlines at +25
and drifted anyway: the segment's mean never saw them coming.
A small fast profile (HP-FAST, 6 days) holds a mixed but
standing record (recall 0.50) over the same period.
"""

from __future__ import annotations

from datetime import date

from ap_agent.paymentruns import (
    drift_forecast,
    grade_lag_statistics,
    handoff_metrics,
    lag_grade_drift_watch,
    lag_statistics,
)
from ap_agent.schemas import PaymentRun, PaymentRunDrift, PaymentRunLine
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore


def _run(
    run_id: str,
    *,
    currency: str,
    proposed: str,
    exported: str,
    deadline: str,
    drifted: bool,
    profile_id: str = "",
) -> PaymentRun:
    return PaymentRun(
        run_id=run_id,
        currency=currency,
        status="confirmed",
        proposed_at=proposed,
        confirmed_at=proposed,
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
    runs: list[PaymentRun] = []
    # Earlier record: eight slow EUR exports, deadlines inside
    # the segment's lag, all drifted — the segment flagged all.
    for i in range(8):
        runs.append(
            _run(
                f"EUR-EARLY-{i}",
                currency="EUR",
                proposed="2026-09-01T09:00:00+00:00",
                exported="2026-09-21T09:00:00+00:00",
                deadline="2026-09-11",
                drifted=True,
            )
        )
    # The fast profile's mixed record over the same weeks.
    for i, (deadline, drifted) in enumerate(
        [("2026-09-05", True), ("2026-09-05", True),
         ("2026-09-13", True), ("2026-09-13", True)]
    ):
        runs.append(
            _run(
                f"HP-{i}",
                currency="EUR",
                proposed="2026-09-01T09:00:00+00:00",
                exported="2026-09-07T09:00:00+00:00",
                deadline=deadline,
                drifted=drifted,
                profile_id="HP-FAST",
            )
        )
    # Quiet USD exports: saved, no drift — no recall signal.
    for i in range(4):
        runs.append(
            _run(
                f"USD-{i}",
                currency="USD",
                proposed="2026-09-01T09:00:00+00:00",
                exported="2026-09-03T09:00:00+00:00",
                deadline="2026-10-01",
                drifted=False,
            )
        )
    # The recent window: five EUR exports whose members drifted
    # with deadlines past every segment's expected export — the
    # misses the pooled grade was hiding.
    for i in range(5):
        runs.append(
            _run(
                f"EUR-LATE-{i}",
                currency="EUR",
                proposed="2026-10-01T09:00:00+00:00",
                exported="2026-10-21T09:00:00+00:00",
                deadline="2026-10-26",
                drifted=True,
            )
        )
    return runs


def _target() -> PaymentRun:
    return PaymentRun(
        run_id="EUR-NEW",
        currency="EUR",
        status="proposed",
        proposed_at="2026-11-01T09:00:00+00:00",
        invoice_count=1,
        lines=[
            PaymentRunLine(
                invoice_id="INV-NEW-1",
                supplier_name="Supplier",
                discount_amount=40.0,
                discount_deadline="2026-11-13",
                gross_amount=800.0,
                net_amount=760.0,
            )
        ],
    )


# -- the watch ------------------------------------------------------------


def test_watch_names_the_decayed_statistics():
    watch = lag_grade_drift_watch(_history())
    eur = watch["currency:EUR"]
    assert eur["assessed"] is True
    assert eur["earlier_recall"] == 1.0
    assert eur["recent_recall"] == 0.0
    assert eur["recent_graded_members"] == 5
    assert eur["degrading"] is True
    # The global pool drank from the same stream: demoted too.
    assert watch["global"]["degrading"] is True


def test_watch_leaves_standing_and_thin_statistics_alone():
    watch = lag_grade_drift_watch(_history())
    profile = watch["profile:HP-FAST"]
    # No recent members: nothing to assess, and crucially the
    # statistic is NOT degrading — it stands for the selection.
    assert profile["assessed"] is False
    assert profile["degrading"] is False
    assert profile["overall_recall"] == 0.5
    # USD never drifted: no recall exists on either side.
    usd = watch["currency:USD"]
    assert usd["assessed"] is False and usd["degrading"] is False


def test_metrics_carry_the_watch():
    metrics = handoff_metrics(_history())
    assert metrics["lag_grade_drift_watch"]["currency:EUR"]["degrading"] is True
    assert metrics["lag_statistic_grades"]["currency:EUR"]["recall"] == 0.7059


# -- the demotion in the selection -----------------------------------------


def test_without_the_watch_the_decayed_segment_still_wins():
    runs = _history()
    entries, basis = drift_forecast(
        _target(),
        as_of=date(2026, 11, 1),
        lag_segments=lag_statistics(runs),
        lag_grades=grade_lag_statistics(runs),
        profile_id="HP-FAST",
    )
    # The pooled grade (recall 0.71) crowns the currency segment,
    # whose 16.7-day mean flags the +12 deadline.
    assert [e.invoice_id for e in entries] == ["INV-NEW-1"]
    assert "per-currency (EUR)" in basis
    assert "drift watch" not in basis


def test_demotion_falls_to_the_standing_profile():
    runs = _history()
    entries, basis = drift_forecast(
        _target(),
        as_of=date(2026, 11, 1),
        lag_segments=lag_statistics(runs),
        lag_grades=grade_lag_statistics(runs),
        lag_drift_watch=lag_grade_drift_watch(runs),
        profile_id="HP-FAST",
    )
    # The profile's 6-day mean expects export on Nov 7, ahead of
    # the Nov 13 deadline: no lapse is projected, and the basis
    # says why the better-recorded segments did not govern.
    assert entries == []
    assert "per-profile (HP-FAST)" in basis
    assert "demoted by the lag-grade drift watch" in basis
    assert "recent recall 0.00" in basis
    assert basis.startswith("forecast from")


def test_confirmation_names_the_demotion_when_no_candidate_stands():
    store = InMemoryStore()
    for run in _history():
        store.save_payment_run(run)
    store.save_payment_run(_target())
    service = InvoiceService(store=store)
    confirmed = service.confirm_payment_run("EUR-NEW", actor="finance-lead")
    # Ungoverned target: both graded candidates (currency and
    # global) are demoted, so the forecast falls back to the
    # most specific measured segment — with the demotions named
    # on the run's own basis.
    assert "demoted by the lag-grade drift watch" in confirmed.drift_forecast_basis
    assert "no graded candidate stands" in confirmed.drift_forecast_basis
    assert "per-currency (EUR)" in confirmed.drift_forecast_basis
