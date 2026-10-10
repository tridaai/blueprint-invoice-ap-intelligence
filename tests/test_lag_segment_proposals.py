"""Lag-segment proposals: the drift watch's demotion becomes a
governed decision.

The watch demotes a decaying lag statistic inside the
forecast's selection, run by run — automatic, silent, and
lifted the moment the recent window turns over. Nobody ever
DECIDED the segment's fate. The proposals are that decision,
under the profile-change discipline (drafted from the evidence
by one operator, decided by a second, applied only on
approval): a degrading statistic drafts its retirement from the
candidate set; an excluded segment whose recent record
recovered drafts its re-grade back in. Which segments stand
excluded is derived from the proposal ledger itself — the
latest approved decision per segment — so the ledger is the
state and its own audit trail. An approval whose evidence
moved since the draft (the statistic recovered; the segment
was never out) is refused, to be re-drafted against a fresh
watch.

Scenario (from test_lag_drift_watch): EUR exports ran 20 days
with deadlines at +10 — flagged, and drifted. The last five
drifted with deadlines past every segment's expectation: the
EUR statistic and the global pool both read degrading.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.lag_segment_proposals import (
    excluded_segments,
    next_lag_proposal_id,
    proposals_from_watch,
)
from ap_agent.paymentruns import lag_grade_drift_watch
from ap_agent.schemas import LagSegmentProposal, PaymentRun
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore
from test_lag_drift_watch import _history, _run, _target


def _service_with_history() -> InvoiceService:
    store = InMemoryStore()
    for run in _history():
        store.save_payment_run(run)
    return InvoiceService(store=store, checkpointer=False)


def _recovery_runs() -> list[PaymentRun]:
    """Five recent EUR exports the segment DOES catch: 20-day
    lags, deadlines at +10, all drifted — the recent window
    turns over and the decayed statistics read recovered."""
    return [
        _run(
            f"EUR-RECOVERY-{i}",
            currency="EUR",
            proposed="2026-10-16T09:00:00+00:00",
            exported="2026-11-05T09:00:00+00:00",
            deadline="2026-10-26",
            drifted=True,
        )
        for i in range(5)
    ]


def _add_recovery(service: InvoiceService) -> None:
    store = service._get_store()
    for run in _recovery_runs():
        store.save_payment_run(run)


# -- the pure drafting half ---------------------------------------------------


def test_next_ids_are_sequential():
    assert next_lag_proposal_id([]) == "LSP-0001"
    assert next_lag_proposal_id(["LSP-0001", "LSP-0007"]) == "LSP-0008"


def test_exclusion_is_the_latest_approved_decision_per_segment():
    def proposal(pid, segment, kind, status):
        return LagSegmentProposal(
            proposal_id=pid, segment=segment, kind=kind, status=status
        )

    proposals = [
        proposal("LSP-0001", "currency:EUR", "retire_segment", "approved"),
        proposal("LSP-0002", "global", "retire_segment", "rejected"),
        proposal("LSP-0003", "currency:USD", "retire_segment", "pending"),
    ]
    # Approved retires exclude; rejected and pending move nothing.
    assert excluded_segments(proposals) == {"currency:EUR"}
    proposals.append(
        proposal("LSP-0004", "currency:EUR", "regrade_segment", "approved")
    )
    assert excluded_segments(proposals) == set()


def test_watch_drafts_retirements_for_degrading_segments_only():
    watch = lag_grade_drift_watch(_history())
    specs = proposals_from_watch(watch, {}, set())
    by_segment = {spec["segment"]: spec for spec in specs}
    assert set(by_segment) == {"currency:EUR", "global"}
    assert all(
        spec["kind"] == "retire_segment" for spec in specs
    )
    eur = by_segment["currency:EUR"]
    assert eur["evidence"]["earlier_recall"] == 1.0
    assert eur["evidence"]["recent_recall"] == 0.0
    assert "per-currency (EUR)" in eur["basis"]
    # An already-excluded degrading segment drafts nothing new.
    assert proposals_from_watch(watch, {}, {"currency:EUR", "global"}) == []


# -- the governed workflow ----------------------------------------------------


def test_drafting_names_the_degrading_statistics():
    service = _service_with_history()
    drafted = service.draft_lag_segment_proposals(actor="Asha")
    assert {(p.segment, p.kind) for p in drafted} == {
        ("currency:EUR", "retire_segment"),
        ("global", "retire_segment"),
    }
    assert [p.proposal_id for p in drafted] == ["LSP-0001", "LSP-0002"]
    # One open question per segment+kind: a re-draft is a no-op.
    assert service.draft_lag_segment_proposals(actor="Asha") == []
    # Drafting applies nothing.
    assert service.excluded_lag_segments() == []


def test_the_drafter_cannot_decide_their_own_proposal():
    service = _service_with_history()
    (proposal,) = [
        p
        for p in service.draft_lag_segment_proposals(actor="Asha")
        if p.segment == "currency:EUR"
    ]
    with pytest.raises(ValueError, match="four-eyes"):
        service.decide_lag_segment_proposal(
            proposal.proposal_id, actor="Asha", approve=True
        )


def test_an_approved_retirement_excludes_and_the_forecast_says_so():
    service = _service_with_history()
    for proposal in service.draft_lag_segment_proposals(actor="Asha"):
        service.decide_lag_segment_proposal(
            proposal.proposal_id, actor="Boris", approve=True
        )
    assert service.excluded_lag_segments() == ["currency:EUR", "global"]
    # Both candidates for an ungoverned EUR run stand excluded:
    # no statistic may project it, and the basis says why.
    store = service._get_store()
    store.save_payment_run(_target())
    confirmed = service.confirm_payment_run("EUR-NEW", actor="finance-lead")
    assert confirmed.drift_forecast == []
    assert "governed exclusion" in confirmed.drift_forecast_basis
    assert "per-currency (EUR)" in confirmed.drift_forecast_basis
    # Metrics name the state.
    block = service.metrics_snapshot()["lag_segment_proposals"]
    assert block["approved"] == 2
    assert block["excluded_segments"] == ["currency:EUR", "global"]


def test_a_rejection_leaves_the_selection_as_it_was():
    service = _service_with_history()
    for proposal in service.draft_lag_segment_proposals(actor="Asha"):
        service.decide_lag_segment_proposal(
            proposal.proposal_id,
            actor="Boris",
            approve=False,
            note="one bad month is not a retirement",
        )
    assert service.excluded_lag_segments() == []
    store = service._get_store()
    store.save_payment_run(_target())
    confirmed = service.confirm_payment_run("EUR-NEW", actor="finance-lead")
    # The watch's own demotion still applies — rejection refused
    # the standing decision, not the automatic one.
    assert "demoted by the lag-grade drift watch" in (
        confirmed.drift_forecast_basis
    )
    assert "governed exclusion" not in confirmed.drift_forecast_basis


def test_a_retirement_whose_decay_recovered_is_refused():
    service = _service_with_history()
    drafted = service.draft_lag_segment_proposals(actor="Asha")
    _add_recovery(service)
    eur = next(p for p in drafted if p.segment == "currency:EUR")
    with pytest.raises(ValueError, match="no longer reads degrading"):
        service.decide_lag_segment_proposal(
            eur.proposal_id, actor="Boris", approve=True
        )
    # The proposal stands undecided — reject it, don't force it.
    assert service.get_lag_segment_proposal(eur.proposal_id).status == "pending"
    assert service.excluded_lag_segments() == []


def test_a_recovered_segment_is_regraded_back_in():
    service = _service_with_history()
    for proposal in service.draft_lag_segment_proposals(actor="Asha"):
        service.decide_lag_segment_proposal(
            proposal.proposal_id, actor="Boris", approve=True
        )
    assert service.excluded_lag_segments() == ["currency:EUR", "global"]
    _add_recovery(service)
    drafted = service.draft_lag_segment_proposals(actor="Asha")
    assert {(p.segment, p.kind) for p in drafted} == {
        ("currency:EUR", "regrade_segment"),
        ("global", "regrade_segment"),
    }
    for proposal in drafted:
        service.decide_lag_segment_proposal(
            proposal.proposal_id, actor="Boris", approve=True
        )
    assert service.excluded_lag_segments() == []
    # The statistic is a candidate again, and wins on its record.
    store = service._get_store()
    store.save_payment_run(_target())
    confirmed = service.confirm_payment_run("EUR-NEW", actor="finance-lead")
    assert confirmed.drift_forecast_basis.startswith("forecast from")
    assert "per-currency (EUR)" in confirmed.drift_forecast_basis
    assert "excluded" not in confirmed.drift_forecast_basis


def test_a_regrade_for_a_segment_never_excluded_is_refused():
    service = _service_with_history()
    proposal = LagSegmentProposal(
        proposal_id="LSP-0099",
        segment="currency:EUR",
        kind="regrade_segment",
        proposed_by="Asha",
    )
    service._get_store().save_lag_segment_proposal(proposal)
    with pytest.raises(ValueError, match="does not stand excluded"):
        service.decide_lag_segment_proposal(
            "LSP-0099", actor="Boris", approve=True
        )


# -- the API -------------------------------------------------------------------


def test_lag_segment_proposal_endpoints():
    service = _service_with_history()
    client = TestClient(create_app(service))
    drafted = client.post(
        "/forecast/lag-segment-proposals/draft", json={"actor": "Asha"}
    )
    assert drafted.status_code == 200
    proposals = drafted.json()
    assert {p["segment"] for p in proposals} == {"currency:EUR", "global"}
    proposal_id = proposals[0]["proposal_id"]
    refused = client.post(
        f"/forecast/lag-segment-proposals/{proposal_id}/decide",
        json={"actor": "Asha", "approve": True},
    )
    assert refused.status_code == 422  # four-eyes
    decided = client.post(
        f"/forecast/lag-segment-proposals/{proposal_id}/decide",
        json={"actor": "Boris", "approve": True},
    )
    assert decided.status_code == 200
    assert decided.json()["status"] == "approved"
    listed = client.get("/forecast/lag-segment-proposals?status=approved")
    assert [p["proposal_id"] for p in listed.json()] == [proposal_id]
    missing = client.get("/forecast/lag-segment-proposals/LSP-9999")
    assert missing.status_code == 404
