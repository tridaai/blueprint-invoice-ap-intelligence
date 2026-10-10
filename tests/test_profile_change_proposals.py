"""Effectiveness-driven profile proposals: the grades have named
configuration theatre since round 13 — a profile governing no
exports, or routing no safer than ungoverned — but a read changed
nothing, and carrying the finding into a configuration change was
hand work. The drafting sweep turns those two grades into
governed proposals (retire the dormant profile; turn on the
marker flag whose absence the rate indicts), decided under the
proposals discipline: drafter ≠ decider, approval applies,
rejection leaves the profile exactly as it was, and a proposal
whose evidence moved since the draft is refused rather than
applied on stale grades. The other two grade families draft
nothing on purpose — a flagged supplier still marking is an
onboarding failure for the master-change path, and a breached
lead promise is a resolution-path regression no profile edit
un-breaches.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.profile_proposals import next_proposal_id, proposals_from_grades
from ap_agent.schemas import (
    HandoffCreditorEntry,
    HandoffProfile,
    PaymentRun,
    PaymentRunLine,
)
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore


@pytest.fixture()
def service() -> InvoiceService:
    return InvoiceService(store=InMemoryStore(), checkpointer=False)


def _line(invoice_id: str, supplier: str, net: float) -> PaymentRunLine:
    return PaymentRunLine(
        invoice_id=invoice_id,
        supplier_name=supplier,
        invoice_number=f"{invoice_id}-N",
        gross_amount=net,
        net_amount=net,
    )


def _entry(invoice_id: str, supplier: str, source: str) -> HandoffCreditorEntry:
    return HandoffCreditorEntry(
        invoice_id=invoice_id, supplier_name=supplier, source=source
    )


def _exported(run_id: str, **kw) -> PaymentRun:
    base = dict(
        run_id=run_id,
        currency="EUR",
        status="confirmed",
        proposed_at="2026-10-05T09:00:00+00:00",
        handoff_generated_at="2026-10-11T10:00:00+00:00",
        handoff_format="pain001",
    )
    base.update(kw)
    return PaymentRun(**base)


def _save_run(service: InvoiceService, run: PaymentRun) -> None:
    service._get_store().save_payment_run(run)


def _dormant_gbp(service: InvoiceService) -> None:
    service.set_handoff_profile(
        actor="Asha", scope="currency", currency="GBP", format="csv"
    )


def _unsafe_eur(service: InvoiceService) -> None:
    """An EUR profile (marker flag off) governing one fully
    marked export, beside an ungoverned export just as marked —
    the equal-rate grade."""
    service.set_handoff_profile(
        actor="Asha", scope="currency", currency="EUR", format="pain001"
    )
    _save_run(
        service,
        _exported(
            "RUN-G",
            handoff_profile_id="HP-CUR-EUR",
            lines=[_line("INV-1", "Alpen", 100.0)],
            handoff_creditor_entries=[_entry("INV-1", "Alpen", "at_import")],
        ),
    )
    _save_run(
        service,
        _exported(
            "RUN-U",
            lines=[_line("INV-2", "Cima", 50.0)],
            handoff_creditor_entries=[_entry("INV-2", "Cima", "at_import")],
        ),
    )


# --- The pure drafting core ----------------------------------------


def test_drafting_rules_directly():
    dormant = HandoffProfile(
        profile_id="HP-CUR-GBP", scope="currency", currency="GBP", format="csv"
    )
    specs = proposals_from_grades(
        [dormant],
        [{"profile_id": "HP-CUR-GBP", "exports_governed": 0, "notes": []}],
    )
    (spec,) = specs
    assert spec["kind"] == "retire"
    assert spec["proposed"] is None
    assert "governs no exports" in spec["basis"]


def test_flagged_and_lead_grades_draft_nothing():
    profile = HandoffProfile(
        profile_id="HP-CUR-EUR", scope="currency", currency="EUR",
        format="pain001", lead_days=3, markers_rejected=True,
    )
    grades = [
        {
            "profile_id": "HP-CUR-EUR",
            "exports_governed": 2,
            "notes": [
                "Berg was flagged at proposal time (its bank rejects "
                "at-import markers under this profile) and still "
                "exported with the marker on 1 export(s)",
                "the lead promise was breached on 1 governed "
                "export(s): an execution date was written inside "
                "the promised lead",
            ],
        }
    ]
    assert proposals_from_grades([profile], grades) == []


def test_next_proposal_id_is_sequential():
    assert next_proposal_id([]) == "PCP-0001"
    assert next_proposal_id(["PCP-0001", "PCP-0007"]) == "PCP-0008"


# --- Drafting -------------------------------------------------------


def test_a_dormant_profile_is_drafted_for_retirement(service):
    _dormant_gbp(service)
    (proposal,) = service.draft_profile_change_proposals(actor="Asha")
    assert proposal.proposal_id == "PCP-0001"
    assert proposal.profile_id == "HP-CUR-GBP"
    assert proposal.kind == "retire"
    assert proposal.proposed is None
    assert proposal.current["currency"] == "GBP"
    assert proposal.status == "pending"
    assert proposal.proposed_by == "Asha"
    # Drafting applies nothing — the profile stands until decided.
    assert service.get_handoff_profile("HP-CUR-GBP") is not None


def test_an_unsafe_profile_is_drafted_for_tightening(service):
    _unsafe_eur(service)
    (proposal,) = service.draft_profile_change_proposals(actor="Asha")
    assert proposal.kind == "tighten_markers"
    assert proposal.current["markers_rejected"] is False
    assert proposal.proposed["markers_rejected"] is True
    assert "same rate as ungoverned" in proposal.basis
    profile = service.get_handoff_profile("HP-CUR-EUR")
    assert profile.markers_rejected is False  # nothing applied by drafting


def test_drafting_needs_an_actor_and_skips_open_questions(service):
    _dormant_gbp(service)
    with pytest.raises(ValueError, match="actor"):
        service.draft_profile_change_proposals(actor="")
    service.draft_profile_change_proposals(actor="Asha")
    # The question is open — re-running the sweep drafts nothing.
    assert service.draft_profile_change_proposals(actor="Asha") == []
    # …but once it is decided, a fresh grade can ask it again.
    service.decide_profile_change_proposal(
        "PCP-0001", actor="Priya", approve=False, note="keep for now"
    )
    (again,) = service.draft_profile_change_proposals(actor="Asha")
    assert again.proposal_id == "PCP-0002"


def test_a_safe_profile_drafts_nothing(service):
    _unsafe_eur(service)
    service.set_handoff_profile(
        actor="Asha", scope="currency", currency="EUR", format="pain001",
        markers_rejected=True,
    )
    assert service.draft_profile_change_proposals(actor="Asha") == []


# --- Decisions --------------------------------------------------------


def test_approval_retires_the_profile(service):
    _dormant_gbp(service)
    service.draft_profile_change_proposals(actor="Asha")
    decided = service.decide_profile_change_proposal(
        "PCP-0001", actor="Priya", approve=True, note="agreed — theatre"
    )
    assert decided.status == "approved"
    assert decided.decided_by == "Priya"
    assert decided.applied_at
    assert service.get_handoff_profile("HP-CUR-GBP") is None
    metrics = service.metrics_snapshot()
    assert metrics["handoff_profile_proposals"]["approved"] == 1


def test_approval_tightens_through_the_governed_path(service):
    _unsafe_eur(service)
    service.draft_profile_change_proposals(actor="Asha")
    service.decide_profile_change_proposal(
        "PCP-0001", actor="Priya", approve=True
    )
    profile = service.get_handoff_profile("HP-CUR-EUR")
    assert profile.markers_rejected is True
    assert profile.updated_by == "Priya"
    # The profile's own audit trail carries the application.
    assert any("PCP-0001" in (e.detail or "") or "markers rejected by the payee's bank" in (e.detail or "") for e in profile.history)


def test_rejection_leaves_the_profile_exactly_as_it_was(service):
    _dormant_gbp(service)
    service.draft_profile_change_proposals(actor="Asha")
    decided = service.decide_profile_change_proposal(
        "PCP-0001", actor="Priya", approve=False, note="GBP onboarding is coming"
    )
    assert decided.status == "rejected"
    assert decided.applied_at is None
    profile = service.get_handoff_profile("HP-CUR-GBP")
    assert profile is not None
    assert profile.format == "csv"


def test_the_drafter_cannot_decide_their_own_proposal(service):
    _dormant_gbp(service)
    service.draft_profile_change_proposals(actor="Asha")
    with pytest.raises(ValueError, match="four-eyes"):
        service.decide_profile_change_proposal(
            "PCP-0001", actor="Asha", approve=True
        )
    assert service.get_profile_change_proposal("PCP-0001").status == "pending"


def test_a_decided_proposal_cannot_be_decided_again(service):
    _dormant_gbp(service)
    service.draft_profile_change_proposals(actor="Asha")
    service.decide_profile_change_proposal(
        "PCP-0001", actor="Priya", approve=False
    )
    with pytest.raises(ValueError, match="already rejected"):
        service.decide_profile_change_proposal(
            "PCP-0001", actor="Marcus", approve=True
        )


def test_a_retirement_whose_profile_started_governing_is_refused(service):
    _dormant_gbp(service)
    service.draft_profile_change_proposals(actor="Asha")
    # Since the draft, a GBP export routed through the profile.
    _save_run(
        service,
        _exported(
            "RUN-GBP",
            currency="GBP",
            handoff_profile_id="HP-CUR-GBP",
            lines=[_line("INV-9", "Alpen", 10.0)],
            handoff_creditor_entries=[
                _entry("INV-9", "Alpen", "supplier_master")
            ],
        ),
    )
    with pytest.raises(ValueError, match="started governing"):
        service.decide_profile_change_proposal(
            "PCP-0001", actor="Priya", approve=True
        )
    assert service.get_handoff_profile("HP-CUR-GBP") is not None
    assert service.get_profile_change_proposal("PCP-0001").status == "pending"


def test_a_tightening_already_in_force_is_refused_as_moot(service):
    _unsafe_eur(service)
    service.draft_profile_change_proposals(actor="Asha")
    # An operator set the flag directly in the meantime.
    service.set_handoff_profile(
        actor="Marcus", scope="currency", currency="EUR",
        format="pain001", markers_rejected=True,
    )
    with pytest.raises(ValueError, match="already"):
        service.decide_profile_change_proposal(
            "PCP-0001", actor="Priya", approve=True
        )


def test_proposal_endpoints(service):
    _dormant_gbp(service)
    client = TestClient(create_app(service))
    drafted = client.post(
        "/handoff/profile-change-proposals/draft", json={"actor": "Asha"}
    )
    assert drafted.status_code == 200
    (proposal,) = drafted.json()
    assert proposal["proposal_id"] == "PCP-0001"
    assert proposal["kind"] == "retire"
    same_hands = client.post(
        "/handoff/profile-change-proposals/PCP-0001/decide",
        json={"actor": "Asha", "approve": True},
    )
    assert same_hands.status_code == 422
    decided = client.post(
        "/handoff/profile-change-proposals/PCP-0001/decide",
        json={"actor": "Priya", "approve": True, "note": "agreed"},
    )
    assert decided.status_code == 200
    assert decided.json()["status"] == "approved"
    listing = client.get(
        "/handoff/profile-change-proposals", params={"status": "approved"}
    )
    assert [p["proposal_id"] for p in listing.json()] == ["PCP-0001"]
    missing = client.get("/handoff/profile-change-proposals/PCP-9999")
    assert missing.status_code == 404
