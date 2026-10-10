"""Documents-up-front effectiveness: the listings, graded.

A supplier on the documents-up-front list is an intervention
— their onboarding checklist was changed because their cases
kept blocking on information. Round 19 made the listing
governed and reversible on the re-analysis pattern. This
grades the intervention itself: information waits and
clarification loops per case, before vs after the listing
moment. A listing that moved both rates is working; one
that moved neither drafts its own release through the same
proposals discipline, on the grade's evidence — and an
approval of that release is re-graded at decision time, so a
listing that started working since the draft is not released
on a stale grade.
"""

from __future__ import annotations

import pytest

from ap_agent.schemas import (
    AgentResult,
    InformationRequest,
    ReanalysisEntry,
    SupplierContactProposal,
)
from ap_agent.service import InvoiceService
from ap_agent.store import ApprovalRecord, InMemoryStore
from ap_agent.supplier_contact_proposals import (
    documents_upfront_effectiveness,
    listing_moments,
)

LISTED_AT = "2026-10-01T09:00:00+00:00"
FLAT = "Flatline Packaging"
SETTLING = "Settling Supplies"
QUIET = "Quietline GmbH"
PATTERNLESS = "Patternless Ltd"


def _entry(at: str, moved: bool) -> ReanalysisEntry:
    return ReanalysisEntry(
        previous_outcome="needs_information",
        new_outcome="approved" if moved else "needs_information",
        supplied_items=["goods_receipt"] if moved else [],
        at=at,
    )


def _record(
    invoice_id: str,
    supplier: str,
    submitted: str,
    *,
    waited: bool = False,
    requested_at: str = "",
    entries: list[ReanalysisEntry] | None = None,
) -> ApprovalRecord:
    result = AgentResult(
        invoice_id=invoice_id,
        status="needs_information" if waited else "approved",
        supplier_name=supplier,
        needs_information=waited,
        information_request=(
            InformationRequest(
                missing=["goods receipt"],
                questions=["Please send the goods receipt."],
                requested_at=requested_at,
            )
            if waited
            else None
        ),
        reanalysis_history=entries or [],
    )
    invoice = {
        "invoice_id": invoice_id,
        "supplier_name": supplier,
        "submitted_at": submitted,
        "invoice_date": submitted,
        "total_amount": 100.0,
        "currency": "USD",
    }
    return ApprovalRecord(result=result, invoice=invoice)


def _collect(supplier: str, n: int) -> SupplierContactProposal:
    return SupplierContactProposal(
        proposal_id=f"SCP-{n:04d}",
        supplier_name=supplier,
        kind="collect_documents_upfront",
        status="approved",
        basis="drafted from the unblocking pattern",
        evidence={"reanalyses": 3, "unchanged": 3},
        proposed_by="AP Lead",
        proposed_at="2026-09-25T09:00:00+00:00",
        decided_by="AP Manager",
        decided_at="2026-09-26T09:00:00+00:00",
        decision_note="collect the settling documents first",
        applied_at=LISTED_AT,
    )


def _flatline_records() -> list[ApprovalRecord]:
    """Before the listing: 2 cases, both waited, 2 loops.
    Since it: 2 cases, both waited, 2 loops. Nothing moved."""
    return [
        _record("INV-F1", FLAT, "2026-09-10", waited=True,
                requested_at="2026-09-11T09:00:00+00:00",
                entries=[_entry("2026-09-12T09:00:00+00:00", moved=False)]),
        _record("INV-F2", FLAT, "2026-09-15", waited=True,
                requested_at="2026-09-16T09:00:00+00:00",
                entries=[_entry("2026-09-17T09:00:00+00:00", moved=False)]),
        _record("INV-F3", FLAT, "2026-10-05", waited=True,
                requested_at="2026-10-06T09:00:00+00:00",
                entries=[_entry("2026-10-07T09:00:00+00:00", moved=False)]),
        _record("INV-F4", FLAT, "2026-10-08", waited=True,
                requested_at="2026-10-09T09:00:00+00:00",
                entries=[_entry("2026-10-09T15:00:00+00:00", moved=False)]),
    ]


def _settling_records() -> list[ApprovalRecord]:
    """Before: 2 cases, both waited, 2 loops. Since: 2 cases,
    no waits, 1 loop. The checklist worked."""
    return [
        _record("INV-S1", SETTLING, "2026-09-10", waited=True,
                requested_at="2026-09-11T09:00:00+00:00",
                entries=[_entry("2026-09-12T09:00:00+00:00", moved=False)]),
        _record("INV-S2", SETTLING, "2026-09-15", waited=True,
                requested_at="2026-09-16T09:00:00+00:00",
                entries=[_entry("2026-09-17T09:00:00+00:00", moved=True)]),
        _record("INV-S3", SETTLING, "2026-10-05",
                entries=[_entry("2026-10-06T09:00:00+00:00", moved=True)]),
        _record("INV-S4", SETTLING, "2026-10-08"),
    ]


def _service(records, proposals) -> InvoiceService:
    store = InMemoryStore()
    for record in records:
        store.save(record)
    for proposal in proposals:
        store.save_supplier_contact_proposal(proposal)
    return InvoiceService(store=store, checkpointer=False)


# -- the grading ------------------------------------------------------------------------


def test_the_grades_split_working_no_effect_and_unmeasured():
    records = (
        _flatline_records()
        + _settling_records()
        + [_record("INV-Q1", QUIET, "2026-09-10", waited=True,
                   requested_at="2026-09-11T09:00:00+00:00")]
        + [
            _record("INV-P1", PATTERNLESS, "2026-09-10"),
            _record("INV-P2", PATTERNLESS, "2026-10-05", waited=True,
                    requested_at="2026-10-06T09:00:00+00:00"),
        ]
    )
    proposals = [
        _collect(FLAT, 1),
        _collect(SETTLING, 2),
        _collect(QUIET, 3),
        _collect(PATTERNLESS, 4),
    ]
    graded = documents_upfront_effectiveness(records, proposals)
    assert graded["listed"] == 4
    assert graded["working"] == 1
    assert graded["no_effect"] == 1
    assert graded["unmeasured"] == 2
    rows = {row["supplier_name"]: row for row in graded["suppliers"]}

    flat = rows[FLAT]
    assert flat["listed_at"] == LISTED_AT
    assert flat["cases_before"] == 2 and flat["cases_after"] == 2
    assert flat["waits_per_case_before"] == 1.0
    assert flat["waits_per_case_after"] == 1.0
    assert flat["loops_per_case_after"] == 1.0
    assert flat["verdict"] == "no_effect"

    settling = rows[SETTLING]
    assert settling["waits_per_case_after"] == 0.0
    assert settling["loops_per_case_after"] == 0.5
    assert settling["verdict"] == "working"

    # No cases since listing: unmeasured, not a success.
    assert rows[QUIET]["verdict"] == "unmeasured"
    # No pre-listing pattern to compare against: unmeasured,
    # even though the one case since waited.
    assert rows[PATTERNLESS]["verdict"] == "unmeasured"


def test_listing_moments_follow_the_ledger_not_the_drafts():
    proposals = [_collect(FLAT, 1)]
    assert listing_moments(proposals) == {FLAT: LISTED_AT}
    release = SupplierContactProposal(
        proposal_id="SCP-0009-release",
        supplier_name=FLAT,
        kind="release_documents_upfront",
        status="approved",
        basis="the pattern settled",
        evidence={},
        proposed_by="AP Lead",
        proposed_at="2026-11-01T09:00:00+00:00",
        decided_by="AP Manager",
        decided_at="2026-11-02T09:00:00+00:00",
        applied_at="2026-11-02T09:30:00+00:00",
    )
    assert listing_moments(proposals + [release]) == {}


def test_the_grade_rides_metrics_beside_the_proposals():
    svc = _service(_flatline_records(), [_collect(FLAT, 1)])
    metrics = svc.metrics_snapshot()
    graded = metrics["documents_upfront_effectiveness"]
    assert graded["listed"] == 1
    assert graded["no_effect"] == 1
    assert metrics["supplier_contact_proposals"][
        "documents_upfront_suppliers"
    ] == [FLAT]


# -- the loop, closed by evidence --------------------------------------------------------


def test_a_no_effect_listing_drafts_its_own_release_on_the_grade():
    svc = _service(_flatline_records(), [_collect(FLAT, 1)])
    drafted = svc.draft_supplier_contact_proposals(actor="AP Lead")
    assert len(drafted) == 1
    proposal = drafted[0]
    assert proposal.kind == "release_documents_upfront"
    assert proposal.supplier_name == FLAT
    assert proposal.evidence["drafted_from"] == "effectiveness_grade"
    assert proposal.evidence["verdict"] == "no_effect"
    assert proposal.evidence["waits_per_case_before"] == 1.0
    assert "moved nothing" in proposal.basis


def test_a_working_listing_drafts_no_release():
    svc = _service(_settling_records(), [_collect(SETTLING, 2)])
    assert svc.draft_supplier_contact_proposals(actor="AP Lead") == []


def test_approving_the_grade_release_unlists_the_supplier():
    svc = _service(_flatline_records(), [_collect(FLAT, 1)])
    proposal = svc.draft_supplier_contact_proposals(actor="AP Lead")[0]
    decided = svc.decide_supplier_contact_proposal(
        proposal.proposal_id, actor="AP Manager", approve=True
    )
    assert decided.status == "approved"
    assert decided.applied_at is not None
    assert any(
        "graded and changed nothing" in event.note
        for event in decided.history
    )
    assert svc.documents_upfront_suppliers() == []


def test_the_release_is_refused_when_the_grade_has_moved_since_the_draft():
    svc = _service(_flatline_records(), [_collect(FLAT, 1)])
    proposal = svc.draft_supplier_contact_proposals(actor="AP Lead")[0]
    # Two clean cases land after the draft: waits per case
    # falls to 0.5 and loops to 0.5 — the listing now reads
    # as working, and the stale grade cannot release it.
    store = svc._get_store()
    store.save(_record("INV-F5", FLAT, "2026-10-12"))
    store.save(_record("INV-F6", FLAT, "2026-10-13"))
    with pytest.raises(ValueError, match="working"):
        svc.decide_supplier_contact_proposal(
            proposal.proposal_id, actor="AP Manager", approve=True
        )
    # The refusal leaves the listing standing; rejecting the
    # proposal is the decider's way to close the question.
    assert FLAT in svc.documents_upfront_suppliers()
    rejected = svc.decide_supplier_contact_proposal(
        proposal.proposal_id, actor="AP Manager", approve=False
    )
    assert rejected.status == "rejected"
    assert FLAT in svc.documents_upfront_suppliers()
