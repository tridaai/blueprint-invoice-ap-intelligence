"""Supplier-contact proposals: the unblocking pattern becomes
a governed onboarding decision.

The re-analysis analytics name the suppliers whose replies
never move their cases — the record is re-read, the evidence
weighed, the outcome lands where it was, because the settling
document was never collected. The proposals are the governed
path from that pattern to a checklist change, under the same
discipline as the lag-segment proposals: drafted from the
evidence by one operator, decided by a second, applied only
on approval. An approved collect puts the supplier on the
documents-up-front list — derived from this ledger, like the
lag exclusions — and the queue's information waits for a
listed supplier say so. A listed supplier whose re-analyses
since listing all moved their cases drafts its release; one
that lands unchanged again keeps the release refused.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.schemas import (
    AgentResult,
    InformationRequest,
    ReanalysisEntry,
    SupplierContactProposal,
    Triage,
)
from ap_agent.service import InvoiceService
from ap_agent.store import ApprovalRecord, InMemoryStore
from ap_agent.supplier_contact_proposals import (
    documents_upfront_suppliers,
    listing_evidence,
    next_supplier_proposal_id,
    proposals_from_analytics,
)

STUCK = "Stuckfast Ltd"
MOVER = "Mover GmbH"


def _entry(previous: str, new: str) -> ReanalysisEntry:
    return ReanalysisEntry(previous_outcome=previous, new_outcome=new)


def _record(
    invoice_id: str, supplier: str, chain: list[ReanalysisEntry], **kw
) -> ApprovalRecord:
    result = AgentResult(
        invoice_id=invoice_id,
        supplier_name=supplier,
        triage=Triage(outcome="missing_po"),
        reanalysis_history=chain,
        **kw,
    )
    return ApprovalRecord(result=result, invoice={"invoice_id": invoice_id})


def _stuck_records() -> list[ApprovalRecord]:
    """Stuckfast: three re-analyses, two landing unchanged.
    Mover: two re-analyses, both moving the case."""
    return [
        _record(
            "INV-S1",
            STUCK,
            [
                _entry("missing_po", "missing_po"),
                _entry("missing_po", "missing_po"),
                _entry("missing_po", "clean_match"),
            ],
        ),
        _record(
            "INV-M1",
            MOVER,
            [
                _entry("missing_po", "clean_match"),
                _entry("price_variance", "clean_match"),
            ],
        ),
    ]


def _service(records=()) -> InvoiceService:
    store = InMemoryStore()
    for record in records:
        store.save(record)
    return InvoiceService(store=store, checkpointer=False)


# -- the pure drafting half ---------------------------------------------------


def test_next_ids_are_sequential():
    assert next_supplier_proposal_id([]) == "SCP-0001"
    assert next_supplier_proposal_id(["SCP-0001", "SCP-0007"]) == "SCP-0008"


def test_the_list_is_the_latest_approved_decision_per_supplier():
    def proposal(pid, supplier, kind, status):
        return SupplierContactProposal(
            proposal_id=pid, supplier_name=supplier, kind=kind, status=status
        )

    proposals = [
        proposal("SCP-0001", STUCK, "collect_documents_upfront", "approved"),
        proposal("SCP-0002", MOVER, "collect_documents_upfront", "rejected"),
        proposal("SCP-0003", "Third Co", "collect_documents_upfront", "pending"),
    ]
    assert documents_upfront_suppliers(proposals) == {STUCK}
    proposals.append(
        proposal("SCP-0004", STUCK, "release_documents_upfront", "approved")
    )
    assert documents_upfront_suppliers(proposals) == set()


def test_analytics_draft_a_collect_only_for_the_pattern():
    from ap_agent.clarify import reanalysis_analytics

    specs = proposals_from_analytics(
        reanalysis_analytics(_stuck_records()), []
    )
    assert len(specs) == 1
    (spec,) = specs
    assert spec["supplier_name"] == STUCK
    assert spec["kind"] == "collect_documents_upfront"
    assert spec["evidence"]["reanalyses"] == 3
    assert spec["evidence"]["unchanged"] == 2
    assert STUCK in spec["basis"]


def test_one_unchanged_round_is_a_week_not_a_pattern():
    from ap_agent.clarify import reanalysis_analytics

    records = [
        _record(
            "INV-X",
            "Once Co",
            [_entry("missing_po", "missing_po"), _entry("missing_po", "clean_match")],
        )
    ]
    assert proposals_from_analytics(reanalysis_analytics(records), []) == []


def test_a_listed_supplier_drafts_its_release_only_when_all_moved():
    approved_collect = SupplierContactProposal(
        proposal_id="SCP-0001",
        supplier_name=STUCK,
        kind="collect_documents_upfront",
        status="approved",
        evidence={"reanalyses": 3, "unchanged": 2},
    )
    assert listing_evidence([approved_collect])[STUCK]["unchanged"] == 2
    # Two more re-analyses since listing, both moving the case.
    analytics = {
        "suppliers_not_unblocked": [
            {"supplier_name": STUCK, "reanalyses": 5, "unchanged": 2}
        ]
    }
    (spec,) = proposals_from_analytics(analytics, [approved_collect])
    assert spec["kind"] == "release_documents_upfront"
    assert spec["evidence"]["reanalyses_at_listing"] == 3
    # One landing unchanged since listing: no release question.
    analytics["suppliers_not_unblocked"][0]["unchanged"] = 3
    assert proposals_from_analytics(analytics, [approved_collect]) == []
    # Quiet since listing (no new re-analyses): also no question.
    analytics["suppliers_not_unblocked"][0] = {
        "supplier_name": STUCK,
        "reanalyses": 3,
        "unchanged": 2,
    }
    assert proposals_from_analytics(analytics, [approved_collect]) == []


# -- the governed workflow ------------------------------------------------------


def test_draft_decide_and_the_list_moves():
    svc = _service(_stuck_records())
    drafted = svc.draft_supplier_contact_proposals(actor="Ana Ruiz")
    assert [p.supplier_name for p in drafted] == [STUCK]
    proposal = drafted[0]
    assert proposal.proposal_id == "SCP-0001"
    assert proposal.status == "pending"
    # Drafting applies nothing, and the sweep does not re-draft
    # an open question.
    assert svc.documents_upfront_suppliers() == []
    assert svc.draft_supplier_contact_proposals(actor="Ana Ruiz") == []
    # The drafter cannot decide their own proposal.
    with pytest.raises(ValueError, match="four-eyes"):
        svc.decide_supplier_contact_proposal(
            proposal.proposal_id, actor="Ana Ruiz", approve=True
        )
    decided = svc.decide_supplier_contact_proposal(
        proposal.proposal_id,
        actor="Tomas Reid",
        approve=True,
        note="collect the certs at onboarding",
    )
    assert decided.status == "approved"
    assert decided.applied_at
    assert svc.documents_upfront_suppliers() == [STUCK]
    section = svc.metrics_snapshot()["supplier_contact_proposals"]
    assert section["approved"] == 1
    assert section["documents_upfront_suppliers"] == [STUCK]


def test_rejection_leaves_the_list_untouched_and_redrafting_reasks():
    svc = _service(_stuck_records())
    (proposal,) = svc.draft_supplier_contact_proposals(actor="Ana Ruiz")
    svc.decide_supplier_contact_proposal(
        proposal.proposal_id, actor="Tomas Reid", approve=False, note="not yet"
    )
    assert svc.documents_upfront_suppliers() == []
    # The question was answered, not left open: the pattern
    # still stands, so a fresh sweep asks it again.
    (again,) = svc.draft_supplier_contact_proposals(actor="Ana Ruiz")
    assert again.proposal_id == "SCP-0002"
    assert again.status == "pending"


def test_the_release_loop_and_its_stale_evidence_guard():
    svc = _service(_stuck_records())
    (collect,) = svc.draft_supplier_contact_proposals(actor="Ana Ruiz")
    svc.decide_supplier_contact_proposal(
        collect.proposal_id, actor="Tomas Reid", approve=True
    )
    assert svc.documents_upfront_suppliers() == [STUCK]
    # Two more re-analyses land — both move their cases.
    store = svc._get_store()
    store.save(
        _record(
            "INV-S2",
            STUCK,
            [
                _entry("missing_po", "clean_match"),
                _entry("price_variance", "clean_match"),
            ],
        )
    )
    (release,) = svc.draft_supplier_contact_proposals(actor="Ana Ruiz")
    assert release.kind == "release_documents_upfront"
    # Before the decision lands, another re-analysis lands
    # unchanged: the release's evidence is stale, and approval
    # is refused — the supplier stays listed.
    store.save(
        _record("INV-S3", STUCK, [_entry("missing_po", "missing_po")])
    )
    with pytest.raises(ValueError, match="since this release was drafted"):
        svc.decide_supplier_contact_proposal(
            release.proposal_id, actor="Tomas Reid", approve=True
        )
    assert svc.documents_upfront_suppliers() == [STUCK]
    svc.decide_supplier_contact_proposal(
        release.proposal_id, actor="Tomas Reid", approve=False
    )
    assert svc.documents_upfront_suppliers() == [STUCK]


def test_a_clean_release_takes_the_supplier_off_the_list():
    svc = _service(_stuck_records())
    (collect,) = svc.draft_supplier_contact_proposals(actor="Ana Ruiz")
    svc.decide_supplier_contact_proposal(
        collect.proposal_id, actor="Tomas Reid", approve=True
    )
    svc._get_store().save(
        _record(
            "INV-S2",
            STUCK,
            [_entry("missing_po", "clean_match")],
        )
    )
    (release,) = svc.draft_supplier_contact_proposals(actor="Ana Ruiz")
    svc.decide_supplier_contact_proposal(
        release.proposal_id, actor="Tomas Reid", approve=True
    )
    assert svc.documents_upfront_suppliers() == []


def test_an_unattributed_draft_is_refused():
    svc = _service(_stuck_records())
    with pytest.raises(ValueError, match="actor"):
        svc.draft_supplier_contact_proposals(actor=" ")


# -- the consumer: information waits name the pattern ----------------------------


def test_a_listed_suppliers_information_wait_reads_documents_upfront():
    waiting = _record(
        "INV-S9",
        STUCK,
        [_entry("missing_po", "missing_po"), _entry("missing_po", "missing_po")],
        status="awaiting_approver",
        needs_information=True,
        information_request=InformationRequest(
            requested_at="2026-10-01T09:00:00+00:00",
            missing_items=["supplier tax certificate"],
            reply_status="awaiting_reply",
        ),
    )
    svc = _service([*_stuck_records(), waiting])
    queue = svc.queue_view(as_of="2026-10-20")
    (item,) = [
        w for w in queue.information_waits if w.invoice_id == "INV-S9"
    ]
    assert item.documents_upfront is False  # not listed yet
    (collect,) = svc.draft_supplier_contact_proposals(actor="Ana Ruiz")
    svc.decide_supplier_contact_proposal(
        collect.proposal_id, actor="Tomas Reid", approve=True
    )
    queue = svc.queue_view(as_of="2026-10-20")
    (item,) = [
        w for w in queue.information_waits if w.invoice_id == "INV-S9"
    ]
    assert item.documents_upfront is True


# -- the API -----------------------------------------------------------------------


def test_contact_proposal_endpoints():
    svc = _service(_stuck_records())
    client = TestClient(create_app(svc))
    drafted = client.post(
        "/suppliers/contact-proposals/draft", json={"actor": "Ana Ruiz"}
    )
    assert drafted.status_code == 200
    (proposal,) = drafted.json()
    assert proposal["proposal_id"] == "SCP-0001"
    refused = client.post(
        f"/suppliers/contact-proposals/{proposal['proposal_id']}/decide",
        json={"actor": "Ana Ruiz", "approve": True},
    )
    assert refused.status_code == 422  # four-eyes
    decided = client.post(
        f"/suppliers/contact-proposals/{proposal['proposal_id']}/decide",
        json={"actor": "Tomas Reid", "approve": True, "note": "yes"},
    )
    assert decided.status_code == 200
    assert decided.json()["status"] == "approved"
    listing = client.get("/suppliers/contact-proposals?status=approved")
    assert [p["proposal_id"] for p in listing.json()] == ["SCP-0001"]
    missing = client.get("/suppliers/contact-proposals/SCP-9999")
    assert missing.status_code == 404
