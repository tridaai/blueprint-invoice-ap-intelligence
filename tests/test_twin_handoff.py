"""Cluster hand-off: the escalated twin cluster is decided as
one case, and the decision lands per member.

Escalation (round 14) put the cluster on the worklist with its
graph attached — but working it still meant opening every
member and deciding record by record, the exact chore the
escalation existed to end. The hand-off is one act:
``approve_all`` approves every waiting member THROUGH the
cluster (each via the standard approval path, guardrails and
holds included — a copy on a duplicate hold is refused by name,
so the bulk act never approves the same bill twice), or ``split`` dissolves the grouping with a named
reason when the members are not the same invoice after all.
Either way the act is stamped on every member's own record —
cluster, action, actor, reason, and that member's outcome —
and the cluster leaves the queue: decided, or dissolved.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore, _dump_record, _load_record
from ap_agent.twins import twin_clusters

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
CLUSTER_ID = "TWIN-INV-1001"
MEMBERS = {"INV-1001", "INV-1004", "INV-9001"}


@pytest.fixture()
def service() -> InvoiceService:
    return InvoiceService(store=InMemoryStore(), checkpointer=False)


def _cluster(service: InvoiceService) -> None:
    """The same Northbeam invoice arriving three ways."""
    service.analyze(SAMPLES["INV-1001"])
    service.analyze(SAMPLES["INV-1004"])
    third = SAMPLES["INV-1001"].model_copy(
        update={
            "invoice_id": "INV-9001",
            "supplier_name": "northbeam office supplies",
            "invoice_number": "nb 4471",
        }
    )
    service.analyze(third)
    assert service.queue_view().twin_escalation_count == 1


def _stamps(service: InvoiceService, invoice_id: str) -> list[dict]:
    return service._get_store().get(invoice_id).cluster_decisions


# -- approve_all -----------------------------------------------------------


def test_approve_all_decides_every_member_through_the_cluster(service):
    _cluster(service)
    report = service.decide_twin_cluster(
        CLUSTER_ID,
        actor="Priya Nair",
        action="approve_all",
        reason="All three copies verified against the PO — one payment, approved once",
    )
    assert report["cluster_id"] == CLUSTER_ID
    outcomes = {m["invoice_id"]: m["outcome"] for m in report["members"]}
    # One payment, approved once: the original is approved through the
    # cluster; the copies carry a duplicate hold, and the bulk act does
    # not acknowledge holds — it refuses them by name, exactly as it
    # refuses a guardrail-blocked member. Approving all three would put
    # the same bill in a payment run three times.
    assert outcomes["INV-1001"] == "approved"
    for copy_id in ("INV-1004", "INV-9001"):
        assert outcomes[copy_id].startswith("blocked:")
        assert "suspected duplicate of INV-1001" in outcomes[copy_id]
        assert service._get_store().get(copy_id).result.status == "awaiting_approver"
    original = service._get_store().get("INV-1001")
    assert original.result.status == "approved"
    assert original.result.decided_by == "Priya Nair"
    for member in MEMBERS:
        (stamp,) = _stamps(service, member)
        assert stamp["cluster_id"] == CLUSTER_ID
        assert stamp["action"] == "approve_all"
        assert stamp["actor"] == "Priya Nair"
        assert stamp["outcome"] == outcomes[member]
        assert "verified against the PO" in stamp["reason"]
    # The copies still wait for a decision (a rejection, or an
    # acknowledged approval), so the case stays on the worklist — and
    # only the original reaches a payment run.
    assert service.queue_view().twin_escalation_count == 1
    runs = service.propose_payment_runs(window_days=60, as_of="2026-10-01")
    assert {line.invoice_id for run in runs for line in run.lines} == {"INV-1001"}


def test_approve_all_keeps_a_decision_already_made(service):
    _cluster(service)
    service.approve("INV-1001", approver="Priya Nair", reason="the original")
    report = service.decide_twin_cluster(
        CLUSTER_ID,
        actor="Tomas Reid",
        action="approve_all",
        reason="Copies confirmed as resends of the approved original",
    )
    outcomes = {m["invoice_id"]: m["outcome"] for m in report["members"]}
    assert outcomes["INV-1001"] == "already_approved"
    # The copies are on a duplicate hold; the bulk act refuses them by name.
    assert outcomes["INV-1004"].startswith("blocked:")
    assert outcomes["INV-9001"].startswith("blocked:")
    # The earlier decision stands untouched; the cluster act is
    # recorded beside it, not over it.
    original = service._get_store().get("INV-1001")
    assert original.result.decided_by == "Priya Nair"
    assert _stamps(service, "INV-1001")[0]["outcome"] == "already_approved"


# -- split -----------------------------------------------------------------


def test_split_dissolves_the_case_and_stamps_every_member(service):
    _cluster(service)
    report = service.decide_twin_cluster(
        CLUSTER_ID,
        actor="Priya Nair",
        action="split",
        reason="Two are monthly statements for different periods — not one invoice",
    )
    outcomes = {m["invoice_id"]: m["outcome"] for m in report["members"]}
    assert outcomes == {member: "split" for member in MEMBERS}
    for member in MEMBERS:
        record = service._get_store().get(member)
        # A split decides nothing about any member's own case.
        assert record.result.status == "awaiting_approver"
        (stamp,) = _stamps(service, member)
        assert stamp["action"] == "split" and stamp["outcome"] == "split"
    # Off the queue as a case — and the graph read carries the
    # stamps, so the dissolution is auditable member by member.
    assert service.queue_view().twin_escalation_count == 0
    clusters = twin_clusters(service._get_store().list_records())
    node = next(
        m for c in clusters for m in c["members"] if m["invoice_id"] == "INV-1004"
    )
    assert node["cluster_decisions"][0]["action"] == "split"
    # A split cluster refuses further cluster acts.
    with pytest.raises(ValueError, match="already split"):
        service.decide_twin_cluster(
            CLUSTER_ID, actor="Priya Nair", action="approve_all", reason="again"
        )


# -- guards ------------------------------------------------------------------


def test_handoff_guards(service):
    _cluster(service)
    with pytest.raises(ValueError, match="carries its reason"):
        service.decide_twin_cluster(
            CLUSTER_ID, actor="Priya Nair", action="split", reason="  "
        )
    with pytest.raises(ValueError, match="carries its reason"):
        service.decide_twin_cluster(
            CLUSTER_ID, actor="Priya Nair", action="approve_all", reason=""
        )
    with pytest.raises(ValueError, match="Unknown cluster action"):
        service.decide_twin_cluster(
            CLUSTER_ID, actor="Priya Nair", action="merge", reason="why"
        )
    with pytest.raises(KeyError, match="Unknown twin cluster"):
        service.decide_twin_cluster(
            "TWIN-INV-9999", actor="Priya Nair", action="split", reason="why"
        )


def test_cluster_stamps_survive_the_store_round_trip(service):
    _cluster(service)
    service.decide_twin_cluster(
        CLUSTER_ID, actor="Priya Nair", action="split", reason="Not one invoice"
    )
    record = service._get_store().get("INV-1001")
    loaded = _load_record(_dump_record(record))
    assert loaded.cluster_decisions == record.cluster_decisions


# -- the API -----------------------------------------------------------------


def test_cluster_decide_endpoint(service):
    _cluster(service)
    client = TestClient(create_app(service))
    response = client.post(
        f"/queue/twin-clusters/{CLUSTER_ID}/decide",
        json={
            "actor": "Priya Nair",
            "action": "split",
            "reason": "Different periods, not one invoice",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "split"
    assert {m["invoice_id"] for m in body["members"]} == MEMBERS
    missing = client.post(
        "/queue/twin-clusters/TWIN-INV-9999/decide",
        json={"actor": "Priya Nair", "action": "split", "reason": "why"},
    )
    assert missing.status_code == 404


def test_cluster_decide_endpoint_refuses_a_reasonless_split(service):
    _cluster(service)
    client = TestClient(create_app(service))
    response = client.post(
        f"/queue/twin-clusters/{CLUSTER_ID}/decide",
        json={"actor": "Priya Nair", "action": "split", "reason": ""},
    )
    assert response.status_code == 422
