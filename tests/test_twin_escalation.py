"""Twin-cluster escalation: clusters become queue cases.

The twin graph (round 13) read the duplicate clusters back for
the audit export — but the approver works the queue, and the
queue never showed a cluster as a case. A twin could arrive
while its sibling waited, or one copy could be approved while
another sat at the gate, and the only surface that knew was an
audit view nobody opens mid-shift.

Now a cluster escalates onto the worklist with its graph
attached when ANY trigger fires: size (TWIN_CLUSTER_MIN_SIZE,
default 3), combined value in one currency
(TWIN_CLUSTER_VALUE_THRESHOLD, default 10000), or an
approved+awaiting span — the double-payment shape itself, at
any size or value. Under every trigger, a cluster stays
audit-only: the queue is for work.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.queue import build_queue, twin_escalation_items
from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}

_SEGMENT_TEXT = (
    "Northbeam Office Supplies\n"
    "Combined billing for purchase order PO-9015\n\n"
    "Invoice NB-4471\n"
    "Date: 2026-10-01\n"
    "Due: 2026-10-31\n"
    "PO: PO-9015\n"
    "Printer paper pallet x10 @ 100.00 = 1000.00\n"
    "Subtotal 1000.00\n"
    "Tax 180.00\n"
    "Total 1180.00 USD\n"
)


def _bundle(bundle_id: str, text: str) -> dict:
    return {
        "bundle_id": bundle_id,
        "supplier_name": "Northbeam Office Supplies",
        "currency": "USD",
        "submitted_at": "2026-10-03T09:20:00",
        "documents": [
            {
                "doc_type": "invoice",
                "document_id": f"DOC-{bundle_id}",
                "ocr_text": text,
                "ocr_confidence": 0.98,
                "fields": {},
            }
        ],
        "purchase_orders": [],
        "goods_receipts": [],
    }


@pytest.fixture()
def service() -> InvoiceService:
    return InvoiceService(store=InMemoryStore(), checkpointer=False)


def _pair(service: InvoiceService) -> None:
    """INV-1001 and its resend INV-1004: a pair, USD 1180 each."""
    service.analyze(SAMPLES["INV-1001"])
    service.analyze(SAMPLES["INV-1004"])


def _big_pair(service: InvoiceService) -> None:
    """A pair of USD 6000 copies under a different number."""
    first = SAMPLES["INV-1001"].model_copy(
        update={
            "invoice_id": "INV-9101",
            "invoice_number": "NB-9001",
            "total_amount": 6000.0,
        }
    )
    second = first.model_copy(update={"invoice_id": "INV-9102"})
    service.analyze(first)
    service.analyze(second)


# ---------------------------------------------------------------------------
# The triggers
# ---------------------------------------------------------------------------


def test_a_small_quiet_pair_stays_audit_only(service):
    _pair(service)
    view = service.queue_view()
    assert view.twin_escalations == []
    assert view.twin_escalation_count == 0
    # The thresholds in force ride on the view.
    assert view.twin_cluster_min_size == 3
    assert view.twin_cluster_value_threshold == 10000.0


def test_three_arrivals_escalate_on_size_with_the_graph_attached(service):
    service.analyze(SAMPLES["INV-1001"])
    service.analyze_bundle(_bundle("BND-TWIN", _SEGMENT_TEXT))
    third = SAMPLES["INV-1001"].model_copy(
        update={
            "invoice_id": "INV-9001",
            "supplier_name": "northbeam office supplies",
            "invoice_number": "nb 4471",
        }
    )
    service.analyze(third)

    view = service.queue_view()
    assert view.twin_escalation_count == 1
    item = view.twin_escalations[0]
    assert item.triggers == ["cluster_size"]
    assert item.size == 3
    assert item.totals_by_currency == {"USD": 3540.0}
    assert {m["invoice_id"] for m in item.members} >= {"INV-1001", "INV-9001"}
    assert "bundle_segment" in item.arrival_shapes
    assert item.edges  # the screened duplicate_of links ride along
    assert all(
        e["basis"].startswith("screened") for e in item.edges
    )


def test_a_pair_escalates_on_combined_value(service):
    _big_pair(service)
    view = service.queue_view()
    assert view.twin_escalation_count == 1
    item = view.twin_escalations[0]
    assert item.size == 2
    assert item.triggers == ["combined_value"]
    assert item.totals_by_currency == {"USD": 12000.0}


def test_an_approved_twin_spanning_a_waiting_one_always_escalates(service):
    _pair(service)
    service.approve("INV-1001", approver="Priya Nair")
    view = service.queue_view()
    assert view.twin_escalation_count == 1
    item = view.twin_escalations[0]
    # Neither size nor value fires — the span alone seats it.
    assert item.triggers == ["approved_and_awaiting"]
    statuses = {m["invoice_id"]: m["status"] for m in item.members}
    assert statuses["INV-1001"] == "approved"
    assert statuses["INV-1004"] == "awaiting_approver"


def test_the_span_outranks_value_in_work_order(service):
    _pair(service)
    _big_pair(service)
    service.approve("INV-1001", approver="Priya Nair")
    view = service.queue_view()
    assert view.twin_escalation_count == 2
    # The live double-payment risk (2360 committed + waiting)
    # precedes the larger but undecided cluster (12000).
    assert view.twin_escalations[0].triggers == ["approved_and_awaiting"]
    assert view.twin_escalations[1].triggers == ["combined_value"]


# ---------------------------------------------------------------------------
# Configuration + surfaces
# ---------------------------------------------------------------------------


def test_min_size_is_env_tunable(service, monkeypatch):
    monkeypatch.setenv("TWIN_CLUSTER_MIN_SIZE", "2")
    _pair(service)
    view = service.queue_view()
    assert view.twin_cluster_min_size == 2
    assert view.twin_escalation_count == 1
    assert view.twin_escalations[0].triggers == ["cluster_size"]


def test_value_threshold_is_env_tunable(service, monkeypatch):
    monkeypatch.setenv("TWIN_CLUSTER_VALUE_THRESHOLD", "2000")
    _pair(service)
    view = service.queue_view()
    assert view.twin_cluster_value_threshold == 2000.0
    assert view.twin_escalation_count == 1
    assert view.twin_escalations[0].triggers == ["combined_value"]


def test_direct_build_queue_callers_default_to_no_twin_section():
    view = build_queue([], as_of=None)
    assert view.twin_escalations == []
    assert view.twin_escalation_count == 0


def test_the_function_reads_records_directly(service):
    _pair(service)
    records = service._get_store().list_records()
    assert twin_escalation_items(records) == []
    escalated = twin_escalation_items(records, min_size=2)
    assert len(escalated) == 1
    assert escalated[0].cluster_id.startswith("TWIN-")


def test_queue_api_carries_the_escalations(service):
    service.analyze(SAMPLES["INV-1001"])
    service.analyze_bundle(_bundle("BND-TWIN", _SEGMENT_TEXT))
    third = SAMPLES["INV-1001"].model_copy(
        update={
            "invoice_id": "INV-9001",
            "supplier_name": "northbeam office supplies",
            "invoice_number": "nb 4471",
        }
    )
    service.analyze(third)
    client = TestClient(create_app(service))
    response = client.get("/queue")
    assert response.status_code == 200
    body = response.json()
    assert body["twin_escalation_count"] == 1
    assert body["twin_escalations"][0]["size"] == 3
    assert body["twin_escalations"][0]["triggers"] == ["cluster_size"]
