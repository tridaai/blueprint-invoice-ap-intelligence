"""Duplicate-twin graph: the audit view of cross-shape screening.

Screening flags the later arrival and names its twin on that one
record; the graph reads the links back as clusters, so a reviewer
sees the whole set — original, copies, and the shape each arrived
in — without opening every record. The audit export gains the
twin columns (duplicate_of, both sides' arrival shapes, cluster
id + size) on the same read.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore, _dump_record, _load_record
from ap_agent.twins import twin_clusters

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


def _three_copies(service: InvoiceService) -> None:
    """The same invoice (NB-4471, USD 1180) three times: a portal
    single, a bundle segment, and a re-typed single."""
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


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


def test_three_arrivals_form_one_cluster_with_shapes_and_edges(service):
    _three_copies(service)
    clusters = twin_clusters(service._get_store().list_records())

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster["size"] == 3
    assert cluster["canonical_invoice_number"] == "NB4471"
    assert cluster["totals"] == [1180.0]
    assert cluster["arrival_shapes"] == ["bundle_segment", "single"]
    members = {m["invoice_id"]: m for m in cluster["members"]}
    assert members["INV-1001"]["arrival_shape"] == "single"
    assert members["BND-TWIN-01"]["arrival_shape"] == "bundle_segment"
    assert members["BND-TWIN-01"]["duplicate_of"] == "INV-1001"
    edges = {(e["duplicate"], e["original"]) for e in cluster["edges"]}
    assert ("BND-TWIN-01", "INV-1001") in edges
    assert ("INV-9001", "INV-1001") in edges


def test_distinct_invoices_form_no_cluster(service):
    service.analyze(SAMPLES["INV-1001"])
    service.analyze(SAMPLES["INV-1002"])
    assert twin_clusters(service._get_store().list_records()) == []
    assert service.duplicate_twins() == {"clusters": [], "count": 0}


# ---------------------------------------------------------------------------
# The audit export's twin columns
# ---------------------------------------------------------------------------


def test_audit_rows_carry_the_twin_columns(service):
    _three_copies(service)
    rows = {row["invoice_id"]: row for row in service.audit_rows()}

    segment = rows["BND-TWIN-01"]
    assert segment["duplicate_of"] == "INV-1001"
    assert segment["arrival_shape"] == "bundle_segment"
    assert segment["twin_arrival_shape"] == "single"
    assert segment["twin_cluster_size"] == 3
    original = rows["INV-1001"]
    assert original["duplicate_of"] == ""
    assert original["twin_cluster"] == segment["twin_cluster"]
    assert original["twin_cluster_size"] == 3


def test_audit_export_csv_and_twins_endpoint(service):
    _three_copies(service)
    client = TestClient(create_app(service))

    csv_export = client.get("/audit/export", params={"format": "csv"})
    assert csv_export.status_code == 200
    header = csv_export.text.strip().splitlines()[0]
    for column in (
        "duplicate_of",
        "arrival_shape",
        "twin_arrival_shape",
        "twin_cluster",
        "twin_cluster_size",
    ):
        assert column in header

    twins = client.get("/audit/twins")
    assert twins.status_code == 200
    body = twins.json()
    assert body["count"] == 1
    assert body["clusters"][0]["size"] == 3


# ---------------------------------------------------------------------------
# The arrival-shape stamp itself
# ---------------------------------------------------------------------------


def test_arrival_shape_round_trips_and_legacy_records_read_as_single(service):
    service.analyze(SAMPLES["INV-1001"])
    record = service._get_store().get("INV-1001")
    assert record.arrival_shape == "single"
    restored = _load_record(_dump_record(record))
    assert restored.arrival_shape == "single"

    payload = json.loads(_dump_record(record))
    del payload["arrival_shape"]  # a record saved before the stamp
    legacy = _load_record(json.dumps(payload))
    assert legacy.arrival_shape == "single"
