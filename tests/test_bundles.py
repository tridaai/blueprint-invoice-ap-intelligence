"""Multi-invoice bundles: one arrival carrying several invoices is
segmented in code by invoice-number anchors, and every segment runs
the full pipeline as its own invoice with its own gate
(bundles.py + service.analyze_bundle)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.bundles import segment_bundle
from ap_agent.samples import load_sample_bundles, sample_bundle_models
from ap_agent.schemas import InvoiceBundleInput
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore


@pytest.fixture()
def bundle() -> InvoiceBundleInput:
    return sample_bundle_models()[0]


def test_segmentation_splits_on_invoice_number_anchors(bundle):
    invoices, note = segment_bundle(bundle)
    assert [inv.invoice_id for inv in invoices] == ["BND-4001-01", "BND-4001-02"]
    assert [inv.invoice_number for inv in invoices] == ["NB-4490", "NB-4491"]
    assert "segmented into 2 invoice(s)" in note
    assert "1 document(s) treated as context" in note  # the cover email


def test_segment_parsing_reads_lines_totals_and_dates(bundle):
    invoices, _ = segment_bundle(bundle)
    first, second = invoices
    assert first.total_amount == 212.40
    assert first.tax_amount == 32.40
    assert first.invoice_date == "2026-10-01"
    assert first.due_date == "2026-10-31"
    assert len(first.line_items) == 1
    assert first.line_items[0].description == "A4 paper case"
    assert first.line_items[0].sku == "PAP-A4"  # adopted from the PO line
    assert first.line_items[0].quantity == 4
    assert first.line_items[0].unit_price == 45.0
    assert second.total_amount == 259.60
    # The bundle's system records attach by PO number.
    assert first.purchase_order is not None
    assert first.purchase_order.po_number == "PO-9015"
    assert first.goods_receipt is not None
    # The segment document carries the parsed fields for cross-check.
    doc = first.documents[0]
    assert doc.fields["invoice_number"] == "NB-4490"
    assert doc.fields["total"] == "212.40"


def test_bundle_processes_each_invoice_as_its_own_run(bundle):
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    summary = svc.analyze_bundle(bundle)
    assert summary.segment_count == 2
    assert summary.processed_count == 2
    assert summary.outcome_counts == {"clean_match": 2}
    assert summary.totals_by_currency == {"USD": 472.0}
    for invoice_id in ("BND-4001-01", "BND-4001-02"):
        result = svc.get(invoice_id)
        assert result is not None
        assert result.status == "awaiting_approver"  # its own gate
        assert result.facts.match_status == "full_match"
    stored = svc.get_bundle("BND-4001")
    assert stored is not None and stored.segment_count == 2


def test_bundle_invoice_decisions_are_independent(bundle):
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze_bundle(bundle)
    svc.approve("BND-4001-01", approver="Priya Nair")
    assert svc.get("BND-4001-01").status == "approved"
    assert svc.get("BND-4001-02").status == "awaiting_approver"


def test_two_single_invoice_documents_form_a_bundle():
    payload = {
        "bundle_id": "BND-T2",
        "supplier_name": "Cascade IT Hardware",
        "submitted_at": "2026-10-03T10:00:00",
        "documents": [
            {
                "doc_type": "invoice",
                "document_id": "D1",
                "ocr_text": "Cascade IT Hardware\nInvoice CI-3001\nDate: 2026-10-02\nCable x2 @ 25.00 = 50.00\nSubtotal 50.00\nTax 9.00\nTotal 59.00 USD",
            },
            {
                "doc_type": "invoice",
                "document_id": "D2",
                "ocr_text": "Cascade IT Hardware\nInvoice CI-3002\nDate: 2026-10-02\nCable x1 @ 25.00 = 25.00\nSubtotal 25.00\nTax 4.50\nTotal 29.50 USD",
            },
        ],
    }
    invoices, _ = segment_bundle(InvoiceBundleInput(**payload))
    assert [inv.invoice_number for inv in invoices] == ["CI-3001", "CI-3002"]
    assert invoices[0].total_amount == 59.0
    assert invoices[1].total_amount == 29.5


def test_document_without_anchor_is_one_segment():
    payload = {
        "bundle_id": "BND-T3",
        "supplier_name": "Bluepine Catering Co",
        "documents": [
            {
                "doc_type": "invoice",
                "document_id": "D1",
                "ocr_text": "Bluepine Catering Co\nCatering for the October event.\nTotal 120.00 USD",
            }
        ],
    }
    invoices, _ = segment_bundle(InvoiceBundleInput(**payload))
    assert len(invoices) == 1
    assert invoices[0].invoice_number == ""
    assert invoices[0].total_amount == 120.0


def test_empty_bundle_summarises_to_zero():
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    summary = svc.analyze_bundle({"bundle_id": "BND-T4", "documents": []})
    assert summary.segment_count == 0
    assert summary.processed_count == 0
    assert summary.outcome_counts == {}


def test_bundle_endpoints():
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    client = TestClient(create_app(svc))
    missing = client.get("/bundles/BND-4001")
    assert missing.status_code == 404
    response = client.post("/bundles/analyze", json=load_sample_bundles()[0])
    assert response.status_code == 200
    body = response.json()
    assert body["bundle_id"] == "BND-4001"
    assert body["segment_count"] == 2
    assert body["outcome_counts"] == {"clean_match": 2}
    fetched = client.get("/bundles/BND-4001")
    assert fetched.status_code == 200
    assert fetched.json()["processed_count"] == 2
