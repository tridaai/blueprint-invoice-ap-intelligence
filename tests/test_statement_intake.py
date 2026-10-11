"""Statement document intake: a supplier statement that arrives as
a document is read through the OCR + vision ports and parsed in
code into the same reconciliation the structured path runs — with
the confidence of every read on the record, low-confidence lines
flagged, credit lines read as credits, and non-statements refused
with the reason stored."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import (
    sample_invoice_models,
    sample_statement_document_models,
)
from ap_agent.schemas import DocumentInput
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
DOCS = {d.document_id: d for d in sample_statement_document_models()}


@pytest.fixture()
def service() -> InvoiceService:
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    # Northbeam: INV-1001 (NB-4471, 1180.00), INV-1008 (NB-4480, 1180.00).
    # Cascade: INV-1002 (CI-2210, 1274.40).
    for invoice_id in ("INV-1001", "INV-1008", "INV-1002"):
        svc.analyze(SAMPLES[invoice_id])
    return svc


def test_document_twin_reconciles_like_the_structured_statement(service):
    report = service.reconcile_statement_document(DOCS["DOC-STMT-NB-2026-10"])
    assert report.source == "document"
    assert report.refused is False
    assert report.statement_id == "STMT-NB-2026-10"
    assert report.supplier_name == "Northbeam Office Supplies"
    assert report.statement_date == "2026-10-05"
    # Exactly the structured twin's figures.
    assert report.matched_count == 1
    assert report.differs_count == 1
    assert report.missing_from_ap_count == 1
    assert report.missing_from_statement_count == 0
    assert report.statement_total == 2355.00
    assert report.ap_total == 2360.00
    assert report.claimed_total == 2400.00
    assert report.claimed_total_difference == 45.00
    # The extraction read rides on the record.
    extraction = report.extraction
    assert extraction is not None and extraction.accepted
    assert extraction.detected_type == "remittance_note"
    header = {f.field: f for f in extraction.header}
    assert header["supplier_name"].value == "Northbeam Office Supplies"
    assert header["claimed_total"].value == "2400.00"


def test_garbled_line_is_flagged_low_confidence_not_trusted(service):
    report = service.reconcile_statement_document(DOCS["DOC-STMT-NB-2026-10"])
    by_number = {line.invoice_number: line for line in report.lines}
    garbled = by_number["NB-9999"]  # the document writes its amount 75.0O
    assert garbled.statement_amount == 75.00  # recovered by confusable repair
    assert garbled.confidence == 0.45
    assert garbled.low_confidence is True
    assert by_number["NB-4471"].low_confidence is False
    assert report.low_confidence_count == 1
    assert "low confidence" in report.summary and "NB-9999" in report.summary


def test_credit_lines_are_read_as_credits(service):
    report = service.reconcile_statement_document(DOCS["DOC-STMT-CI-2026-11"])
    assert report.refused is False
    by_number = {line.invoice_number: line for line in report.lines}
    assert by_number["CI-2210"].status == "matched"
    credit = by_number["CN-3002"]
    assert credit.status == "credit_line"
    assert credit.statement_amount == -120.00
    assert report.credit_line_count == 1
    assert report.credit_total == -120.00
    # The credit signs into the recomputed total, so the claimed
    # total (which nets it) agrees — and the credit never counts
    # as an invoice AP failed to process.
    assert report.statement_total == 1154.40
    assert report.claimed_total_difference == 0.00
    assert report.missing_from_ap_count == 0
    assert report.missing_from_statement_count == 0
    assert "credit" in report.summary


def test_an_invoice_document_is_refused_with_the_reason_on_the_record(service):
    invoice_doc = SAMPLES["INV-1001"].documents[0]
    report = service.reconcile_statement_document(invoice_doc)
    assert report.refused is True
    assert "invoice" in report.refusal_reason
    assert report.lines == []
    assert report.matched_count == 0
    # The refusal is stored, fetchable, and reconciled nothing.
    fetched = service.get_reconciliation(invoice_doc.document_id)
    assert fetched is not None and fetched.refused is True
    assert fetched.summary == fetched.refusal_reason


def test_a_document_with_no_readable_lines_is_refused(service):
    doc = DocumentInput(
        doc_type="statement",
        document_id="DOC-EMPTY-STMT",
        ocr_text="Statement of Account\n\nNo lines survived the scan.",
        ocr_confidence=0.9,
    )
    report = service.reconcile_statement_document(doc)
    assert report.refused is True
    assert "No statement lines" in report.refusal_reason


def test_a_poor_scan_flags_every_line(service):
    doc = DOCS["DOC-STMT-NB-2026-10"].model_copy(
        update={"ocr_confidence": 0.5, "scan_quality": "poor"}
    )
    report = service.reconcile_statement_document(doc)
    assert report.refused is False
    assert report.low_confidence_count == 3
    assert all(line.low_confidence for line in report.lines)
    # The figures still reconcile — flagged, not dropped.
    assert report.statement_total == 2355.00


def test_an_unparseable_amount_keeps_the_line_flagged(service):
    doc = DocumentInput(
        doc_type="statement",
        document_id="DOC-STMT-GARBLED",
        ocr_confidence=0.97,
        ocr_text=(
            "Northbeam Office Supplies\n"
            "Statement of Account\n"
            "Statement: STMT-NB-GARBLED\n"
            "Date: 2026-10-05\n"
            "Currency: USD\n\n"
            "NB-4471    2026-09-28  1,180.00\n"
            "NB-4472    2026-09-30  smudged\n"
        ),
    )
    report = service.reconcile_statement_document(doc)
    assert report.refused is False
    by_number = {line.invoice_number: line for line in report.lines}
    assert by_number["NB-4471"].status == "matched"
    unknown = by_number["NB-4472"]
    assert unknown.statement_amount is None
    assert unknown.low_confidence is True
    assert unknown.status == "missing_from_ap"


def test_document_reconcile_endpoints():
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])
    client = TestClient(create_app(svc))

    samples = client.get("/samples/statement-documents")
    assert samples.status_code == 200
    assert {d["document_id"] for d in samples.json()} == {
        "DOC-STMT-NB-2026-10",
        "DOC-STMT-CI-2026-11",
    }

    payload = DOCS["DOC-STMT-NB-2026-10"].model_dump(mode="json")
    response = client.post("/statements/reconcile-document", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "document"
    assert body["statement_id"] == "STMT-NB-2026-10"
    assert body["low_confidence_count"] == 1
    assert body["extraction"]["detected_type"] == "remittance_note"

    refused = client.post(
        "/statements/reconcile-document",
        json=SAMPLES["INV-1002"].documents[0].model_dump(mode="json"),
    )
    assert refused.status_code == 200
    assert refused.json()["refused"] is True
