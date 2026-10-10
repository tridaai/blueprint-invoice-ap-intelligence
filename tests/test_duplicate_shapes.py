"""Duplicate screening across arrival shapes: the same invoice can
arrive as a structured single, as a segment cut out of a combined
bundle scan, or as a line on a supplier statement — and the
duplicate control must not depend on which shape arrived first.

- The store compares canonical keys (supplier case/spacing and
  invoice-number separators normalised), so a bundle segment that
  re-sends a portal single is flagged with the single as its twin,
  and a single that re-sends a bundle segment is flagged with the
  segment as its twin — the same recovery path opens either way.
- Statement lines are screened too: a line whose number AP holds
  twice, or that the statement itself lists twice, is flagged with
  the twin reference.
"""

from __future__ import annotations

import pytest

from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore, duplicate_key, find_duplicate

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


def _bundle(bundle_id: str, text: str, supplier: str = "Northbeam Office Supplies") -> dict:
    return {
        "bundle_id": bundle_id,
        "supplier_name": supplier,
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


# ---------------------------------------------------------------------------
# Canonical keys
# ---------------------------------------------------------------------------


def test_duplicate_key_normalises_case_spacing_and_separators():
    assert duplicate_key("Northbeam Office Supplies", "NB-4471") == duplicate_key(
        "  northbeam   office supplies ", "nb 4471"
    )
    assert duplicate_key("Acme, Inc.", "INV-100") == duplicate_key("acme inc", "INV100")
    assert duplicate_key("Acme", "") == ("acme", "")


def test_find_duplicate_matches_across_written_forms():
    priors = [
        {
            "invoice_id": "INV-A",
            "supplier_name": "Northbeam Office Supplies",
            "invoice_number": "NB-4471",
            "total_amount": 1180.0,
        }
    ]

    class _Invoice:
        supplier_name = "NORTHBEAM OFFICE SUPPLIES"
        invoice_number = "NB 4471"
        total_amount = 1180.0

    assert find_duplicate(priors, _Invoice()) == "INV-A"


def test_find_duplicate_still_requires_same_total():
    priors = [
        {
            "invoice_id": "INV-A",
            "supplier_name": "Northbeam Office Supplies",
            "invoice_number": "NB-4471",
            "total_amount": 1180.0,
        }
    ]

    class _Invoice:
        supplier_name = "Northbeam Office Supplies"
        invoice_number = "NB-4471"
        total_amount = 1181.0

    assert find_duplicate(priors, _Invoice()) == ""


# ---------------------------------------------------------------------------
# Bundle segment <-> portal single, both directions
# ---------------------------------------------------------------------------


def test_bundle_segment_duplicating_an_approved_single_is_flagged(service):
    service.analyze(SAMPLES["INV-1001"])
    service.approve("INV-1001", approver="Priya Nair", reason="clean match")
    summary = service.analyze_bundle(_bundle("BND-5001", _SEGMENT_TEXT))
    assert summary.segment_count == 1
    outcome = summary.outcomes[0]
    assert outcome.invoice_id == "BND-5001-01"
    assert outcome.outcome == "duplicate_suspected"
    result = service.get("BND-5001-01")
    assert result is not None
    assert result.facts.duplicate_of == "INV-1001"
    # The same recovery path opens for a segment as for a single.
    pack = service.draft_duplicate_recovery("BND-5001-01")
    assert pack.original_invoice_id == "INV-1001"
    assert pack.amount == 1180.0
    assert pack.exposure_status == "twin_approved"


def test_single_duplicating_a_bundle_segment_is_flagged(service):
    summary = service.analyze_bundle(_bundle("BND-5002", _SEGMENT_TEXT))
    assert summary.outcomes[0].outcome != "duplicate_suspected"
    service.analyze(SAMPLES["INV-1013"])  # the portal resend, as a single
    result = service.get("INV-1013")
    assert result is not None
    assert result.triage.outcome == "duplicate_suspected"
    assert result.facts.duplicate_of == "BND-5002-01"


def test_second_identical_segment_in_one_bundle_is_flagged(service):
    text = _SEGMENT_TEXT + "\n" + _SEGMENT_TEXT.split("\n\n", 1)[1]
    summary = service.analyze_bundle(_bundle("BND-5003", text))
    assert summary.segment_count == 2
    first, second = summary.outcomes
    assert first.outcome != "duplicate_suspected"
    assert second.outcome == "duplicate_suspected"
    result = service.get(second.invoice_id)
    assert result is not None
    assert result.facts.duplicate_of == first.invoice_id


def test_case_variant_supplier_still_duplicates(service):
    service.analyze(SAMPLES["INV-1001"])
    summary = service.analyze_bundle(
        _bundle("BND-5004", _SEGMENT_TEXT, supplier="NORTHBEAM OFFICE SUPPLIES")
    )
    assert summary.outcomes[0].outcome == "duplicate_suspected"


# ---------------------------------------------------------------------------
# Statement lines
# ---------------------------------------------------------------------------


def _statement(lines: list[dict]) -> dict:
    return {
        "statement_id": "STMT-NB-DUP",
        "supplier_name": "Northbeam Office Supplies",
        "statement_date": "2026-10-05",
        "currency": "USD",
        "lines": lines,
    }


def test_statement_line_flagged_when_ap_holds_the_invoice_twice(service):
    service.analyze(SAMPLES["INV-1001"])
    service.analyze(SAMPLES["INV-1013"])  # the resend: AP now holds two records
    report = service.reconcile_statement(
        _statement([{"invoice_number": "NB-4471", "amount": 1180.00}])
    )
    line = report.lines[0]
    assert line.status == "matched"
    assert line.duplicate_suspected is True
    assert line.duplicate_invoice_id == "INV-1013"
    assert line.duplicate_of == "INV-1001"
    assert report.duplicate_count == 1
    assert "paying this line" in report.summary
    assert "suspected duplicates" in report.summary


def test_statement_line_repeated_on_the_statement_is_flagged(service):
    service.analyze(SAMPLES["INV-1001"])
    report = service.reconcile_statement(
        _statement(
            [
                {"invoice_number": "NB-4471", "amount": 1180.00},
                {"invoice_number": "NB-4471", "amount": 1180.00},
            ]
        )
    )
    first, second = report.lines
    assert first.duplicate_suspected is False
    assert second.duplicate_suspected is True
    assert second.duplicate_invoice_id == "INV-1001"
    assert report.duplicate_count == 1


def test_statement_matching_uses_the_canonical_number(service):
    service.analyze(SAMPLES["INV-1001"])
    report = service.reconcile_statement(
        _statement([{"invoice_number": "NB 4471", "amount": 1180.00}])
    )
    line = report.lines[0]
    assert line.status == "matched"
    assert line.ap_invoice_id == "INV-1001"
    assert line.duplicate_suspected is False
    assert report.missing_from_statement_count == 0
