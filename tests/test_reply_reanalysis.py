"""Reply-driven re-analysis: a reply that carries the documents
and closes the gap re-runs the case.

Round 16 closed the clarification loop on the record, but a fully
answered request still left the case sitting at its blocked
triage — a human had to notice and re-submit it by hand. Now the
reply can carry the supplied documents themselves (the PO, the
GRN, the legible copy); they accumulate on the request as a named
overlay, and when a reply completes the gap the case re-runs
with the overlay merged over the stored invoice. The new outcome
supersedes, the supersession is recorded in the case's
re-analysis chain, and the stored invoice is never rewritten —
original + recorded overlay reconstructs every re-run's input.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import DocumentInput, GoodsReceipt, LineItem, PurchaseOrder
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


@pytest.fixture()
def service() -> InvoiceService:
    return InvoiceService(store=InMemoryStore(), checkpointer=False)


def _po_1005() -> PurchaseOrder:
    """The PO purchasing raises for INV-1005's catering invoice."""
    return PurchaseOrder(
        po_number="PO-9019",
        supplier_name="Bluepine Catering Co",
        currency="USD",
        lines=[
            LineItem(
                description="Event catering service",
                sku="SVC-CAT",
                quantity=1,
                unit_price=500.0,
            )
        ],
        total=500.0,
    )


def _grn_1005() -> GoodsReceipt:
    return GoodsReceipt(
        receipt_id="GRN-519",
        po_number="PO-9019",
        received_date="2026-09-26",
        lines=[
            LineItem(
                description="Event catering service",
                sku="SVC-CAT",
                quantity=1,
                unit_price=500.0,
            )
        ],
    )


def _grn_1006() -> GoodsReceipt:
    return GoodsReceipt(
        receipt_id="GRN-506",
        po_number="PO-9006",
        received_date="2026-09-25",
        lines=[
            LineItem(
                description="Deep cleaning service",
                sku="SVC-CLN",
                quantity=1,
                unit_price=2400.0,
            )
        ],
    )


# -- the completing reply re-runs the case -------------------------------


def test_grn_reply_reanalyses_a_missing_receipt_case_to_clean(service):
    blocked = service.analyze(SAMPLES["INV-1006"])
    assert blocked.triage.outcome == "missing_receipt"
    result = service.record_information_reply(
        "INV-1006",
        actor="Sam (AP)",
        supplied_items=["goods_receipt"],
        goods_receipt=_grn_1006(),
        note="Facilities filed GRN-506",
        received_at="2026-10-08T10:00:00+00:00",
    )
    # The blocked outcome is superseded by the re-run's.
    assert result.triage.outcome == "clean_match"
    assert result.facts.has_receipt
    assert result.facts.match_status == "full_match"
    assert result.status == "awaiting_approver"
    # The closed loop stays on the case.
    assert result.information_request.reply_status == "answered"
    # The chain records the supersession, with the overlay in hand.
    (entry,) = result.reanalysis_history
    assert entry.trigger == "information_reply"
    assert entry.previous_outcome == "missing_receipt"
    assert entry.new_outcome == "clean_match"
    assert entry.overlay == ["goods_receipt"]
    assert entry.overlay_goods_receipt.receipt_id == "GRN-506"
    assert entry.closed_request.reply_status == "answered"
    assert entry.by == "Sam (AP)"
    # The stored invoice is untouched: the GRN lives in the
    # overlay, not in the original extraction.
    record = service._get_store().get("INV-1006")
    assert record.invoice["goods_receipt"] is None
    assert record.result is result


def test_po_reply_reanalyses_and_the_new_gap_opens_its_own_request(service):
    blocked = service.analyze(SAMPLES["INV-1005"])
    assert blocked.triage.outcome == "missing_po"
    result = service.record_information_reply(
        "INV-1005",
        actor="Sam (AP)",
        supplied_items=["purchase_order"],
        purchase_order=_po_1005(),
        note="PO-9019 raised by purchasing",
        received_at="2026-10-08T10:00:00+00:00",
    )
    # With the PO in hand the match runs — and finds the next
    # gap, which opens its own fresh request.
    assert result.triage.outcome == "missing_receipt"
    assert result.facts.has_po
    request = result.information_request
    assert request is not result.reanalysis_history[0].closed_request
    assert request.missing_item_keys == ["goods_receipt"]
    assert request.reply_status == "awaiting_reply"
    (entry,) = result.reanalysis_history
    assert entry.previous_outcome == "missing_po"
    assert entry.new_outcome == "missing_receipt"
    assert entry.closed_request.missing_item_keys == ["purchase_order"]


def test_the_chain_accumulates_across_two_loops(service):
    service.analyze(SAMPLES["INV-1005"])
    service.record_information_reply(
        "INV-1005",
        actor="Sam (AP)",
        supplied_items=["purchase_order"],
        purchase_order=_po_1005(),
    )
    result = service.record_information_reply(
        "INV-1005",
        actor="Jo (AP)",
        supplied_items=["goods_receipt"],
        goods_receipt=_grn_1005(),
        note="Catering confirmed received",
    )
    # The second re-run ran on BOTH overlays: without the first
    # loop's PO it would have fallen back to missing_po.
    assert result.triage.outcome == "clean_match"
    assert [e.previous_outcome for e in result.reanalysis_history] == [
        "missing_po",
        "missing_receipt",
    ]
    assert [e.new_outcome for e in result.reanalysis_history] == [
        "missing_receipt",
        "clean_match",
    ]
    assert result.reanalysis_history[1].overlay == ["goods_receipt"]
    # The final case carries the second loop's closed request.
    assert result.information_request.reply_status == "answered"
    record = service._get_store().get("INV-1005")
    assert record.invoice["purchase_order"] is None
    assert record.invoice["goods_receipt"] is None


def test_po_and_grn_in_one_reply_close_the_case_in_one_run(service):
    service.analyze(SAMPLES["INV-1005"])
    result = service.record_information_reply(
        "INV-1005",
        actor="Sam (AP)",
        supplied_items=["purchase_order", "goods_receipt"],
        purchase_order=_po_1005(),
        goods_receipt=_grn_1005(),
    )
    assert result.triage.outcome == "clean_match"
    (entry,) = result.reanalysis_history
    assert entry.overlay == ["purchase_order", "goods_receipt"]


# -- what does NOT re-run ---------------------------------------------------


def test_keys_only_completion_stays_bookkeeping(service):
    service.analyze(SAMPLES["INV-1005"])
    result = service.record_information_reply(
        "INV-1005",
        actor="Sam (AP)",
        supplied_items=["purchase_order"],
        note="PO-9019 raised — paper copy follows",
    )
    # The assertion is recorded; there is nothing new to run the
    # match against, so the blocked outcome stands.
    assert result.triage.outcome == "missing_po"
    assert result.information_request.reply_status == "answered"
    assert result.reanalysis_history == []


def test_a_decided_cases_reply_never_reanalyses(service):
    service.analyze(SAMPLES["INV-1005"])
    service.approve("INV-1005", approver="Priya Nair", reason="verbal order confirmed")
    result = service.record_information_reply(
        "INV-1005",
        actor="Sam (AP)",
        supplied_items=["purchase_order"],
        purchase_order=_po_1005(),
    )
    # The reply is on the record; the human's decision stands and
    # no re-run unmakes it.
    assert result.status == "approved"
    assert result.triage.outcome == "missing_po"
    assert result.reanalysis_history == []
    assert result.information_request.reply_status == "answered"


# -- the overlay's guards ---------------------------------------------------


def test_a_payload_must_be_accounted_for_by_a_supplied_key(service):
    service.analyze(SAMPLES["INV-1006"])
    with pytest.raises(ValueError, match="does not record supplying"):
        service.record_information_reply(
            "INV-1006",
            actor="Sam (AP)",
            supplied_items=[],
            goods_receipt=_grn_1006(),
        )


def test_a_payload_cannot_replace_a_document_on_file(service):
    service.analyze(SAMPLES["INV-1006"])  # has a PO; asks for the GRN
    with pytest.raises(ValueError, match="already has one on file"):
        service.record_information_reply(
            "INV-1006",
            actor="Sam (AP)",
            supplied_items=["purchase_order"],
            purchase_order=_po_1005(),
        )
    # The refusal refused the recording whole: the request is
    # still open and untouched.
    request = service.get("INV-1006").information_request
    assert request.reply_status == "awaiting_reply"
    assert request.reply_received_at == ""


def test_a_legible_copy_must_name_a_document_the_case_holds(service):
    service.analyze(SAMPLES["INV-1009"])
    stranger = DocumentInput(
        doc_type="invoice",
        document_id="DOC-XXXX",
        ocr_text="A perfectly legible invoice for somebody else",
        ocr_confidence=0.99,
        scan_quality="good",
    )
    with pytest.raises(ValueError, match="does not hold"):
        service.record_information_reply(
            "INV-1009",
            actor="Sam (AP)",
            supplied_items=["legible_invoice"],
            documents=[stranger],
        )


# -- documents accumulate across a partial reply ----------------------------


def _legible_1009() -> DocumentInput:
    return DocumentInput(
        doc_type="invoice",
        document_id="DOC-1009",
        fields={
            "invoice_number": "CI-2215",
            "supplier_name": "Cascade IT Hardware",
            "currency": "USD",
            "po_number": "PO-9021",
            "subtotal": "250.00",
            "tax_amount": "45.00",
            "total": "295.00",
        },
        ocr_text=(
            "Cascade IT Hardware\nInvoice CI-2215\nDate: 2026-09-29\n"
            "PO: PO-9021\nUSB-C docking station x1 @ 250.00 = 250.00\n"
            "Subtotal 250.00\nTax 45.00\nTotal 295.00 USD"
        ),
        ocr_confidence=0.98,
        scan_quality="good",
    )


def _po_1009() -> PurchaseOrder:
    return PurchaseOrder(
        po_number="PO-9021",
        supplier_name="Cascade IT Hardware",
        currency="USD",
        lines=[
            LineItem(
                description="USB-C docking station",
                sku="DOCK-USB4",
                quantity=1,
                unit_price=250.0,
            )
        ],
        total=250.0,
    )


def test_legible_copy_accumulates_until_the_gap_closes(service):
    blocked = service.analyze(SAMPLES["INV-1009"])
    assert blocked.triage.outcome == "unreadable"
    assert set(blocked.information_request.missing_item_keys) == {
        "purchase_order",
        "legible_invoice",
    }
    partial = service.record_information_reply(
        "INV-1009",
        actor="Sam (AP)",
        supplied_items=["legible_invoice"],
        documents=[_legible_1009()],
        note="Supplier re-scanned the invoice",
        received_at="2026-10-05T09:00:00+00:00",
    )
    # Partial: the copy is on the request's overlay, but the case
    # is not re-run until the gap closes.
    assert partial.triage.outcome == "unreadable"
    assert partial.reanalysis_history == []
    assert partial.information_request.reply_status == "partially_answered"
    assert len(partial.information_request.overlay_documents) == 1
    closed = service.record_information_reply(
        "INV-1009",
        actor="Sam (AP)",
        supplied_items=["purchase_order"],
        purchase_order=_po_1009(),
        note="PO-9021 from purchasing",
        received_at="2026-10-07T09:00:00+00:00",
    )
    # The re-run read the legible copy AND the PO: no longer
    # unreadable, and the match now runs to its next gap.
    assert closed.triage.outcome == "missing_receipt"
    (entry,) = closed.reanalysis_history
    assert entry.overlay == ["purchase_order", "legible_invoice"]
    assert entry.overlay_documents[0].document_id == "DOC-1009"
    # The stored original still carries the unreadable scan.
    record = service._get_store().get("INV-1009")
    (stored_doc,) = record.invoice["documents"]
    assert stored_doc["scan_quality"] == "unreadable"


# -- the API ------------------------------------------------------------------


def test_information_reply_endpoint_reanalyses(service):
    service.analyze(SAMPLES["INV-1006"])
    client = TestClient(create_app(service))
    response = client.post(
        "/invoices/INV-1006/information-reply",
        json={
            "actor": "Sam (AP)",
            "supplied_items": ["goods_receipt"],
            "goods_receipt": _grn_1006().model_dump(mode="json"),
            "note": "GRN filed",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["triage"]["outcome"] == "clean_match"
    assert body["reanalysis_history"][0]["previous_outcome"] == "missing_receipt"
    assert body["reanalysis_history"][0]["overlay"] == ["goods_receipt"]
