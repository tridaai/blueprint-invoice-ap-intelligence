"""The clarification reply loop: a supplier's answer is recorded
against the open request and analysed in code.

The information request was composed and policed outbound, but
the loop never closed: a reply landed in someone's inbox and a
human re-linked it to the case by memory, and an unanswered
request aged invisibly — the invoice sat at the gate while
purchasing or the supplier held the missing piece. Now the
reply is recorded on the record (which items it supplied, when,
a note), the resolution is arithmetic — answered, partially
answered with the still-missing items named, or still awaiting
— and an unclosed wait past its aging threshold joins the
approver queue like the other stalls.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.clarify import analyse_information_reply, humanize_item
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import InformationRequest
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
PO_PHRASE = humanize_item("purchase_order")
LEGIBLE_PHRASE = humanize_item("legible_invoice")


@pytest.fixture()
def service() -> InvoiceService:
    return InvoiceService(store=InMemoryStore(), checkpointer=False)


def _blocked(service: InvoiceService) -> object:
    """INV-1005: no PO was raised, so the run composes an
    information request for the purchase order."""
    result = service.analyze(SAMPLES["INV-1005"])
    assert result.needs_information
    assert result.information_request is not None
    return result


# -- the request carries its keys and its stamp -------------------------


def test_request_stamps_keys_and_requested_at(service):
    request = _blocked(service).information_request
    assert request.missing_item_keys == ["purchase_order"]
    assert request.missing_items == [PO_PHRASE]
    assert request.requested_at  # the wait's clock has a start
    assert request.reply_status == "awaiting_reply"


# -- recording + analysis ------------------------------------------------


def test_reply_supplying_the_missing_item_answers_the_request(service):
    _blocked(service)
    result = service.record_information_reply(
        "INV-1005",
        actor="Sam (AP)",
        supplied_items=["purchase_order"],
        note="PO-9019 raised by purchasing, attached to the reply",
        received_at="2026-10-08T10:00:00+00:00",
    )
    request = result.information_request
    assert request.reply_status == "answered"
    assert request.reply_supplied_items == [PO_PHRASE]
    assert request.reply_still_missing == []
    assert request.reply_received_at == "2026-10-08T10:00:00+00:00"
    assert request.reply_recorded_by == "Sam (AP)"
    assert "PO-9019" in request.reply_note


def test_partial_reply_names_what_is_still_missing():
    request = InformationRequest(
        needed=True,
        missing_items=[PO_PHRASE, LEGIBLE_PHRASE],
        missing_item_keys=["purchase_order", "legible_invoice"],
        requested_at="2026-10-01T09:00:00+00:00",
    )
    updated = analyse_information_reply(
        request, ["legible_invoice"], received_at="2026-10-05T09:00:00+00:00"
    )
    assert updated.reply_status == "partially_answered"
    assert updated.reply_supplied_items == [LEGIBLE_PHRASE]
    assert updated.reply_still_missing == [PO_PHRASE]
    # Replies accumulate: the second answer closes the gap.
    closed = analyse_information_reply(
        updated, ["purchase_order"], received_at="2026-10-07T09:00:00+00:00"
    )
    assert closed.reply_status == "answered"
    assert closed.reply_still_missing == []
    assert closed.reply_received_at == "2026-10-07T09:00:00+00:00"


def test_reply_supplying_nothing_asked_for_leaves_the_gap(service):
    _blocked(service)
    result = service.record_information_reply(
        "INV-1005",
        actor="Sam (AP)",
        supplied_items=["goods_receipt"],
        note="Supplier sent a delivery note for a different order",
        received_at="2026-10-08T10:00:00+00:00",
    )
    request = result.information_request
    # The attempt is on the record; the resolution is not dressed up.
    assert request.reply_status == "awaiting_reply"
    assert request.reply_received_at == "2026-10-08T10:00:00+00:00"
    assert request.reply_supplied_items == []
    assert request.reply_still_missing == [PO_PHRASE]


def test_reply_guards(service):
    _blocked(service)
    with pytest.raises(ValueError, match="Unknown information item"):
        service.record_information_reply(
            "INV-1005", actor="Sam", supplied_items=["bank_details"]
        )
    with pytest.raises(ValueError, match="must record something"):
        service.record_information_reply("INV-1005", actor="Sam", supplied_items=[])
    with pytest.raises(KeyError):
        service.record_information_reply(
            "INV-9999", actor="Sam", supplied_items=["purchase_order"]
        )
    # A case that never asked has no loop to close.
    service.analyze(SAMPLES["INV-1001"])
    with pytest.raises(ValueError, match="no information request"):
        service.record_information_reply(
            "INV-1001", actor="Sam", supplied_items=["purchase_order"]
        )
    # And an answered request refuses a second closing.
    service.record_information_reply(
        "INV-1005", actor="Sam", supplied_items=["purchase_order"]
    )
    with pytest.raises(ValueError, match="already fully answered"):
        service.record_information_reply(
            "INV-1005", actor="Sam", supplied_items=["purchase_order"]
        )


# -- the queue ages the wait ---------------------------------------------


def test_unanswered_wait_ages_onto_the_queue(service):
    _blocked(service)
    fresh = service.queue_view(as_of="2026-10-12")
    assert fresh.information_wait_count == 0  # inside the threshold
    aged = service.queue_view(as_of="2026-10-25")
    assert aged.information_wait_aging_days == 3
    assert aged.information_wait_count == 1
    item = aged.information_waits[0]
    assert item.invoice_id == "INV-1005"
    assert item.reply_status == "awaiting_reply"
    assert item.waiting_days > 3
    assert item.still_missing == [PO_PHRASE]


def test_answered_request_leaves_the_queue(service):
    _blocked(service)
    service.record_information_reply(
        "INV-1005",
        actor="Sam (AP)",
        supplied_items=["purchase_order"],
        received_at="2026-10-20T09:00:00+00:00",
    )
    view = service.queue_view(as_of="2026-11-15")
    assert view.information_wait_count == 0


def test_partial_reply_restarts_the_wait(service):
    _blocked(service)
    # Shape the record into a partially-answered request: the
    # legible copy arrived on Oct 20; the PO is still owed.
    record = service._get_store().get("INV-1005")
    request = record.result.information_request
    request.missing_items = [PO_PHRASE, LEGIBLE_PHRASE]
    request.missing_item_keys = ["purchase_order", "legible_invoice"]
    request.reply_status = "partially_answered"
    request.reply_received_at = "2026-10-20T09:00:00+00:00"
    request.reply_supplied_items = [LEGIBLE_PHRASE]
    request.reply_still_missing = [PO_PHRASE]
    service._get_store().save(record)
    # Two days after the partial reply the wait is young again,
    # even though the request itself is weeks old.
    young = service.queue_view(as_of="2026-10-22")
    assert young.information_wait_count == 0
    older = service.queue_view(as_of="2026-10-25")
    assert older.information_wait_count == 1
    item = older.information_waits[0]
    assert item.reply_status == "partially_answered"
    assert item.waiting_days == 5
    assert item.still_missing == [PO_PHRASE]


# -- the API ---------------------------------------------------------------


def test_information_reply_endpoint(service):
    _blocked(service)
    client = TestClient(create_app(service))
    response = client.post(
        "/invoices/INV-1005/information-reply",
        json={
            "actor": "Sam (AP)",
            "supplied_items": ["purchase_order"],
            "note": "PO attached",
        },
    )
    assert response.status_code == 200
    assert response.json()["information_request"]["reply_status"] == "answered"
    refused = client.post(
        "/invoices/INV-1005/information-reply",
        json={"actor": "Sam (AP)", "supplied_items": ["purchase_order"]},
    )
    assert refused.status_code == 422  # already answered
    unknown = client.post(
        "/invoices/INV-9999/information-reply",
        json={"actor": "Sam (AP)", "supplied_items": ["purchase_order"]},
    )
    assert unknown.status_code == 404
