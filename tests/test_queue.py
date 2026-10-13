"""The AP aging queue: awaiting invoices aged by due date, overdue
first, with the triage mix — a pure read model over the store
(queue.py + GET /queue)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


@pytest.fixture()
def service() -> InvoiceService:
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    for invoice_id in ("INV-1001", "INV-1005", "INV-1006", "INV-1008"):
        svc.analyze(SAMPLES[invoice_id])
    svc.approve("INV-1001", approver="Priya Nair", reason="approved in test")  # decided: leaves the queue
    return svc


def test_queue_holds_only_awaiting_invoices(service):
    view = service.queue_view(as_of="2026-11-05")
    assert view.count == 3
    assert {item.invoice_id for item in view.items} == {
        "INV-1005",
        "INV-1006",
        "INV-1008",
    }


def test_queue_orders_most_overdue_first_and_buckets(service):
    view = service.queue_view(as_of="2026-11-05")
    # Due dates: INV-1006 10-24 (12d), INV-1005 10-25 (11d), INV-1008 10-29 (7d).
    assert [item.invoice_id for item in view.items] == [
        "INV-1006",
        "INV-1005",
        "INV-1008",
    ]
    assert [item.days_overdue for item in view.items] == [12, 11, 7]
    assert view.items[0].bucket == "overdue_8_30"
    assert view.items[2].bucket == "overdue_1_7"
    assert view.overdue_count == 3
    assert view.buckets == {"overdue_8_30": 2, "overdue_1_7": 1}


def test_queue_carries_triage_mix_and_totals(service):
    view = service.queue_view(as_of="2026-11-05")
    assert view.triage_mix == {
        "missing_po": 1,
        "missing_receipt": 1,
        "price_variance": 1,
    }
    assert view.totals_by_currency == {"USD": 4602.0}
    by_id = {item.invoice_id: item for item in view.items}
    assert by_id["INV-1005"].outcome == "missing_po"
    assert by_id["INV-1005"].age_days is not None


def test_queue_before_due_dates_ages_differently(service):
    view = service.queue_view(as_of="2026-10-26")
    by_id = {item.invoice_id: item for item in view.items}
    assert by_id["INV-1006"].days_overdue == 2
    assert by_id["INV-1006"].bucket == "overdue_1_7"
    assert by_id["INV-1008"].days_overdue == -3
    assert by_id["INV-1008"].bucket == "due_within_7"
    assert view.overdue_count == 2


def test_queue_surfaces_the_discount_deadline():
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])  # 2/10 net 30, deadline 2026-10-08
    view = svc.queue_view(as_of="2026-10-06")
    assert view.count == 1
    item = view.items[0]
    assert item.discount_deadline == "2026-10-08"
    assert item.discount_days_left == 2
    assert item.bucket == "due_later"  # due 2026-10-28, 22 days out


def test_queue_is_a_pure_read(service):
    before = service.get("INV-1005")
    service.queue_view(as_of="2026-11-05")
    after = service.get("INV-1005")
    assert before is not None and after is not None
    assert after.status == before.status == "awaiting_approver"


def test_queue_endpoint():
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1006"])
    client = TestClient(create_app(svc))
    response = client.get("/queue", params={"as_of": "2026-11-05"})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["items"][0]["invoice_id"] == "INV-1006"
    assert body["items"][0]["bucket"] == "overdue_8_30"
    assert body["triage_mix"] == {"missing_receipt": 1}
