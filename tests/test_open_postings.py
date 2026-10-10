"""Approved-not-posted reconciliation: the posting-receipt loop
closes only when a receipt arrives. Approved invoices whose
receipt never comes — or whose only receipt is a failure — are
the loop's open items, reported once they outlast the grace
window (POSTING_RECEIPT_GRACE_DAYS, default 7)."""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.matching import _parse_date
from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


@pytest.fixture()
def service() -> InvoiceService:
    return InvoiceService(store=InMemoryStore(), checkpointer=False)


def _approved(service, invoice_id="INV-1001"):
    service.analyze(SAMPLES[invoice_id])
    return service.approve(invoice_id, approver="Priya Nair", reason="clean match")


def _plus(iso: str, days: int) -> str:
    return (_parse_date(iso) + timedelta(days=days)).isoformat()


def test_approve_and_reject_stamp_the_decision_time(service):
    result = _approved(service)
    assert result.decided_at  # the report's age anchor
    service.analyze(SAMPLES["INV-1005"])
    rejected = service.reject("INV-1005", reviewer="Priya Nair", reason="no PO")
    assert rejected.decided_at


def test_open_item_appears_only_after_the_grace_window(service):
    result = _approved(service)
    inside = service.approved_not_posted_report(as_of=_plus(result.decided_at, 7))
    assert inside.count == 0  # at the grace boundary: still in flight
    outside = service.approved_not_posted_report(as_of=_plus(result.decided_at, 8))
    assert outside.count == 1
    assert outside.grace_days == 7
    item = outside.items[0]
    assert item.invoice_id == "INV-1001"
    assert item.age_days == 8
    assert item.age_basis == "decision"
    assert item.receipt_state == "none"
    assert item.decided_by == "Priya Nair"
    assert outside.totals_by_currency == {"USD": 1180.00}


def test_a_posted_receipt_closes_the_item(service):
    result = _approved(service)
    service.record_posting_receipt(
        "INV-1001",
        status="posted",
        actor="Tomas Reid",
        external_reference="ERP-DOC-1001",
        idempotency_key="post-INV-1001",
    )
    report = service.approved_not_posted_report(as_of=_plus(result.decided_at, 60))
    assert report.count == 0


def test_a_failed_receipt_keeps_the_item_open_with_its_reason(service):
    result = _approved(service)
    service.record_posting_receipt(
        "INV-1001",
        status="failed",
        actor="Tomas Reid",
        detail="ERP refused: cost centre closed",
    )
    report = service.approved_not_posted_report(as_of=_plus(result.decided_at, 30))
    assert report.count == 1
    item = report.items[0]
    assert item.receipt_state == "failed"
    assert item.receipt_detail == "ERP refused: cost centre closed"
    # A later successful posting supersedes the failure and closes it.
    service.record_posting_receipt(
        "INV-1001",
        status="posted",
        actor="Tomas Reid",
        external_reference="ERP-DOC-1001",
        idempotency_key="post-INV-1001",
    )
    assert (
        service.approved_not_posted_report(as_of=_plus(result.decided_at, 30)).count
        == 0
    )


def test_only_approved_invoices_are_open_items(service):
    _approved(service)
    service.analyze(SAMPLES["INV-1005"])  # stays awaiting
    service.analyze(SAMPLES["INV-1012"])
    service.reject("INV-1012", reviewer="Priya Nair", reason="not ours")
    report = service.approved_not_posted_report(as_of="2027-06-01")
    assert [item.invoice_id for item in report.items] == ["INV-1001"]


def test_legacy_records_age_from_the_submission(service):
    result = _approved(service)
    record = service._get_store().get("INV-1001")
    record.result.decided_at = None  # decided before the stamp existed
    service._get_store().save(record)
    report = service.approved_not_posted_report(as_of="2026-10-20")
    assert report.count == 1
    item = report.items[0]
    assert item.age_basis == "submission"
    assert item.age_days == 21  # submitted 2026-09-29


def test_grace_window_override_and_configuration(service, monkeypatch):
    result = _approved(service)
    report = service.approved_not_posted_report(
        as_of=_plus(result.decided_at, 3), grace_days=2
    )
    assert report.count == 1 and report.grace_days == 2
    monkeypatch.setenv("POSTING_RECEIPT_GRACE_DAYS", "45")
    report = service.approved_not_posted_report(as_of=_plus(result.decided_at, 30))
    assert report.count == 0 and report.grace_days == 45


def test_report_endpoint_and_metrics(service):
    result = _approved(service)
    client = TestClient(create_app(service))
    body = client.get(
        f"/reconciliation/approved-not-posted?as_of={_plus(result.decided_at, 10)}"
    ).json()
    assert body["count"] == 1
    assert body["items"][0]["invoice_id"] == "INV-1001"
    bad = client.get("/reconciliation/approved-not-posted?as_of=nope")
    assert bad.status_code == 422
    # /metrics counts the same open items as of today: fresh
    # approval — none yet; backdated decision — one.
    assert client.get("/metrics").json()["posting_open_items"] == {
        "count": 0,
        "grace_days": 7,
        "escalated_count": 0,
        "escalation_days": 21,
    }
    record = service._get_store().get("INV-1001")
    record.result.decided_at = _plus(result.decided_at, -30) + "T00:00:00+00:00"
    service._get_store().save(record)
    assert client.get("/metrics").json()["posting_open_items"]["count"] == 1
