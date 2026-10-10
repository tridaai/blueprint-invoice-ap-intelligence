"""Supplier statement reconciliation: the statement is lined up
against the processed invoices in code — matched / amount differs /
never processed / missing from the statement — and a human confirms
the report. Reconciliation changes no invoice state."""

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
    # Northbeam: INV-1001 (NB-4471, 1180.00), INV-1008 (NB-4480, 1180.00).
    # Cascade: INV-1002 (CI-2210, 1274.40) — must not leak into a
    # Northbeam statement.
    for invoice_id in ("INV-1001", "INV-1008", "INV-1002"):
        svc.analyze(SAMPLES[invoice_id])
    return svc


def _statement(**overrides) -> dict:
    base = {
        "statement_id": "STMT-NB-2026-10",
        "supplier_name": "Northbeam Office Supplies",
        "statement_date": "2026-10-05",
        "currency": "USD",
        "lines": [
            {"invoice_number": "NB-4471", "amount": 1180.00, "date": "2026-09-28"},
            {"invoice_number": "NB-4480", "amount": 1100.00, "date": "2026-09-29"},
            {"invoice_number": "NB-9999", "amount": 75.00, "date": "2026-09-30"},
        ],
    }
    base.update(overrides)
    return base


def test_reconciliation_classifies_every_line(service):
    report = service.reconcile_statement(_statement())
    by_number = {line.invoice_number: line for line in report.lines}
    assert by_number["NB-4471"].status == "matched"
    assert by_number["NB-4471"].ap_invoice_id == "INV-1001"
    assert by_number["NB-4480"].status == "amount_differs"
    assert by_number["NB-4480"].difference == -80.00  # 1100 stated vs 1180 processed
    assert by_number["NB-9999"].status == "missing_from_ap"
    assert report.matched_count == 1
    assert report.differs_count == 1
    assert report.missing_from_ap_count == 1
    assert report.missing_from_statement_count == 0
    assert report.statement_total == 2355.00
    assert report.ap_total == 2360.00  # the two Northbeam invoices only
    # the summary names every exception
    assert "NB-4480" in report.summary and "NB-9999" in report.summary
    assert report.summary_source == "template"


def test_processed_invoice_missing_from_statement_is_reported(service):
    report = service.reconcile_statement(
        _statement(
            statement_id="STMT-NB-SHORT",
            lines=[{"invoice_number": "NB-4471", "amount": 1180.00}],
        )
    )
    assert report.missing_from_statement_count == 1
    missing = report.missing_from_statement[0]
    assert missing["invoice_number"] == "NB-4480"
    assert missing["invoice_id"] == "INV-1008"
    assert "NB-4480" in report.summary


def test_reconciliation_is_report_only(service):
    before = {iid: service.get(iid).status for iid in ("INV-1001", "INV-1008")}
    service.reconcile_statement(_statement())
    service.confirm_reconciliation("STMT-NB-2026-10", actor="Priya (AP lead)")
    after = {iid: service.get(iid).status for iid in ("INV-1001", "INV-1008")}
    assert before == after == {
        "INV-1001": "awaiting_approver",
        "INV-1008": "awaiting_approver",
    }


def test_confirm_records_the_human_and_refuses_twice(service):
    service.reconcile_statement(_statement())
    report = service.confirm_reconciliation("STMT-NB-2026-10", actor="Priya (AP lead)")
    assert report.confirmed_by == "Priya (AP lead)"
    assert report.confirmed_at
    fetched = service.get_reconciliation("STMT-NB-2026-10")
    assert fetched is not None and fetched.confirmed_by == "Priya (AP lead)"
    with pytest.raises(ValueError):
        service.confirm_reconciliation("STMT-NB-2026-10", actor="Someone Else")


def test_confirm_unknown_statement_raises(service):
    with pytest.raises(KeyError):
        service.confirm_reconciliation("STMT-NOPE", actor="Priya")


def test_reconciliation_endpoints():
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])
    client = TestClient(create_app(svc))
    response = client.post("/statements/reconcile", json=_statement())
    assert response.status_code == 200
    body = response.json()
    assert body["matched_count"] == 1
    assert body["confirmed_by"] is None

    missing = client.post(
        "/statements/STMT-NB-2026-10/confirm", json={"reason": "checked"}
    )
    assert missing.status_code == 422  # an actor is required

    confirmed = client.post(
        "/statements/STMT-NB-2026-10/confirm",
        json={"actor": "Priya (AP lead)", "reason": "exceptions noted"},
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["confirmed_by"] == "Priya (AP lead)"

    fetched = client.get("/statements/STMT-NB-2026-10")
    assert fetched.status_code == 200
    assert fetched.json()["confirmed_by"] == "Priya (AP lead)"
    assert client.get("/statements/STMT-UNKNOWN").status_code == 404
