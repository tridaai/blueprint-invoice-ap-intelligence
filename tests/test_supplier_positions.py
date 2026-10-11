"""Per-supplier open-position roll-up: one row per supplier per
currency, summed from the same payable positions the per-invoice
endpoint serves — pinned against that endpoint — with the dispute
figure from the supplier's latest reconciliation. Read-only."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import (
    sample_credit_note_models,
    sample_invoice_models,
    sample_statement_models,
)
from ap_agent.schemas import SupplierStatement
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
NOTES = {n.credit_note_id: n for n in sample_credit_note_models()}
STATEMENTS = {s.statement_id: s for s in sample_statement_models()}
ANALYSED = ("INV-1001", "INV-1008", "INV-1002", "INV-1007")


@pytest.fixture()
def service() -> InvoiceService:
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    # Northbeam INV-1001 (NB-4471, 1180.00) + INV-1008 (NB-4480, 1180.00);
    # Cascade INV-1002 (CI-2210, 1274.40); Alpen INV-1007 (AW-77, EUR 357.00).
    for invoice_id in ANALYSED:
        svc.analyze(SAMPLES[invoice_id])
    return svc


def _rows(roll):
    return {(r.supplier_name, r.currency): r for r in roll.suppliers}


def test_rollup_rows_carry_the_position_figures(service):
    # CN-3001 settles Cascade's computed variance (94.40) in full.
    service.apply_credit_note(NOTES["CN-3001"], actor="Priya (AP lead)")
    service.reconcile_statement(STATEMENTS["STMT-NB-2026-10"])
    roll = service.supplier_positions(as_of="2026-10-05")
    rows = _rows(roll)

    northbeam = rows[("Northbeam Office Supplies", "USD")]
    assert northbeam.invoice_count == 2
    assert northbeam.open_invoice_count == 2
    assert northbeam.gross_invoiced == 2360.00
    assert northbeam.credits_applied == 0.00
    assert northbeam.open_amount == 2360.00
    # In dispute per STMT-NB: |1100 − 1180| on NB-4480 + 75.00 never processed.
    assert northbeam.disputed_amount == 155.00
    assert northbeam.disputed_invoice_numbers == ["NB-4480", "NB-9999"]
    assert northbeam.dispute_statement_id == "STMT-NB-2026-10"
    assert northbeam.oldest_open_invoice_id == "INV-1001"
    assert northbeam.oldest_open_age_days == 7  # invoiced 2026-09-28

    cascade = rows[("Cascade IT Hardware", "USD")]
    assert cascade.gross_invoiced == 1274.40
    assert cascade.credits_applied == 94.40
    assert cascade.open_amount == 1180.00
    assert cascade.disputed_amount == 0.00

    alpen = rows[("Alpen Werkzeug GmbH", "EUR")]
    assert alpen.gross_invoiced == 357.00
    assert alpen.open_amount == 357.00
    assert alpen.oldest_open_age_days == 12  # invoiced 2026-09-23

    assert roll.supplier_count == 3
    # Totals are per currency — never a cross-currency sum.
    assert set(roll.totals_by_currency) == {"USD", "EUR"}
    assert roll.totals_by_currency["USD"]["gross_invoiced"] == 3634.40
    assert roll.totals_by_currency["USD"]["open_amount"] == 3540.00
    assert roll.totals_by_currency["USD"]["disputed_amount"] == 155.00
    assert roll.totals_by_currency["EUR"]["open_amount"] == 357.00


def test_rollup_equals_the_sum_of_its_payable_positions(service):
    service.apply_credit_note(NOTES["CN-3001"], actor="Priya (AP lead)")
    roll = service.supplier_positions(as_of="2026-10-05")
    parts: dict[tuple[str, str], dict[str, float]] = {}
    for invoice_id in ANALYSED:
        position = service.payable_position(invoice_id)
        bucket = parts.setdefault(
            (position.supplier_name, position.currency),
            {"gross": 0.0, "credits": 0.0, "open": 0.0},
        )
        bucket["gross"] += position.gross_amount
        bucket["credits"] += position.total_credits
        bucket["open"] += position.open_amount
    for row in roll.suppliers:
        bucket = parts[(row.supplier_name, row.currency)]
        assert row.gross_invoiced == round(bucket["gross"], 2)
        assert row.credits_applied == round(bucket["credits"], 2)
        assert row.open_amount == round(bucket["open"], 2)


def test_the_latest_reconciliation_speaks_for_the_dispute(service):
    service.reconcile_statement(STATEMENTS["STMT-NB-2026-10"])
    payload = STATEMENTS["STMT-NB-2026-10"].model_dump()
    payload.update(
        {
            "statement_id": "STMT-NB-2026-11",
            "statement_date": "2026-11-05",
            "claimed_total": 2360.00,
            "lines": [
                {"invoice_number": "NB-4471", "amount": 1180.00, "date": "2026-09-28"},
                {"invoice_number": "NB-4480", "amount": 1180.00, "date": "2026-09-29"},
            ],
        }
    )
    service.reconcile_statement(SupplierStatement.model_validate(payload))
    northbeam = _rows(service.supplier_positions(as_of="2026-11-06"))[
        ("Northbeam Office Supplies", "USD")
    ]
    assert northbeam.disputed_amount == 0.00
    assert northbeam.disputed_invoice_numbers == []
    assert northbeam.dispute_statement_id == "STMT-NB-2026-11"


def test_a_refused_document_report_contributes_no_dispute(service):
    service.reconcile_statement(STATEMENTS["STMT-NB-2026-10"])
    # The Northbeam invoice document is refused as not-a-statement;
    # its stored record names Northbeam (the document's fields do)
    # but a refusal must never move the dispute figure.
    service.reconcile_statement_document(SAMPLES["INV-1001"].documents[0])
    northbeam = _rows(service.supplier_positions(as_of="2026-10-05"))[
        ("Northbeam Office Supplies", "USD")
    ]
    assert northbeam.disputed_amount == 155.00
    assert northbeam.dispute_statement_id == "STMT-NB-2026-10"


def test_empty_store_rolls_up_to_nothing():
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    roll = svc.supplier_positions(as_of="2026-10-05")
    assert roll.suppliers == []
    assert roll.supplier_count == 0
    assert roll.totals_by_currency == {}


def test_rollup_is_read_only(service):
    before = {
        iid: service.get(iid).status for iid in ANALYSED
    }
    applications_before = len(service.list_credit_applications())
    service.supplier_positions(as_of="2026-10-05")
    service.supplier_positions()
    assert {iid: service.get(iid).status for iid in ANALYSED} == before
    assert len(service.list_credit_applications()) == applications_before


def test_positions_endpoint():
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])
    client = TestClient(create_app(svc))
    response = client.get("/suppliers/positions", params={"as_of": "2026-10-05"})
    assert response.status_code == 200
    body = response.json()
    assert body["as_of"] == "2026-10-05"
    assert body["supplier_count"] == 1
    row = body["suppliers"][0]
    assert row["supplier_name"] == "Northbeam Office Supplies"
    assert row["open_amount"] == 1180.00
    assert row["oldest_open_age_days"] == 7
