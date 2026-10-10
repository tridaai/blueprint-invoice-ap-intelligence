"""The marker-risk worklist: creditor-completion coverage prices
the marker; this turns the same export receipts into the
supplier-onboarding queue — suppliers ranked by marked export
value, each with its marked invoices and last export date, and
suppliers the master now covers resolved off the list."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.schemas import (
    HandoffCreditorEntry,
    PaymentRun,
    PaymentRunLine,
    SupplierMasterRecord,
)
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore
from ap_agent.supplier_master import InMemorySupplierMaster
from ap_agent.worklist import marker_worklist

IBAN = "DE89370400440532013000"


def _run(run_id: str, exported_at: str, lines, entries, currency="USD") -> PaymentRun:
    return PaymentRun(
        run_id=run_id,
        currency=currency,
        status="confirmed",
        handoff_generated_at=exported_at,
        handoff_format="csv",
        lines=lines,
        invoice_count=len(lines),
        handoff_creditor_entries=entries,
    )


def _line(invoice_id: str, supplier: str, net: float) -> PaymentRunLine:
    return PaymentRunLine(
        invoice_id=invoice_id, supplier_name=supplier, gross_amount=net, net_amount=net
    )


def _marked(invoice_id: str, supplier: str) -> HandoffCreditorEntry:
    return HandoffCreditorEntry(
        invoice_id=invoice_id, supplier_name=supplier, source="at_import"
    )


def _completed(invoice_id: str, supplier: str) -> HandoffCreditorEntry:
    return HandoffCreditorEntry(
        invoice_id=invoice_id,
        supplier_name=supplier,
        source="supplier_master",
        account_kind="iban",
        account_label="iban ••••3000",
    )


def _runs() -> list[PaymentRun]:
    return [
        _run(
            "RUN-USD-A",
            "2026-10-07T09:00:00+00:00",
            [_line("INV-1", "Northbeam Office Supplies", 1156.40),
             _line("INV-2", "Cascade IT Hardware", 500.0)],
            [_marked("INV-1", "Northbeam Office Supplies"),
             _marked("INV-2", "Cascade IT Hardware")],
        ),
        _run(
            "RUN-USD-B",
            "2026-10-20T09:00:00+00:00",
            [_line("INV-3", "Northbeam Office Supplies", 2000.0),
             _line("INV-4", "Bluepine Catering Co", 300.0)],
            [_marked("INV-3", "Northbeam Office Supplies"),
             _completed("INV-4", "Bluepine Catering Co")],
        ),
    ]


def test_suppliers_rank_by_marked_value_with_their_invoices_and_last_export():
    worklist = marker_worklist(_runs())
    assert worklist.supplier_count == 2
    assert worklist.totals_by_currency == {"USD": 3656.40}
    northbeam, cascade = worklist.items  # biggest exposure first
    assert northbeam.supplier_name == "Northbeam Office Supplies"
    assert northbeam.marked_value == 3156.40
    assert northbeam.invoice_count == 2
    assert northbeam.invoice_ids == ["INV-1", "INV-3"]
    assert northbeam.export_count == 2
    assert northbeam.last_export_date == "2026-10-20"
    assert cascade.marked_value == 500.0 and cascade.invoice_count == 1
    # The completed supplier never appears: no marker, no work.
    assert all(i.supplier_name != "Bluepine Catering Co" for i in worklist.items)


def test_ranking_never_compares_values_across_currencies():
    eur = _run(
        "RUN-EUR-A",
        "2026-10-09T09:00:00+00:00",
        [_line("INV-9", "Alpen Logistics", 10.0)],
        [_marked("INV-9", "Alpen Logistics")],
        currency="EUR",
    )
    worklist = marker_worklist([*_runs(), eur])
    # Currency groups in order; the tiny EUR exposure is its own
    # group's top, not the list's bottom by raw number.
    assert [i.currency for i in worklist.items] == ["EUR", "USD", "USD"]
    assert worklist.totals_by_currency == {"EUR": 10.0, "USD": 3656.40}


def test_unexported_and_receiptless_runs_contribute_nothing():
    unexported = _run("RUN-USD-C", "", [], [])
    legacy = _run(
        "RUN-USD-D",
        "2026-09-01T09:00:00+00:00",
        [_line("INV-8", "Northbeam Office Supplies", 999.0)],
        [],  # exported before completion receipts existed
    )
    worklist = marker_worklist([unexported, legacy])
    assert worklist.items == []
    assert worklist.totals_by_currency == {}


def test_onboarded_suppliers_resolve_off_the_list_and_are_counted():
    worklist = marker_worklist(
        _runs(), onboarded_suppliers={"Northbeam Office Supplies"}
    )
    assert [i.supplier_name for i in worklist.items] == ["Cascade IT Hardware"]
    assert worklist.supplier_count == 1
    assert worklist.resolved_supplier_count == 1
    assert worklist.resolved_value_by_currency == {"USD": 3156.40}
    assert worklist.totals_by_currency == {"USD": 500.0}


def _service(master) -> InvoiceService:
    store = InMemoryStore()
    for run in _runs():
        store.save_payment_run(run)
    return InvoiceService(store=store, checkpointer=False, supplier_master=master)


def test_service_resolves_suppliers_the_master_now_covers_with_details():
    master = InMemorySupplierMaster(
        [SupplierMasterRecord(supplier_name="Northbeam Office Supplies", iban=IBAN)]
    )
    worklist = _service(master).marker_worklist()
    assert [i.supplier_name for i in worklist.items] == ["Cascade IT Hardware"]
    assert worklist.resolved_supplier_count == 1


def test_a_master_record_without_bank_details_resolves_nothing():
    # The completion rule is details, not existence: a name-only
    # record completes no creditor account, so the supplier is
    # still work.
    master = InMemorySupplierMaster(
        [SupplierMasterRecord(supplier_name="Northbeam Office Supplies")]
    )
    worklist = _service(master).marker_worklist()
    assert worklist.resolved_supplier_count == 0
    assert worklist.supplier_count == 2


def test_worklist_api():
    client = TestClient(create_app(_service(InMemorySupplierMaster())))
    resp = client.get("/suppliers/marker-worklist")
    assert resp.status_code == 200
    body = resp.json()
    assert body["supplier_count"] == 2
    assert body["items"][0]["supplier_name"] == "Northbeam Office Supplies"
    assert body["items"][0]["marked_value"] == 3156.40
    assert body["items"][0]["last_export_date"] == "2026-10-20"
