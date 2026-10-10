"""Creditor-account completion at hand-off export: the pain.001 /
CSV creditor account is the one bank element the system may
honestly fill, because the governed supplier master is where
approved bank details live. Where — and only where — the master
holds bank details for a member's supplier, the export emits them
(IBAN where recorded); everywhere else the member keeps the
documented at-import marker, and the run's export receipt records
which members went which way, with masked labels only."""

from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET

import pytest

from ap_agent.handoff import (
    CREDITOR_ACCOUNT_MARKER,
    PAIN001_NAMESPACE,
    creditor_completions,
    render_handoff_csv,
    render_handoff_pain001,
)
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import PaymentRun, PaymentRunLine, SupplierMasterRecord
from ap_agent.service import InvoiceService
from ap_agent.storage import InMemoryObjectStore
from ap_agent.store import InMemoryStore
from ap_agent.supplier_master import InMemorySupplierMaster

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
NS = {"p": PAIN001_NAMESPACE}
IBAN = "DE89370400440532013000"


def _run() -> PaymentRun:
    return PaymentRun(
        run_id="RUN-USD-2026-10-05-2026-10-11",
        currency="USD",
        window_start="2026-10-05",
        window_end="2026-10-11",
        status="confirmed",
        lines=[
            PaymentRunLine(
                invoice_id="INV-1001",
                supplier_name="Northbeam Office Supplies",
                invoice_number="NB-4471",
                gross_amount=1180.0,
                net_amount=1180.0,
            ),
            PaymentRunLine(
                invoice_id="INV-1007",
                supplier_name="Alpen Werkzeug GmbH",
                invoice_number="AW-77",
                gross_amount=357.0,
                net_amount=357.0,
            ),
        ],
        invoice_count=2,
        total_gross=1537.0,
        total_net=1537.0,
    )


def _rows(data: bytes):
    return list(csv.reader(io.StringIO(data.decode("utf-8"))))


def test_completion_only_where_the_master_holds_bank_details():
    master = InMemorySupplierMaster(
        [
            SupplierMasterRecord(
                supplier_name="Northbeam Office Supplies", iban=IBAN
            ),
            # A record exists for Alpen — but it holds no bank
            # details, so it completes nothing.
            SupplierMasterRecord(
                supplier_name="Alpen Werkzeug GmbH",
                payment_terms="net 30",
                tax_id="DE-TAX-1",
            ),
        ]
    )
    completions, entries = creditor_completions(_run(), master)
    assert set(completions) == {"INV-1001"}
    assert completions["INV-1001"]["iban"] == IBAN
    by_id = {e.invoice_id: e for e in entries}
    assert by_id["INV-1001"].source == "supplier_master"
    assert by_id["INV-1001"].account_kind == "iban"
    assert by_id["INV-1001"].account_label == "iban ••••3000"
    assert by_id["INV-1007"].source == "at_import"
    assert by_id["INV-1007"].account_label == ""


def test_csv_carries_the_completed_account_and_its_source():
    master = InMemorySupplierMaster(
        [
            SupplierMasterRecord(
                supplier_name="Northbeam Office Supplies",
                account_number="1029384756",
                sort_code="30-12-44",
            )
        ]
    )
    completions, _ = creditor_completions(_run(), master)
    rows = _rows(render_handoff_csv(_run(), creditor_accounts=completions))
    header = rows[0]
    account_col = header.index("creditor_account")
    source_col = header.index("creditor_account_source")
    assert rows[1][account_col] == "1029384756"
    assert rows[1][source_col] == "supplier_master"
    assert rows[2][account_col] == CREDITOR_ACCOUNT_MARKER
    assert rows[2][source_col] == "at_import"
    assert rows[3][account_col] == ""  # totals row stays blank


def test_pain001_emits_iban_or_account_plus_sort_code():
    master = InMemorySupplierMaster(
        [
            SupplierMasterRecord(
                supplier_name="Northbeam Office Supplies", iban=IBAN
            ),
            SupplierMasterRecord(
                supplier_name="Alpen Werkzeug GmbH",
                account_number="77441122",
                sort_code="40-02-11",
            ),
        ]
    )
    completions, _ = creditor_completions(_run(), master)
    root = ET.fromstring(render_handoff_pain001(_run(), creditor_accounts=completions))
    txs = root.findall(".//p:CdtTrfTxInf", NS)
    # IBAN member: the IBAN element, no marker, no creditor agent.
    assert txs[0].findtext(".//p:CdtrAcct/p:Id/p:IBAN", namespaces=NS) == IBAN
    assert txs[0].find(".//p:CdtrAgt", NS) is None
    # Account member: Othr id + the sort code as the creditor agent.
    assert txs[1].findtext(".//p:CdtrAcct/p:Id/p:Othr/p:Id", namespaces=NS) == "77441122"
    assert (
        txs[1].findtext(".//p:CdtrAgt/p:FinInstnId/p:Othr/p:Id", namespaces=NS)
        == "40-02-11"
    )
    text = ET.tostring(root, encoding="unicode")
    assert CREDITOR_ACCOUNT_MARKER not in text
    # The debtor side is never completed from master data.
    assert "DEBTOR-ACCOUNT-AT-IMPORT" in text


def test_no_master_coverage_keeps_every_marker():
    completions, entries = creditor_completions(_run(), InMemorySupplierMaster())
    assert completions == {}
    assert all(e.source == "at_import" for e in entries)
    text = ET.tostring(
        ET.fromstring(render_handoff_pain001(_run())), encoding="unicode"
    )
    assert text.count(CREDITOR_ACCOUNT_MARKER) == 2


@pytest.fixture()
def service_with_iban_master() -> InvoiceService:
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        object_store=InMemoryObjectStore(),
        supplier_master=InMemorySupplierMaster(
            [SupplierMasterRecord(supplier_name="Northbeam Office Supplies", iban=IBAN)]
        ),
    )
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    run = svc.propose_payment_runs(window_days=7, as_of="2026-10-05")[0]
    svc.confirm_payment_run(run.run_id, actor="Tomas Reid")
    return svc


def test_export_receipt_records_completed_vs_marked(service_with_iban_master):
    svc = service_with_iban_master
    run = svc.list_payment_runs()[0]
    exported = svc.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-06", format="pain001"
    )
    entries = exported.handoff_creditor_entries
    assert len(entries) == 1
    entry = entries[0]
    assert entry.invoice_id == "INV-1001"
    assert entry.source == "supplier_master" and entry.account_kind == "iban"
    assert entry.account_label == "iban ••••3000"
    # The full IBAN is in the file — and nowhere on the run record.
    stored = svc.get_payment_run_handoff(run.run_id)
    assert IBAN.encode() in stored
    assert IBAN not in exported.model_dump_json()


def test_export_with_an_empty_master_marks_every_member():
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        object_store=InMemoryObjectStore(),
        supplier_master=InMemorySupplierMaster(),
    )
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    run = svc.propose_payment_runs(window_days=7, as_of="2026-10-05")[0]
    svc.confirm_payment_run(run.run_id, actor="Tomas Reid")
    exported = svc.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-06"
    )
    assert exported.handoff_creditor_entries[0].source == "at_import"
    rows = _rows(svc.get_payment_run_handoff(run.run_id))
    assert rows[1][-2] == CREDITOR_ACCOUNT_MARKER
    assert rows[1][-1] == "at_import"
