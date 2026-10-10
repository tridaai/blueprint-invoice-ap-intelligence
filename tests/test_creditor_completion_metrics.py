"""Creditor-completion coverage in /metrics: master coverage
counts records, but what reaches the bank is exports. This metric
reads each exported run's completion receipt and reports — per
currency and per hand-off format — the share of exported members
and of exported value whose creditor account was completed from
the governed master, versus the share that still carried the
at-import marker. The marker is the residual import risk; this
prices it."""

from __future__ import annotations

from ap_agent.handoff import creditor_completions
from ap_agent.paymentruns import creditor_completion_metrics, handoff_metrics
from ap_agent.schemas import (
    HandoffCreditorEntry,
    PaymentRun,
    PaymentRunLine,
    SupplierMasterRecord,
)
from ap_agent.service import InvoiceService
from ap_agent.storage import InMemoryObjectStore
from ap_agent.store import InMemoryStore
from ap_agent.supplier_master import InMemorySupplierMaster

IBAN = "DE89370400440532013000"


def _run(**overrides) -> PaymentRun:
    base = {
        "run_id": "RUN-USD-2026-10-05-2026-10-11",
        "currency": "USD",
        "status": "confirmed",
        "handoff_generated_at": "2026-10-07T09:00:00+00:00",
        "handoff_format": "csv",
        "lines": [
            PaymentRunLine(invoice_id="INV-1", supplier_name="Northbeam Office Supplies", gross_amount=1180.0, net_amount=1156.40),
            PaymentRunLine(invoice_id="INV-2", supplier_name="Cascade IT Hardware", gross_amount=500.0, net_amount=500.0),
        ],
        "invoice_count": 2,
        "handoff_creditor_entries": [
            HandoffCreditorEntry(invoice_id="INV-1", supplier_name="Northbeam Office Supplies", source="supplier_master", account_kind="iban", account_label="iban ••••3000"),
            HandoffCreditorEntry(invoice_id="INV-2", supplier_name="Cascade IT Hardware", source="at_import"),
        ],
    }
    base.update(overrides)
    return PaymentRun(**base)


def test_coverage_counts_members_and_value_per_currency_format():
    coverage = creditor_completion_metrics([_run()])
    assert coverage["exports_total"] == 1
    assert coverage["exports_with_receipt"] == 1
    assert coverage["exports_without_receipt"] == 0
    overall = coverage["overall"]
    assert overall["members"] == 2
    assert overall["members_completed"] == 1
    assert overall["members_marked"] == 1
    # Value follows the export-time net figures: the completed
    # member is worth 1156.40 of the 1656.40 exported.
    assert overall["value_total"] == 1656.40
    assert overall["value_completed"] == 1156.40
    assert overall["value_marked"] == 500.0
    assert overall["member_completion_share"] == 0.5
    assert overall["value_completion_share"] == round(1156.40 / 1656.40, 4)
    groups = coverage["by_currency_format"]
    assert [(g["currency"], g["format"]) for g in groups] == [("USD", "csv")]
    assert groups[0]["members_completed"] == 1


def test_formats_and_currencies_split_and_receiptless_exports_show():
    pain = _run(
        run_id="RUN-EUR-X",
        currency="EUR",
        handoff_format="pain001",
        lines=[
            PaymentRunLine(invoice_id="INV-9", supplier_name="Alpen Werkzeug GmbH", gross_amount=357.0, net_amount=357.0)
        ],
        invoice_count=1,
        handoff_creditor_entries=[
            HandoffCreditorEntry(invoice_id="INV-9", supplier_name="Alpen Werkzeug GmbH", source="supplier_master", account_kind="account", account_label="account ••••7788")
        ],
    )
    # An export from before completion receipts existed: it counts
    # as an export, and its absence from the coverage is visible
    # rather than silently shrinking the denominator.
    legacy = _run(run_id="RUN-USD-LEGACY", handoff_creditor_entries=[])
    # Never exported: no contribution at all.
    proposed = _run(run_id="RUN-USD-PROPOSED", handoff_generated_at=None, handoff_format="")
    coverage = creditor_completion_metrics([_run(), pain, legacy, proposed])
    assert coverage["exports_total"] == 3
    assert coverage["exports_with_receipt"] == 2
    assert coverage["exports_without_receipt"] == 1
    keys = [(g["currency"], g["format"]) for g in coverage["by_currency_format"]]
    assert keys == [("EUR", "pain001"), ("USD", "csv")]
    eur = coverage["by_currency_format"][0]
    assert eur["member_completion_share"] == 1.0
    assert eur["value_completion_share"] == 1.0
    assert coverage["overall"]["members"] == 3
    assert coverage["overall"]["members_completed"] == 2


def test_empty_history_reports_null_shares_not_zero():
    coverage = creditor_completion_metrics([])
    assert coverage["exports_total"] == 0
    assert coverage["overall"]["members"] == 0
    assert coverage["overall"]["member_completion_share"] is None
    assert coverage["overall"]["value_completion_share"] is None
    assert coverage["by_currency_format"] == []


def test_service_metrics_score_a_real_export_end_to_end():
    master = InMemorySupplierMaster(
        [SupplierMasterRecord(supplier_name="Northbeam Office Supplies", iban=IBAN)]
    )
    store = InMemoryStore()
    svc = InvoiceService(
        store=store,
        checkpointer=False,
        object_store=InMemoryObjectStore(),
        supplier_master=master,
    )
    from ap_agent.samples import sample_invoice_models

    samples = {s.invoice_id: s for s in sample_invoice_models()}
    svc.analyze(samples["INV-1001"])  # Northbeam — master holds an IBAN
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    svc.analyze(samples["INV-1002"])  # a supplier with no master record
    svc.approve("INV-1002", approver="Priya Nair", reason="clean match")
    run = svc.propose_payment_runs(window_days=7, as_of="2026-10-05")[0]
    svc.confirm_payment_run(run.run_id, actor="Tomas Reid")
    svc.export_payment_run_handoff(run.run_id, actor="Tomas Reid", as_of="2026-10-06")
    coverage = svc.metrics_snapshot()["payment_runs"]["creditor_completion"]
    assert coverage["exports_with_receipt"] == 1
    assert coverage["overall"]["members"] == 2
    assert coverage["overall"]["members_completed"] == 1
    assert coverage["overall"]["members_marked"] == 1
    # The pure function and the service agree (same receipts).
    stored = store.get_payment_run(run.run_id)
    _, entries = creditor_completions(stored, master)
    assert len(entries) == 2
    assert handoff_metrics([stored])["creditor_completion"]["overall"]["members_completed"] == 1
