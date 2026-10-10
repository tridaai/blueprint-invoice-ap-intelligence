"""Discount drift at hand-off export: a run confirmed while a
member's early-payment discount was capturable can be exported
after the window closed. The export re-prices every discounted
member as of the export date — a drifted member is exported at
its full gross amount (the amount actually owed once the window
closes), and the drift is itemised on the run record, never
silently exported at the stale net figure."""

from __future__ import annotations

import csv
import io
from datetime import date

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.handoff import creditor_completions, render_handoff_csv
from ap_agent.paymentruns import discount_drift
from ap_agent.samples import sample_invoice_models
from ap_agent.supplier_master import get_supplier_master
from ap_agent.service import InvoiceService
from ap_agent.storage import InMemoryObjectStore
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}

# INV-1001: invoice date 2026-09-28, terms 2/10 net 30 on 1180.00 —
# discount deadline 2026-10-08, gross discount 23.60.
DEADLINE = "2026-10-08"
DISCOUNT = 23.60


@pytest.fixture()
def service() -> InvoiceService:
    """INV-1001 approved and in a run proposed on 2026-10-05, while
    its discount window (to 2026-10-08) was still open."""
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        object_store=InMemoryObjectStore(),
    )
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    svc.propose_payment_runs(window_days=7, as_of="2026-10-05")
    return svc


def _confirmed(svc: InvoiceService):
    run = svc.list_payment_runs()[0]
    svc.confirm_payment_run(run.run_id, actor="Tomas Reid")
    return svc.get_payment_run(run.run_id)


def _rows(data: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(data.decode("utf-8"))))


def test_export_inside_the_window_has_no_drift(service):
    run = _confirmed(service)
    exported = service.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-06"
    )
    assert exported.handoff_drift == []
    assert exported.handoff_drift_total == 0.0
    assert exported.total_discount == DISCOUNT
    assert exported.total_net == 1156.40
    rows = _rows(service.get_payment_run_handoff(run.run_id))
    assert rows[1][8] == "23.60" and rows[1][9] == "1156.40"


def test_export_after_the_window_flags_the_drift(service):
    run = _confirmed(service)
    exported = service.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-20"
    )
    assert len(exported.handoff_drift) == 1
    entry = exported.handoff_drift[0]
    assert entry.invoice_id == "INV-1001"
    assert entry.discount_deadline == DEADLINE
    assert entry.discount_at_proposal == DISCOUNT
    assert entry.discount_at_export == 0.0
    assert entry.discount_lost == DISCOUNT
    assert "window closed" in entry.reason
    assert exported.handoff_drift_total == DISCOUNT
    assert "Discount drift at export" in exported.note
    # The exported member is paid at full gross — the amount owed
    # once the window closed — and the file's totals say so.
    assert exported.total_discount == 0.0
    assert exported.total_net == 1180.00
    rows = _rows(service.get_payment_run_handoff(run.run_id))
    assert rows[1][7] == "1180.00"  # gross
    assert rows[1][8] == "0.00"  # no stale discount netted off
    assert rows[1][9] == "1180.00"  # net = gross
    totals = rows[2]
    assert totals[8] == "0.00" and totals[9] == "1180.00"
    # The stored bytes are exactly the adjusted run rendered, with
    # the creditor completion the export applied.
    completions, _ = creditor_completions(exported, get_supplier_master())
    assert service.get_payment_run_handoff(run.run_id) == render_handoff_csv(
        exported, creditor_accounts=completions
    )
    # The confirmation itself is untouched by the adjustment.
    assert exported.status == "confirmed"
    assert exported.confirmed_by == "Tomas Reid"


def test_reexport_after_drift_is_stable(service):
    run = _confirmed(service)
    first = service.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-20"
    )
    second = service.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-21"
    )
    assert first.handoff_sha256 == second.handoff_sha256
    assert len(second.handoff_drift) == 1  # not re-itemised or doubled
    assert second.handoff_drift_total == DISCOUNT


def test_drift_check_is_pure_over_the_run(service):
    run = _confirmed(service)
    records = {r.result.invoice_id: r for r in service._get_store().list_records()}
    adjusted, drift = discount_drift(run, records, as_of=date(2026, 10, 20))
    assert len(drift) == 1
    assert adjusted.lines[0].discount_amount == 0.0
    assert adjusted.total_net == 1180.00
    # The run it was given keeps its confirmed figures.
    assert run.lines[0].discount_amount == DISCOUNT
    assert run.total_net == 1156.40


def test_unverifiable_member_drifts_to_gross(service):
    run = _confirmed(service)
    adjusted, drift = discount_drift(run, {}, as_of=date(2026, 10, 6))
    assert len(drift) == 1
    assert "cannot be re-verified" in drift[0].reason
    assert adjusted.lines[0].net_amount == 1180.00


def test_mixed_run_only_the_drifted_member_moves():
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        object_store=InMemoryObjectStore(),
    )
    svc.analyze(SAMPLES["INV-1001"])
    svc.analyze(SAMPLES["INV-1003"])  # Harborline — no discount terms
    svc.approve("INV-1001", approver="Priya Nair")
    svc.approve("INV-1003", approver="Priya Nair")
    svc.propose_payment_runs(window_days=7, as_of="2026-10-05")
    runs = svc.list_payment_runs()
    assert len(runs) == 1 and runs[0].invoice_count == 2
    svc.confirm_payment_run(runs[0].run_id, actor="Tomas Reid")
    exported = svc.export_payment_run_handoff(
        runs[0].run_id, actor="Tomas Reid", as_of="2026-10-20"
    )
    assert [entry.invoice_id for entry in exported.handoff_drift] == ["INV-1001"]
    lines = {line.invoice_id: line for line in exported.lines}
    assert lines["INV-1001"].net_amount == 1180.00
    clean = lines["INV-1003"]
    assert clean.discount_amount == 0.0 and clean.net_amount == 708.00
    assert exported.total_gross == 1888.00
    assert exported.total_net == 1888.00


def test_handoff_endpoint_reports_the_drift(service):
    run = _confirmed(service)
    client = TestClient(create_app(service))
    exported = client.post(
        f"/payment-runs/{run.run_id}/handoff?as_of=2026-10-20",
        json={"actor": "Tomas Reid"},
    )
    assert exported.status_code == 200
    body = exported.json()
    assert body["handoff_drift_total"] == DISCOUNT
    assert body["handoff_drift"][0]["invoice_id"] == "INV-1001"
    bad = client.post(
        f"/payment-runs/{run.run_id}/handoff?as_of=not-a-date",
        json={"actor": "Tomas Reid"},
    )
    assert bad.status_code == 422
