"""Payment-run hand-off file: a confirmed run rendered as a
bank-ready CSV (layout v1) and stored through the object-store
port. The file is the routing adapter to the customer's banking /
ERP system — generating it changes no payment state."""

from __future__ import annotations

import csv
import hashlib
import io

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.handoff import (
    HANDOFF_COLUMNS,
    creditor_completions,
    handoff_key,
    render_handoff_csv,
)
from ap_agent.samples import sample_invoice_models
from ap_agent.supplier_master import get_supplier_master
from ap_agent.service import InvoiceService
from ap_agent.storage import InMemoryObjectStore
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


@pytest.fixture()
def service() -> InvoiceService:
    """INV-1001 approved and in a confirmed USD run."""
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        object_store=InMemoryObjectStore(),
    )
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    runs = svc.propose_payment_runs(window_days=7, as_of="2026-10-05")
    svc.confirm_payment_run(runs[0].run_id, actor="Tomas Reid")
    return svc


def _confirmed_run(svc: InvoiceService):
    runs = [r for r in svc.list_payment_runs() if r.status == "confirmed"]
    assert len(runs) == 1
    return runs[0]


def test_csv_layout_and_totals_row(service):
    run = _confirmed_run(service)
    data = render_handoff_csv(run)
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
    assert rows[0] == HANDOFF_COLUMNS
    line = rows[1]
    assert line[2] == "INV-1001" and line[3] == "Northbeam Office Supplies"
    assert line[7] == "1180.00"  # gross
    assert line[8] == "23.60"  # capturable discount (2/10 net 30, in window)
    assert line[9] == "1156.40"  # net
    totals = rows[2]
    assert totals[2] == "TOTAL"
    assert totals[7] == "1180.00" and totals[9] == "1156.40"
    assert len(rows) == 3


def test_export_stores_the_file_and_records_the_receipt(service):
    run = _confirmed_run(service)
    exported = service.export_payment_run_handoff(run.run_id, actor="Tomas Reid")
    assert exported.handoff_key == handoff_key(run.run_id)
    assert exported.handoff_generated_by == "Tomas Reid"
    assert exported.handoff_generated_at
    stored = service.get_payment_run_handoff(run.run_id)
    assert stored is not None
    assert exported.handoff_sha256 == hashlib.sha256(stored).hexdigest()
    # The export completed INV-1001's creditor account from the
    # (seeded sample) supplier master; the stored bytes are the run
    # rendered with those completions.
    completions, _ = creditor_completions(exported, get_supplier_master())
    assert stored == render_handoff_csv(exported, creditor_accounts=completions)


def test_export_is_deterministic_and_changes_no_payment_state(service):
    run = _confirmed_run(service)
    first = service.export_payment_run_handoff(run.run_id, actor="Tomas Reid")
    second = service.export_payment_run_handoff(run.run_id, actor="Tomas Reid")
    assert first.handoff_sha256 == second.handoff_sha256
    after = service.get("INV-1001")
    assert after is not None
    assert after.status == "approved"
    assert after.external_action_taken is False
    assert service.get_payment_run(run.run_id).status == "confirmed"


def test_export_requires_a_confirmed_run():
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        object_store=InMemoryObjectStore(),
    )
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="approved in test")
    run = svc.propose_payment_runs(as_of="2026-10-20")[0]
    with pytest.raises(ValueError, match="only a confirmed run"):
        svc.export_payment_run_handoff(run.run_id, actor="Tomas Reid")
    with pytest.raises(KeyError):
        svc.export_payment_run_handoff("RUN-NOPE", actor="Tomas Reid")


def test_export_requires_object_storage():
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="approved in test")
    run = svc.propose_payment_runs(as_of="2026-10-20")[0]
    svc.confirm_payment_run(run.run_id, actor="Tomas Reid")
    with pytest.raises(ValueError, match="Object storage is not configured"):
        svc.export_payment_run_handoff(run.run_id, actor="Tomas Reid")


def test_handoff_endpoints(service):
    run = _confirmed_run(service)
    client = TestClient(create_app(service))
    missing = client.get(f"/payment-runs/{run.run_id}/handoff")
    assert missing.status_code == 404
    exported = client.post(
        f"/payment-runs/{run.run_id}/handoff", json={"actor": "Tomas Reid"}
    )
    assert exported.status_code == 200
    assert exported.json()["handoff_key"] == handoff_key(run.run_id)
    download = client.get(f"/payment-runs/{run.run_id}/handoff")
    assert download.status_code == 200
    assert "text/csv" in download.headers["content-type"]
    assert download.content == service.get_payment_run_handoff(run.run_id)
