"""Payment-run hand-off, ISO 20022 pain.001 format: the second
serializer behind the hand-off seam. Same gates as the CSV
(confirmed run, four-eyes second confirmation, export-time
discount re-verification with itemised drift); the run record
notes which format was exported, and a golden file pins the
renderer's exact bytes for a fixed run."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.handoff import (
    PAIN001_NAMESPACE,
    creditor_completions,
    handoff_key,
    render_handoff,
    render_handoff_pain001,
)
from ap_agent.supplier_master import get_supplier_master
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import PaymentRun, PaymentRunLine
from ap_agent.service import InvoiceService
from ap_agent.storage import InMemoryObjectStore
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
GOLDEN = Path(__file__).parent / "golden" / "handoff_pain001.xml"
NS = {"p": PAIN001_NAMESPACE}


def _fixed_run() -> PaymentRun:
    return PaymentRun(
        run_id="RUN-USD-2026-10-05-2026-10-11",
        currency="USD",
        window_start="2026-10-05",
        window_end="2026-10-11",
        window_days=7,
        status="confirmed",
        lines=[
            PaymentRunLine(
                invoice_id="INV-1001",
                supplier_name="Northbeam Office Supplies",
                invoice_number="NB-4471",
                due_date="2026-10-20",
                gross_amount=1180.00,
                discount_amount=23.60,
                discount_deadline="2026-10-15",
                net_amount=1156.40,
            ),
            PaymentRunLine(
                invoice_id="INV-1008",
                supplier_name="Northbeam Office Supplies",
                invoice_number="NB-4480",
                due_date="2026-10-21",
                gross_amount=1180.00,
                discount_amount=0.0,
                discount_deadline="",
                net_amount=1180.00,
            ),
        ],
        invoice_count=2,
        total_gross=2360.00,
        total_discount=23.60,
        total_net=2336.40,
        proposed_at="2026-10-05T09:00:00+00:00",
        confirmed_by="Tomas Reid",
        confirmed_at="2026-10-05T09:30:00+00:00",
    )


def test_pain001_matches_the_golden_file():
    assert render_handoff_pain001(_fixed_run()) == GOLDEN.read_bytes()


def test_pain001_structure_and_control_sums():
    root = ET.fromstring(render_handoff_pain001(_fixed_run()))
    assert root.tag == f"{{{PAIN001_NAMESPACE}}}Document"
    header = root.find("./p:CstmrCdtTrfInitn/p:GrpHdr", NS)
    assert header.findtext("p:MsgId", namespaces=NS) == "RUN-USD-2026-10-05-2026-10-11"
    assert header.findtext("p:NbOfTxs", namespaces=NS) == "2"
    assert header.findtext("p:CtrlSum", namespaces=NS) == "2336.40"
    txs = root.findall(".//p:CdtTrfTxInf", NS)
    assert len(txs) == 2
    amounts = [
        (tx.findtext(".//p:EndToEndId", namespaces=NS),
         tx.find(".//p:InstdAmt", NS).text,
         tx.find(".//p:InstdAmt", NS).get("Ccy"),
         tx.findtext(".//p:Ustrd", namespaces=NS))
        for tx in txs
    ]
    assert amounts == [
        ("INV-1001", "1156.40", "USD", "Invoice NB-4471 (INV-1001)"),
        ("INV-1008", "1180.00", "USD", "Invoice NB-4480 (INV-1008)"),
    ]
    # The control sum is the sum of the instructed amounts.
    assert sum(float(a[1]) for a in amounts) == 2336.40
    # Bank-account elements carry the documented completion markers.
    text = ET.tostring(root, encoding="unicode")
    assert "DEBTOR-ACCOUNT-AT-IMPORT" in text
    assert "SUPPLIER-MASTER-AT-IMPORT" in text


@pytest.fixture()
def service() -> InvoiceService:
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


def _confirmed_run(svc: InvoiceService) -> PaymentRun:
    runs = [r for r in svc.list_payment_runs() if r.status == "confirmed"]
    assert len(runs) == 1
    return runs[0]


def test_export_pain001_stores_xml_and_records_the_format(service):
    run = _confirmed_run(service)
    # Exported while the member's discount window is still open —
    # the instructed amount is the net figure.
    exported = service.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-06", format="pain001"
    )
    assert exported.handoff_format == "pain001"
    assert exported.handoff_key == handoff_key(run.run_id, "pain001")
    assert exported.handoff_key.endswith("handoff.pain001.xml")
    stored = service.get_payment_run_handoff(run.run_id)
    assert stored is not None
    assert exported.handoff_sha256 == hashlib.sha256(stored).hexdigest()
    completions, _ = creditor_completions(exported, get_supplier_master())
    assert stored == render_handoff(
        exported, "pain001", creditor_accounts=completions
    )
    root = ET.fromstring(stored)
    amount = root.find(".//p:InstdAmt", NS)
    assert amount.text == "1156.40"  # net of the capturable discount


def test_pain001_export_applies_the_same_drift_gate_as_csv(service):
    run = _confirmed_run(service)
    # Exported after the member's discount deadline: the drift gate
    # re-prices the member to full gross and itemises the loss,
    # exactly as the CSV export does.
    exported = service.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-20", format="pain001"
    )
    assert exported.handoff_drift_total == 23.60
    assert [d.invoice_id for d in exported.handoff_drift] == ["INV-1001"]
    stored = service.get_payment_run_handoff(run.run_id)
    root = ET.fromstring(stored)
    amount = root.find(".//p:InstdAmt", NS)
    assert amount.text == "1180.00"  # full gross once the window closed
    assert root.findtext(".//p:CtrlSum", namespaces=NS) == "1180.00"


def test_export_rejects_an_unknown_format(service):
    run = _confirmed_run(service)
    with pytest.raises(ValueError, match="Unknown hand-off format"):
        service.export_payment_run_handoff(run.run_id, actor="Tomas Reid", format="mt940")
    with pytest.raises(ValueError, match="Unknown hand-off format"):
        render_handoff(run, "mt940")


def test_pain001_export_keeps_the_four_eyes_gate(monkeypatch):
    monkeypatch.setenv("PAYMENT_RUN_FOUR_EYES_THRESHOLD", "100")
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        object_store=InMemoryObjectStore(),
    )
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    runs = svc.propose_payment_runs(window_days=7, as_of="2026-10-05")
    run = runs[0]
    assert run.four_eyes_required is True
    svc.confirm_payment_run(run.run_id, actor="Tomas Reid")
    with pytest.raises(ValueError, match="second confirmer"):
        svc.export_payment_run_handoff(run.run_id, actor="Tomas Reid", format="pain001")
    svc.confirm_payment_run_second(run.run_id, actor="Priya Nair")
    exported = svc.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", format="pain001"
    )
    assert exported.handoff_format == "pain001"


def test_api_exports_and_downloads_pain001(service):
    run = _confirmed_run(service)
    client = TestClient(create_app(service))
    resp = client.post(
        f"/payment-runs/{run.run_id}/handoff?format=pain001",
        json={"actor": "Tomas Reid"},
    )
    assert resp.status_code == 200
    assert resp.json()["handoff_format"] == "pain001"
    download = client.get(f"/payment-runs/{run.run_id}/handoff")
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("application/xml")
    assert "handoff.pain001.xml" in download.headers["content-disposition"]
    assert ET.fromstring(download.content) is not None
    bad = client.post(
        f"/payment-runs/{run.run_id}/handoff?format=mt940",
        json={"actor": "Tomas Reid"},
    )
    assert bad.status_code == 422
