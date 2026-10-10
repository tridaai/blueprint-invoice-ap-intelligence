"""Bounced-group re-export: a rejected batch group is
re-presented alone, as its own corrected file.

The bank answers a batch-booked pain.001 per PmtInf group.
When a group bounces, the choices were bad ones: re-send the
whole file (re-presenting groups that already booked — the
bank could answer money twice) or leave the bounced group's
members stranded until the next run. The re-export is the
third way: the group goes out alone, a single-PmtInf pain.001
carrying the group's own id and only its members, at their
export-time figures with the discount drift re-verified as of
the re-export date. The bank's rejection reason is carried
onto the re-export record — the corrected file exists because
of it — and the group's receipts reset: the rejected answer
answered the first file, so the group stands unanswered
against the new one, and the bank's next answer records
against the group as before. Only a group whose standing
answer is a rejection can be re-exported: an accepted group
is final, and an unanswered group has not bounced.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.handoff import (
    PAIN001_NAMESPACE,
    batch_group_members,
    current_group_receipts,
    group_reexports,
    latest_group_reexport,
)
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import HandoffGroupReceiptInput
from ap_agent.service import InvoiceService
from ap_agent.storage import InMemoryObjectStore
from ap_agent.store import InMemoryStore
from test_handoff_batch_booking import _service_with_run

NS = {"p": PAIN001_NAMESPACE}
SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


def _receipt(date: str, status: str, **kw) -> HandoffGroupReceiptInput:
    return HandoffGroupReceiptInput(execution_date=date, status=status, **kw)


def _batch_exported() -> tuple[InvoiceService, object]:
    """A batch-booked EUR export: groups 2026-10-20 (INV-1001)
    and 2026-10-21 (INV-1008)."""
    svc, run = _service_with_run()
    svc.set_handoff_profile(
        actor="Ana Ruiz", scope="currency", currency="EUR",
        format="pain001", lead_days=3, batch_booking=True,
    )
    exported = svc.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-11"
    )
    assert exported.handoff_batch_booking is True
    return svc, exported


def _rejected(svc: InvoiceService, run) -> object:
    return svc.record_handoff_group_receipts(
        run.run_id,
        actor="Tomas Reid",
        receipts=[
            _receipt(
                "2026-10-21",
                "rejected",
                detail="creditor account closed at the receiving bank",
            )
        ],
    )


def _pmt_infs(data: bytes) -> list[ET.Element]:
    root = ET.fromstring(data)
    return root.findall("./p:CstmrCdtTrfInitn/p:PmtInf", NS)


# -- the re-export itself -------------------------------------------------------


def test_a_rejected_group_reexports_alone():
    svc, exported = _batch_exported()
    _rejected(svc, exported)
    run = svc.reexport_handoff_group(
        exported.run_id, "2026-10-21", actor="Tomas Reid", as_of="2026-10-15"
    )
    (entry,) = run.handoff_group_reexports
    assert entry.pmt_inf_id == f"{exported.run_id}-2026-10-21"
    assert entry.execution_date == "2026-10-21"
    # The group's date is still ahead of the re-export date, so
    # the new file asks for the same execution date.
    assert entry.reexport_execution_date == "2026-10-21"
    assert entry.member_invoice_ids == ["INV-1008"]
    assert entry.rejection_reason == (
        "creditor account closed at the receiving bank"
    )
    assert entry.reexported_by == "Tomas Reid"
    assert entry.key.endswith("handoff-group-2026-10-21-reexport-1.pain001.xml")
    assert entry.sha256
    assert entry.drift == []
    assert latest_group_reexport(run, "2026-10-21") is entry
    assert group_reexports(run, "2026-10-20") == []
    assert "re-exported alone" in run.note

    data = svc.get_handoff_group_reexport(exported.run_id, "2026-10-21")
    assert data is not None
    (pmt_inf,) = _pmt_infs(data)
    assert pmt_inf.findtext("p:PmtInfId", namespaces=NS) == entry.pmt_inf_id
    # A solo file: no batch booking, one transaction, the member only.
    assert pmt_inf.findtext("p:BtchBookg", namespaces=NS) is None
    assert pmt_inf.findtext("p:NbOfTxs", namespaces=NS) == "1"
    assert pmt_inf.findtext("p:ReqExctnDt", namespaces=NS) == "2026-10-21"
    txs = pmt_inf.findall("p:CdtTrfTxInf", NS)
    assert [
        tx.findtext(".//p:EndToEndId", namespaces=NS) for tx in txs
    ] == ["INV-1008"]
    header = ET.fromstring(data).find(
        "./p:CstmrCdtTrfInitn/p:GrpHdr", NS
    )
    assert header.findtext("p:MsgId", namespaces=NS) == entry.pmt_inf_id
    assert header.findtext("p:CtrlSum", namespaces=NS) == "1180.00"


def test_the_receipts_reset_and_the_bank_can_answer_the_new_file():
    svc, exported = _batch_exported()
    _rejected(svc, exported)
    svc.reexport_handoff_group(
        exported.run_id, "2026-10-21", actor="Tomas Reid", as_of="2026-10-15"
    )
    run = svc.get_payment_run(exported.run_id)
    # The rejected answer answered the first file: superseded,
    # and the group stands unanswered against the new one.
    (old,) = run.handoff_group_receipts
    assert old.superseded is True
    assert current_group_receipts(run) == {}
    run = svc.record_handoff_group_receipts(
        exported.run_id,
        actor="Tomas Reid",
        receipts=[_receipt("2026-10-21", "accepted", bank_reference="BNK-31")],
    )
    standing = current_group_receipts(run)["2026-10-21"]
    assert standing.status == "accepted"
    assert standing.member_invoice_ids == ["INV-1008"]
    # The other group was never touched by any of this.
    assert batch_group_members(run)["2026-10-20"] == ["INV-1001"]


def test_a_group_date_already_past_reexports_for_the_reexport_date():
    svc, exported = _batch_exported()
    _rejected(svc, exported)
    run = svc.reexport_handoff_group(
        exported.run_id, "2026-10-21", actor="Tomas Reid", as_of="2026-10-25"
    )
    (entry,) = run.handoff_group_reexports
    # Re-presenting on a date already past asks the bank for
    # yesterday: the new file executes on the re-export date.
    assert entry.reexport_execution_date == "2026-10-25"
    data = svc.get_handoff_group_reexport(exported.run_id, "2026-10-21")
    (pmt_inf,) = _pmt_infs(data)
    assert pmt_inf.findtext("p:ReqExctnDt", namespaces=NS) == "2026-10-25"


def test_a_group_can_bounce_twice_and_reexport_twice():
    svc, exported = _batch_exported()
    _rejected(svc, exported)
    svc.reexport_handoff_group(
        exported.run_id, "2026-10-21", actor="Tomas Reid", as_of="2026-10-15"
    )
    svc.record_handoff_group_receipts(
        exported.run_id,
        actor="Tomas Reid",
        receipts=[
            _receipt("2026-10-21", "rejected", detail="value date refused")
        ],
    )
    run = svc.reexport_handoff_group(
        exported.run_id, "2026-10-21", actor="Tomas Reid", as_of="2026-10-16"
    )
    assert len(run.handoff_group_reexports) == 2
    second = run.handoff_group_reexports[1]
    assert second.rejection_reason == "value date refused"
    assert second.key.endswith("reexport-2.pain001.xml")
    assert latest_group_reexport(run, "2026-10-21") is second


# -- the guards -------------------------------------------------------------------


def test_an_unanswered_group_has_not_bounced():
    svc, exported = _batch_exported()
    with pytest.raises(ValueError, match="nothing has bounced"):
        svc.reexport_handoff_group(
            exported.run_id, "2026-10-21", actor="Tomas Reid"
        )


def test_an_accepted_group_is_final():
    svc, exported = _batch_exported()
    svc.record_handoff_group_receipts(
        exported.run_id,
        actor="Tomas Reid",
        receipts=[_receipt("2026-10-21", "accepted", bank_reference="BNK-1")],
    )
    with pytest.raises(ValueError, match="stands accepted"):
        svc.reexport_handoff_group(
            exported.run_id, "2026-10-21", actor="Tomas Reid"
        )


def test_a_second_reexport_while_awaiting_the_answer_is_refused():
    svc, exported = _batch_exported()
    _rejected(svc, exported)
    svc.reexport_handoff_group(
        exported.run_id, "2026-10-21", actor="Tomas Reid", as_of="2026-10-15"
    )
    with pytest.raises(ValueError, match="already re-exported"):
        svc.reexport_handoff_group(
            exported.run_id, "2026-10-21", actor="Tomas Reid", as_of="2026-10-16"
        )


def test_an_unknown_group_is_refused_naming_the_real_groups():
    svc, exported = _batch_exported()
    with pytest.raises(ValueError, match="2026-10-20, 2026-10-21"):
        svc.reexport_handoff_group(
            exported.run_id, "2026-12-25", actor="Tomas Reid"
        )


def test_a_flat_export_has_no_group_to_reexport():
    svc, run = _service_with_run()
    svc.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-11", format="csv"
    )
    with pytest.raises(ValueError, match="batch-booked pain.001"):
        svc.reexport_handoff_group(run.run_id, "2026-10-21", actor="Tomas Reid")


def test_an_unattributed_reexport_is_refused():
    svc, exported = _batch_exported()
    _rejected(svc, exported)
    with pytest.raises(ValueError, match="actor"):
        svc.reexport_handoff_group(exported.run_id, "2026-10-21", actor=" ")


# -- drift is re-verified for the new file -----------------------------------------


def test_drift_is_reverified_as_of_the_reexport_date():
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        object_store=InMemoryObjectStore(),
    )
    svc.analyze(SAMPLES["INV-1001"])
    svc.analyze(SAMPLES["INV-1008"])
    _svc, run = _service_with_run()
    store = svc._get_store()
    # INV-1001's real discount: 2/10 net 30 on the 2026-09-28
    # sample — deadline 2026-10-08, 23.60 capturable.
    line = run.lines[0]
    line.discount_amount = 23.60
    line.discount_deadline = "2026-10-08"
    line.net_amount = 1156.40
    run.total_discount = 23.60
    run.total_net = run.total_gross - 23.60
    store.save_payment_run(run)
    svc.set_handoff_profile(
        actor="Ana Ruiz", scope="currency", currency="EUR",
        format="pain001", lead_days=3, batch_booking=True,
    )
    # Exported on 2026-10-05, inside the window: no drift yet.
    exported = svc.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-05"
    )
    assert exported.handoff_drift == []
    svc.record_handoff_group_receipts(
        run.run_id,
        actor="Tomas Reid",
        receipts=[
            _receipt("2026-10-20", "rejected", detail="bank holiday backlog")
        ],
    )
    # Re-exported on 2026-10-12, past the deadline: the member
    # is re-priced to full gross for the new file, itemised on
    # the re-export record.
    run = svc.reexport_handoff_group(
        run.run_id, "2026-10-20", actor="Tomas Reid", as_of="2026-10-12"
    )
    (entry,) = run.handoff_group_reexports
    assert entry.drift_total == 23.60
    (drift,) = entry.drift
    assert drift.invoice_id == "INV-1001"
    assert drift.discount_lost == 23.60
    data = svc.get_handoff_group_reexport(run.run_id, "2026-10-20")
    (pmt_inf,) = _pmt_infs(data)
    assert pmt_inf.findtext("p:CtrlSum", namespaces=NS) == "1180.00"
    # The run's own lines keep the FIRST file's figures — the
    # re-export record is the second file's receipt.
    stored = svc.get_payment_run(run.run_id)
    assert stored.lines[0].discount_amount == 23.60


# -- the API -----------------------------------------------------------------------


def test_reexport_endpoints():
    svc, exported = _batch_exported()
    _rejected(svc, exported)
    client = TestClient(create_app(svc))
    response = client.post(
        f"/payment-runs/{exported.run_id}/handoff/groups/2026-10-21/reexport",
        json={"actor": "Tomas Reid"},
        params={"as_of": "2026-10-15"},
    )
    assert response.status_code == 200
    body = response.json()
    (entry,) = body["handoff_group_reexports"]
    assert entry["execution_date"] == "2026-10-21"
    download = client.get(
        f"/payment-runs/{exported.run_id}/handoff/groups/2026-10-21/reexport"
    )
    assert download.status_code == 200
    assert entry["pmt_inf_id"] in download.text
    missing = client.get(
        f"/payment-runs/{exported.run_id}/handoff/groups/2026-10-20/reexport"
    )
    assert missing.status_code == 404
    refused = client.post(
        f"/payment-runs/{exported.run_id}/handoff/groups/2026-10-20/reexport",
        json={"actor": "Tomas Reid"},
    )
    assert refused.status_code == 422
