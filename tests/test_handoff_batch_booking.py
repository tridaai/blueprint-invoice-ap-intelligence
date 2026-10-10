"""pain.001 batch booking: one PmtInf per requested execution date.

Some banks book per value date. When the governing hand-off
profile calls for batch booking, the pain.001 export groups the
run's lines by requested execution date — one ``PmtInf`` per
date, ``BtchBookg`` true — instead of one ``PmtInf`` for the
whole run. The profile's lead days floor every group's date:
the floor wins when it is later (the bank keeps its processing
lead), the line's due date wins when IT is later (nothing
executes before it falls due). The completion and drift gates
are per member, so they hold inside each group exactly as they
hold for the file. The default single-PmtInf output is
byte-identical to before (its golden file still pins it), and
the CSV stays the flat list.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ap_agent.handoff import (
    PAIN001_NAMESPACE,
    batch_groups,
    render_handoff,
    render_handoff_pain001,
)
from ap_agent.handoff_profiles import governing_profile, profile_detail
from ap_agent.paymentruns import creditor_completion_metrics
from ap_agent.schemas import HandoffProfile, PaymentRun, PaymentRunLine
from ap_agent.service import InvoiceService
from ap_agent.storage import InMemoryObjectStore
from ap_agent.store import InMemoryStore
from test_handoff_pain001 import _fixed_run

NS = {"p": PAIN001_NAMESPACE}
BATCH_GOLDEN = Path(__file__).parent / "golden" / "handoff_pain001_batch.xml"
IBAN = "DE89370400440532013000"


def _pmt_infs(data: bytes) -> list[ET.Element]:
    root = ET.fromstring(data)
    return root.findall("./p:CstmrCdtTrfInitn/p:PmtInf", NS)


def _group_lines(pmt_inf: ET.Element) -> list[str]:
    return [
        tx.findtext(".//p:EndToEndId", namespaces=NS)
        for tx in pmt_inf.findall("p:CdtTrfTxInf", NS)
    ]


# ---------------------------------------------------------------------------
# The grouping rule (which date wins, per group)
# ---------------------------------------------------------------------------


def test_batch_groups_by_due_date():
    groups = batch_groups(_fixed_run())
    assert [date for date, _ in groups] == ["2026-10-20", "2026-10-21"]
    assert [line.invoice_id for line in groups[0][1]] == ["INV-1001"]
    assert [line.invoice_id for line in groups[1][1]] == ["INV-1008"]


def test_batch_groups_lead_floor_wins_when_later():
    # Floor 2026-10-23 is past both due dates: one group, floored.
    groups = batch_groups(_fixed_run(), floor_date="2026-10-23")
    assert [date for date, _ in groups] == ["2026-10-23"]
    assert [line.invoice_id for line in groups[0][1]] == ["INV-1001", "INV-1008"]


def test_batch_groups_due_date_wins_when_later():
    # Floor 2026-10-14 precedes both due dates: the dues stand.
    groups = batch_groups(_fixed_run(), floor_date="2026-10-14")
    assert [date for date, _ in groups] == ["2026-10-20", "2026-10-21"]


def test_batch_groups_line_without_due_date_uses_the_window_end():
    run = _fixed_run()
    run.lines[0].due_date = ""
    groups = batch_groups(run)
    assert groups[0][0] == run.window_end
    assert groups[0][1][0].invoice_id == "INV-1001"


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------


def test_batch_render_matches_the_golden_file():
    assert render_handoff_pain001(_fixed_run(), batch_booking=True) == (
        BATCH_GOLDEN.read_bytes()
    )


def test_batch_render_groups_carry_their_own_control_sums():
    data = render_handoff_pain001(_fixed_run(), batch_booking=True)
    root = ET.fromstring(data)
    header = root.find("./p:CstmrCdtTrfInitn/p:GrpHdr", NS)
    assert header.findtext("p:NbOfTxs", namespaces=NS) == "2"
    assert header.findtext("p:CtrlSum", namespaces=NS) == "2336.40"

    pmt_infs = _pmt_infs(data)
    assert len(pmt_infs) == 2
    seen: list[str] = []
    for pmt_inf, expected_date, expected_sum in (
        (pmt_infs[0], "2026-10-20", "1156.40"),
        (pmt_infs[1], "2026-10-21", "1180.00"),
    ):
        assert pmt_inf.findtext("p:BtchBookg", namespaces=NS) == "true"
        assert pmt_inf.findtext("p:ReqExctnDt", namespaces=NS) == expected_date
        assert pmt_inf.findtext("p:CtrlSum", namespaces=NS) == expected_sum
        amounts = [
            float(tx.find("p:Amt/p:InstdAmt", NS).text)
            for tx in pmt_inf.findall("p:CdtTrfTxInf", NS)
        ]
        assert sum(amounts) == float(expected_sum)
        seen += _group_lines(pmt_inf)
    # Every run line lands in exactly one group.
    assert sorted(seen) == ["INV-1001", "INV-1008"]


def test_batch_render_completion_gate_holds_inside_each_group():
    completions = {
        "INV-1001": {"iban": IBAN, "account_number": "", "sort_code": ""}
    }
    data = render_handoff_pain001(
        _fixed_run(), creditor_accounts=completions, batch_booking=True
    )
    first, second = _pmt_infs(data)
    assert first.findtext(".//p:IBAN", namespaces=NS) == IBAN
    assert second.findtext(".//p:IBAN", namespaces=NS) is None
    assert "SUPPLIER-MASTER-AT-IMPORT" in ET.tostring(second, encoding="unicode")


def test_single_pmtinf_default_has_no_batch_booking_element():
    data = render_handoff_pain001(_fixed_run())
    assert b"BtchBookg" not in data
    assert len(_pmt_infs(data)) == 1


# ---------------------------------------------------------------------------
# The profile's say
# ---------------------------------------------------------------------------


def test_profile_detail_and_unison_cover_batch_booking():
    profile = HandoffProfile(
        profile_id="HP-CUR-EUR", scope="currency", currency="EUR",
        format="pain001", batch_booking=True,
    )
    assert "batch booking" in profile_detail(profile)

    run = PaymentRun(
        run_id="RUN-EUR-X", currency="EUR", status="confirmed",
        lines=[
            PaymentRunLine(invoice_id="INV-1", supplier_name="Alpen"),
            PaymentRunLine(invoice_id="INV-2", supplier_name="Berg"),
        ],
        invoice_count=2,
    )
    alpen = HandoffProfile(
        profile_id="HP-SUP-ALPEN", scope="supplier", supplier_name="Alpen",
        format="pain001", batch_booking=True,
    )
    berg_split = HandoffProfile(
        profile_id="HP-SUP-BERG", scope="supplier", supplier_name="Berg",
        format="pain001", batch_booking=False,
    )
    # Split voices on batch booking alone: no profile governs.
    assert governing_profile(run, [alpen, berg_split]) is None
    berg_unison = berg_split.model_copy(update={"batch_booking": True})
    assert governing_profile(run, [alpen, berg_unison]) is not None


# ---------------------------------------------------------------------------
# End to end: profile → export → stamps → coverage split
# ---------------------------------------------------------------------------


def _service_with_run() -> tuple[InvoiceService, PaymentRun]:
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        object_store=InMemoryObjectStore(),
    )
    run = _fixed_run()
    run.currency = "EUR"
    for line in run.lines:
        line.discount_amount = 0.0
        line.discount_deadline = ""
        line.net_amount = line.gross_amount
    run.total_discount = 0.0
    run.total_net = run.total_gross
    svc._get_store().save_payment_run(run)
    return svc, run


def test_export_batch_books_and_stamps_groups_and_entries():
    svc, run = _service_with_run()
    svc.set_handoff_profile(
        actor="Ana Ruiz", scope="currency", currency="EUR",
        format="pain001", lead_days=3, batch_booking=True,
    )
    exported = svc.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-11"
    )
    assert exported.handoff_format == "pain001"
    assert exported.handoff_profile_id == "HP-CUR-EUR"
    # Floor = 2026-10-11 + 3d = 2026-10-14 — before both due
    # dates, so the dues govern their groups.
    assert exported.handoff_execution_date == "2026-10-20, 2026-10-21"
    assert "batch-booked" in exported.note

    data = svc.get_payment_run_handoff(run.run_id)
    pmt_infs = _pmt_infs(data)
    assert len(pmt_infs) == 2
    assert all(
        p.findtext("p:BtchBookg", namespaces=NS) == "true" for p in pmt_infs
    )

    entries = {e.invoice_id: e for e in exported.handoff_creditor_entries}
    assert entries["INV-1001"].execution_date == "2026-10-20"
    assert entries["INV-1008"].execution_date == "2026-10-21"

    coverage = creditor_completion_metrics([exported])
    by_group = {
        cell["execution_date"]: cell
        for cell in coverage["by_execution_group"]
    }
    assert set(by_group) == {"2026-10-20", "2026-10-21"}
    assert by_group["2026-10-20"]["members"] == 1
    assert by_group["2026-10-20"]["runs"] == 1


def test_export_batch_floor_collapses_groups_when_it_is_latest():
    svc, run = _service_with_run()
    svc.set_handoff_profile(
        actor="Ana Ruiz", scope="currency", currency="EUR",
        format="pain001", lead_days=12, batch_booking=True,
    )
    exported = svc.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-11"
    )
    # Floor = 2026-10-23, past both due dates: a single group.
    assert exported.handoff_execution_date == "2026-10-23"
    pmt_infs = _pmt_infs(svc.get_payment_run_handoff(run.run_id))
    assert len(pmt_infs) == 1
    assert sorted(_group_lines(pmt_infs[0])) == ["INV-1001", "INV-1008"]


def test_batch_profile_with_csv_format_stays_a_flat_csv():
    svc, run = _service_with_run()
    svc.set_handoff_profile(
        actor="Ana Ruiz", scope="currency", currency="EUR",
        format="csv", batch_booking=True,
    )
    exported = svc.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-11"
    )
    assert exported.handoff_format == "csv"
    assert exported.handoff_execution_date == ""
    assert all(
        entry.execution_date == "" for entry in exported.handoff_creditor_entries
    )
    assert render_handoff(exported, "csv").count(b"\n") >= 3  # header + lines + total


def test_explicit_format_still_batch_books_under_a_batch_profile():
    svc, run = _service_with_run()
    svc.set_handoff_profile(
        actor="Ana Ruiz", scope="currency", currency="EUR",
        format="pain001", batch_booking=True,
    )
    exported = svc.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-11", format="pain001"
    )
    assert len(_pmt_infs(svc.get_payment_run_handoff(run.run_id))) == 2


def test_profile_without_batch_booking_keeps_single_pmtinf():
    svc, run = _service_with_run()
    svc.set_handoff_profile(
        actor="Ana Ruiz", scope="currency", currency="EUR",
        format="pain001", lead_days=3, batch_booking=False,
    )
    exported = svc.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-11"
    )
    assert len(_pmt_infs(svc.get_payment_run_handoff(run.run_id))) == 1
    assert exported.handoff_execution_date == "2026-10-14"  # window end + lead rule
    entries = {e.invoice_id: e for e in exported.handoff_creditor_entries}
    assert entries["INV-1001"].execution_date == "2026-10-14"
