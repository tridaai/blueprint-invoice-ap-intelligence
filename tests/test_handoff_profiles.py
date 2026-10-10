"""Hand-off routing profiles: the payee side's say in the export.

One run leaves as one file, but the right format belongs to the
payee side — an ERP that ingests CSV, a bank that wants pain.001
with lead days, a bank that rejects at-import markers. The
governed profile records that knowledge per supplier or per
currency (actor named, audit trail appended); the export
resolves it deterministically — an explicit format wins, then
the currency profile, then a supplier profile the run's members
agree on, else the CSV default — and the marker-rejection
stance is flagged at proposal time, not discovered at import.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from ap_agent.handoff import PAIN001_NAMESPACE
from ap_agent.handoff_profiles import (
    governing_profile,
    marker_flags,
    profile_id_for,
    requested_execution_date,
    resolve_export_format,
)
from ap_agent.schemas import HandoffProfile, PaymentRun, PaymentRunLine
from ap_agent.service import InvoiceService
from ap_agent.storage import InMemoryObjectStore
from ap_agent.store import InMemoryStore
from ap_agent.supplier_master import InMemorySupplierMaster
from ap_agent.schemas import SupplierMasterRecord

NS = {"p": PAIN001_NAMESPACE}
IBAN = "DE89370400440532013000"


def _line(invoice_id: str, supplier: str) -> PaymentRunLine:
    return PaymentRunLine(
        invoice_id=invoice_id,
        supplier_name=supplier,
        invoice_number=f"{invoice_id}-N",
        gross_amount=100.0,
        net_amount=100.0,
    )


def _run(currency: str = "EUR", suppliers=("Alpen Werkzeug GmbH",)) -> PaymentRun:
    lines = [
        _line(f"INV-{2000 + i}", supplier)
        for i, supplier in enumerate(suppliers)
    ]
    return PaymentRun(
        run_id=f"RUN-{currency}-2026-10-05-2026-10-11",
        currency=currency,
        window_start="2026-10-05",
        window_end="2026-10-11",
        status="confirmed",
        lines=lines,
        invoice_count=len(lines),
        total_gross=100.0 * len(lines),
        total_net=100.0 * len(lines),
    )


def _profile(**kw) -> HandoffProfile:
    base = dict(
        profile_id="HP-CUR-EUR",
        scope="currency",
        currency="EUR",
        format="pain001",
    )
    base.update(kw)
    return HandoffProfile(**base)


def _service(master=None) -> InvoiceService:
    return InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        object_store=InMemoryObjectStore(),
        supplier_master=master or InMemorySupplierMaster(),
    )


# --- profile ids and resolution -------------------------------------------


def test_profile_ids_are_deterministic_per_scope_key():
    assert profile_id_for("currency", currency="eur") == "HP-CUR-EUR"
    assert (
        profile_id_for("supplier", supplier_name="Alpen Werkzeug GmbH")
        == "HP-SUP-ALPEN-WERKZEUG-GMBH"
    )


def test_currency_profile_governs_over_supplier_profiles():
    run = _run()
    currency_profile = _profile()
    supplier_profile = _profile(
        profile_id="HP-SUP-ALPEN-WERKZEUG-GMBH",
        scope="supplier",
        supplier_name="Alpen Werkzeug GmbH",
        currency="",
        format="csv",
    )
    assert governing_profile(run, [supplier_profile, currency_profile]) is (
        currency_profile
    )


def test_supplier_profiles_govern_only_in_unison():
    run = _run(suppliers=("Supplier A", "Supplier B"))
    a = _profile(
        profile_id="HP-SUP-SUPPLIER-A",
        scope="supplier",
        supplier_name="Supplier A",
        currency="",
        format="pain001",
        lead_days=2,
    )
    b = _profile(
        profile_id="HP-SUP-SUPPLIER-B",
        scope="supplier",
        supplier_name="Supplier B",
        currency="",
        format="pain001",
        lead_days=2,
    )
    # Same routing values under two records: one voice.
    assert governing_profile(run, [a, b]) is not None
    b.lead_days = 5
    # Split voices: no profile governs the run.
    assert governing_profile(run, [a, b]) is None


def test_explicit_format_wins_and_default_is_csv():
    run = _run()
    fmt, profile, basis = resolve_export_format(run, [_profile()], "csv")
    assert fmt == "csv" and profile is not None and "requested" in basis
    fmt, profile, basis = resolve_export_format(run, [_profile()], "")
    assert fmt == "pain001" and "HP-CUR-EUR" in basis
    fmt, profile, basis = resolve_export_format(run, [], "")
    assert fmt == "csv" and profile is None and "default" in basis


def test_lead_days_push_execution_date_but_never_before_window_end():
    from datetime import date

    run = _run()
    profile = _profile(lead_days=3)
    # Export on the window's first day: 2026-10-05 + 3d is still
    # inside the window — the window end stands.
    assert (
        requested_execution_date(run, profile, export_date=date(2026, 10, 5))
        == "2026-10-11"
    )
    # Export on the window end: the lead pushes past it.
    assert (
        requested_execution_date(run, profile, export_date=date(2026, 10, 11))
        == "2026-10-14"
    )
    assert (
        requested_execution_date(run, None, export_date=date(2026, 10, 11))
        == "2026-10-11"
    )


# --- the governed write ----------------------------------------------------


def test_set_profile_requires_an_actor_and_validates():
    svc = _service()
    with pytest.raises(ValueError, match="actor"):
        svc.set_handoff_profile(actor="", scope="currency", currency="EUR")
    with pytest.raises(ValueError, match="3-letter"):
        svc.set_handoff_profile(actor="Ana", scope="currency", currency="EU")
    with pytest.raises(ValueError, match="supplier_name"):
        svc.set_handoff_profile(actor="Ana", scope="supplier")
    with pytest.raises(ValueError, match="format"):
        svc.set_handoff_profile(
            actor="Ana", scope="currency", currency="EUR", format="pdf"
        )
    with pytest.raises(ValueError, match="lead_days"):
        svc.set_handoff_profile(
            actor="Ana", scope="currency", currency="EUR", lead_days=45
        )


def test_set_profile_upserts_with_an_append_only_audit_trail():
    svc = _service()
    created = svc.set_handoff_profile(
        actor="Ana Ruiz",
        scope="currency",
        currency="eur",
        format="pain001",
        lead_days=3,
        note="Deutsche Bank portal",
    )
    assert created.profile_id == "HP-CUR-EUR"
    assert created.currency == "EUR"
    assert [e.action for e in created.history] == ["created"]
    updated = svc.set_handoff_profile(
        actor="Tomas Reid",
        scope="currency",
        currency="EUR",
        format="pain001",
        lead_days=5,
    )
    assert updated.profile_id == created.profile_id
    assert updated.lead_days == 5
    assert updated.updated_by == "Tomas Reid"
    assert [e.action for e in updated.history] == ["created", "updated"]
    assert [e.actor for e in updated.history] == ["Ana Ruiz", "Tomas Reid"]
    assert svc.list_handoff_profiles() == [updated]
    summary = svc.metrics_snapshot()["handoff_profiles"]
    assert summary == {
        "count": 1,
        "by_scope": {"currency": 1},
        "by_format": {"pain001": 1},
        "markers_rejected": 0,
    }


# --- export resolution end to end ------------------------------------------


def test_export_uses_the_currency_profile_and_stamps_the_run():
    svc = _service()
    run = _run()
    svc._get_store().save_payment_run(run)
    svc.set_handoff_profile(
        actor="Ana Ruiz", scope="currency", currency="EUR",
        format="pain001", lead_days=3,
    )
    exported = svc.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-11"
    )
    assert exported.handoff_format == "pain001"
    assert exported.handoff_profile_id == "HP-CUR-EUR"
    assert exported.handoff_execution_date == "2026-10-14"
    assert "HP-CUR-EUR" in exported.note
    data = svc.get_payment_run_handoff(run.run_id)
    root = ET.fromstring(data)
    req = root.find(".//p:PmtInf/p:ReqExctnDt", NS)
    assert req is not None and req.text == "2026-10-14"


def test_export_without_a_profile_stays_csv_and_unstamped():
    svc = _service()
    run = _run()
    svc._get_store().save_payment_run(run)
    exported = svc.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-11"
    )
    assert exported.handoff_format == "csv"
    assert exported.handoff_profile_id == ""
    assert exported.handoff_execution_date == ""


def test_explicit_format_beats_the_profile_but_the_lead_still_applies():
    svc = _service()
    run = _run()
    svc._get_store().save_payment_run(run)
    svc.set_handoff_profile(
        actor="Ana Ruiz", scope="currency", currency="EUR",
        format="csv", lead_days=2,
    )
    exported = svc.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-11", format="pain001"
    )
    assert exported.handoff_format == "pain001"
    assert exported.handoff_profile_id == "HP-CUR-EUR"
    assert exported.handoff_execution_date == "2026-10-13"


# --- proposal-time marker flags --------------------------------------------


def test_marker_flags_name_the_uncovered_rejecting_supplier_only():
    run = _run(suppliers=("Supplier A", "Supplier B"))
    a = _profile(
        profile_id="HP-SUP-SUPPLIER-A",
        scope="supplier",
        supplier_name="Supplier A",
        currency="",
        markers_rejected=True,
    )
    b = _profile(
        profile_id="HP-SUP-SUPPLIER-B",
        scope="supplier",
        supplier_name="Supplier B",
        currency="",
        markers_rejected=True,
    )
    flags = marker_flags(run, [a, b], master_covered={"Supplier B"})
    assert len(flags) == 1
    assert flags[0].startswith("Supplier A:")
    assert "HP-SUP-SUPPLIER-A" in flags[0]
    # A profile that accepts markers flags nothing; B, still
    # rejecting and now uncovered, is flagged instead.
    a.markers_rejected = False
    flags = marker_flags(run, [a, b], master_covered=set())
    assert len(flags) == 1 and flags[0].startswith("Supplier B:")


def test_proposal_stamps_marker_flags_end_to_end():
    from ap_agent.samples import sample_invoice_models

    samples = {s.invoice_id: s for s in sample_invoice_models()}
    # Northbeam (INV-1001): its bank rejects markers and the
    # master holds nothing for it — the run must be flagged at
    # proposal. Give the master an IBAN and the flag clears.
    svc = _service()
    svc.set_handoff_profile(
        actor="Ana Ruiz", scope="supplier",
        supplier_name="Northbeam Office Supplies", markers_rejected=True,
    )
    svc.analyze(samples["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    run = svc.propose_payment_runs(window_days=7, as_of="2026-10-05")[0]
    assert len(run.handoff_profile_flags) == 1
    assert "Northbeam Office Supplies" in run.handoff_profile_flags[0]

    master = InMemorySupplierMaster(
        [SupplierMasterRecord(supplier_name="Northbeam Office Supplies", iban=IBAN)]
    )
    svc2 = _service(master=master)
    svc2.set_handoff_profile(
        actor="Ana Ruiz", scope="supplier",
        supplier_name="Northbeam Office Supplies", markers_rejected=True,
    )
    svc2.analyze(samples["INV-1001"])
    svc2.approve("INV-1001", approver="Priya Nair", reason="clean match")
    run2 = svc2.propose_payment_runs(window_days=7, as_of="2026-10-05")[0]
    assert run2.handoff_profile_flags == []
