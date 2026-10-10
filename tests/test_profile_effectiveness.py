"""Profile effectiveness: grading the governed hand-off profiles.

Round 12's profiles govern exports and stamp themselves on the
run; this read grades them like the forecast is graded — per
profile: the exports it routed and their format mix, the marker
rate on its exports against the ungoverned population, whether
suppliers its marker-rejection stance flagged at proposal time
stopped exporting marked, and whether the lead promise it made
was kept by the dates actually written. A profile that governs
nothing, or whose files are no safer than ungoverned ones, is
configuration theatre — the grade says so in its notes.
"""

from __future__ import annotations

from ap_agent.handoff_profiles import (
    marker_flags,
    profile_effectiveness,
    profile_id_for,
)
from ap_agent.schemas import (
    HandoffCreditorEntry,
    HandoffProfile,
    PaymentRun,
    PaymentRunLine,
)
from ap_agent.service import InvoiceService
from ap_agent.storage import InMemoryObjectStore
from ap_agent.store import InMemoryStore


def _line(invoice_id: str, supplier: str, net: float) -> PaymentRunLine:
    return PaymentRunLine(
        invoice_id=invoice_id,
        supplier_name=supplier,
        invoice_number=f"{invoice_id}-N",
        gross_amount=net,
        net_amount=net,
    )


def _entry(invoice_id: str, supplier: str, source: str) -> HandoffCreditorEntry:
    return HandoffCreditorEntry(
        invoice_id=invoice_id, supplier_name=supplier, source=source
    )


def _exported(run_id: str, **kw) -> PaymentRun:
    base = dict(
        run_id=run_id,
        currency="EUR",
        status="confirmed",
        proposed_at="2026-10-05T09:00:00+00:00",
        handoff_generated_at="2026-10-11T10:00:00+00:00",
        handoff_format="pain001",
    )
    base.update(kw)
    return PaymentRun(**base)


EUR_PROFILE = HandoffProfile(
    profile_id="HP-CUR-EUR", scope="currency", currency="EUR",
    format="pain001", lead_days=3,
)


def _eur_runs() -> list[PaymentRun]:
    kept = _exported(
        "RUN-A",
        handoff_profile_id="HP-CUR-EUR",
        handoff_execution_date="2026-10-20",
        lines=[_line("INV-1", "Alpen", 100.0), _line("INV-2", "Berg", 300.0)],
        handoff_creditor_entries=[
            _entry("INV-1", "Alpen", "supplier_master"),
            _entry("INV-2", "Berg", "at_import"),
        ],
    )
    breached = _exported(
        "RUN-B",
        handoff_profile_id="HP-CUR-EUR",
        handoff_execution_date="2026-10-12",  # inside the 3d lead from 10-11
        lines=[_line("INV-3", "Alpen", 50.0)],
        handoff_creditor_entries=[_entry("INV-3", "Alpen", "supplier_master")],
    )
    ungoverned = _exported(
        "RUN-C",
        lines=[_line("INV-4", "Cima", 10.0), _line("INV-5", "Cima", 20.0)],
        handoff_creditor_entries=[
            _entry("INV-4", "Cima", "at_import"),
            _entry("INV-5", "Cima", "at_import"),
        ],
    )
    return [kept, breached, ungoverned]


def test_effectiveness_grades_exports_markers_and_lead():
    (grade,) = profile_effectiveness([EUR_PROFILE], _eur_runs())
    assert grade["exports_governed"] == 2
    assert grade["format_mix"] == {"pain001": 2}
    assert grade["members"] == 3
    assert grade["members_marked"] == 1
    assert grade["value_marked"] == 300.0
    assert grade["marker_rate"] == round(1 / 3, 4)
    assert grade["ungoverned_marker_rate"] == 1.0
    # Lead: RUN-A kept (10-20 >= 10-11 + 3d), RUN-B breached.
    assert grade["lead_checked"] == 2
    assert grade["lead_kept"] == 1
    assert grade["lead_breached"] == 1
    assert any("lead promise was breached" in note for note in grade["notes"])
    # The governed marker rate beats the ungoverned one: no
    # configuration-theatre note.
    assert not any("same rate as ungoverned" in note for note in grade["notes"])


def test_effectiveness_names_a_profile_that_governs_nothing():
    dormant = HandoffProfile(
        profile_id="HP-CUR-GBP", scope="currency", currency="GBP", format="csv"
    )
    (grade,) = profile_effectiveness([dormant], _eur_runs())
    assert grade["exports_governed"] == 0
    assert grade["marker_rate"] is None
    assert any("governs no exports yet" in note for note in grade["notes"])


def test_effectiveness_batch_execution_dates_all_count_for_the_lead():
    run = _exported(
        "RUN-BATCH",
        handoff_profile_id="HP-CUR-EUR",
        handoff_execution_date="2026-10-20, 2026-10-21",
        lines=[_line("INV-1", "Alpen", 100.0)],
        handoff_creditor_entries=[_entry("INV-1", "Alpen", "supplier_master")],
    )
    (grade,) = profile_effectiveness([EUR_PROFILE], [run])
    assert grade["lead_checked"] == 1
    assert grade["lead_kept"] == 1


def _flagged_profile(supplier: str) -> HandoffProfile:
    return HandoffProfile(
        profile_id=profile_id_for("supplier", supplier_name=supplier),
        scope="supplier",
        supplier_name=supplier,
        format="pain001",
        markers_rejected=True,
    )


def _proposal_with_flag(profile: HandoffProfile, supplier: str) -> PaymentRun:
    run = PaymentRun(
        run_id=f"RUN-PROP-{supplier}",
        currency="EUR",
        status="proposed",
        proposed_at="2026-10-05T09:00:00+00:00",
        lines=[_line("INV-P", supplier, 250.0)],
        invoice_count=1,
    )
    run.handoff_profile_flags = marker_flags(run, [profile], master_covered=())
    assert run.handoff_profile_flags  # the scenario really flags
    return run


def test_flagged_supplier_still_marked_is_named():
    profile = _flagged_profile("Acme Parts")
    proposal = _proposal_with_flag(profile, "Acme Parts")
    later = _exported(
        "RUN-LATER",
        lines=[_line("INV-9", "Acme Parts", 250.0)],
        handoff_creditor_entries=[_entry("INV-9", "Acme Parts", "at_import")],
    )
    (grade,) = profile_effectiveness([profile], [proposal, later])
    (slot,) = grade["flagged_suppliers"]
    assert slot["supplier"] == "Acme Parts"
    assert slot["still_marked_exports"] == 1
    assert slot["still_marked_value"] == 250.0
    assert slot["stopped_exporting_marked"] is False
    assert any("Acme Parts" in note and "still exported marked" in note
               for note in grade["notes"])


def test_flagged_supplier_who_stops_reads_as_stopped():
    profile = _flagged_profile("Berg GmbH")
    proposal = _proposal_with_flag(profile, "Berg GmbH")
    later = _exported(
        "RUN-LATER-2",
        lines=[_line("INV-10", "Berg GmbH", 80.0)],
        handoff_creditor_entries=[
            _entry("INV-10", "Berg GmbH", "supplier_master")
        ],
    )
    (grade,) = profile_effectiveness([profile], [proposal, later])
    (slot,) = grade["flagged_suppliers"]
    assert slot["later_exports"] == 1
    assert slot["still_marked_exports"] == 0
    assert slot["stopped_exporting_marked"] is True
    assert not any("still exported marked" in note for note in grade["notes"])


def test_metrics_snapshot_carries_the_grades():
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        object_store=InMemoryObjectStore(),
    )
    svc.set_handoff_profile(
        actor="Ana Ruiz", scope="currency", currency="EUR",
        format="pain001", lead_days=3,
    )
    svc._get_store().save_payment_run(_eur_runs()[0])
    grades = svc.metrics_snapshot()["handoff_profile_effectiveness"]
    assert [g["profile_id"] for g in grades] == ["HP-CUR-EUR"]
    assert grades[0]["exports_governed"] == 1
