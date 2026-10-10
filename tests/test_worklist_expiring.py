"""Worklist in-flight aging: the closing window, warned early.

A pending master change undecided past its window expires, and
the supplier silently returns to open work — the onboarding
effort evaporates on day 31 without anyone noticing. The
worklist now ages each in-flight row against that window: days
left on the row, and inside the warning band
(MASTER_CHANGE_EXPIRY_WARNING_DAYS, default 7) the change is
flagged expiring soon, counted in the header, and named in the
digest — decision owed, window closing — while a decision can
still rescue it. Approved changes never expire and carry no
warning.
"""

from __future__ import annotations

from datetime import date

from ap_agent.schemas import (
    HandoffCreditorEntry,
    MasterChangeRequest,
    PaymentRun,
    PaymentRunLine,
    SupplierMasterRecord,
)
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore
from ap_agent.supplier_master import InMemorySupplierMaster
from ap_agent.worklist import marker_worklist

NORTHBEAM = "Northbeam Office Supplies"
CASCADE = "Cascade IT Hardware"
AS_OF = date(2026, 10, 20)


def _run() -> PaymentRun:
    lines = [
        PaymentRunLine(
            invoice_id="INV-1", supplier_name=NORTHBEAM,
            gross_amount=500.0, net_amount=500.0,
        ),
        PaymentRunLine(
            invoice_id="INV-2", supplier_name=CASCADE,
            gross_amount=300.0, net_amount=300.0,
        ),
    ]
    return PaymentRun(
        run_id="RUN-USD-1",
        currency="USD",
        status="confirmed",
        handoff_generated_at="2026-10-06T09:00:00",
        handoff_format="csv",
        lines=lines,
        invoice_count=2,
        handoff_creditor_entries=[
            HandoffCreditorEntry(
                invoice_id="INV-1", supplier_name=NORTHBEAM, source="at_import"
            ),
            HandoffCreditorEntry(
                invoice_id="INV-2", supplier_name=CASCADE, source="at_import"
            ),
        ],
    )


def _change(
    request_id: str, supplier: str, proposed_at: str, status: str = "pending"
) -> MasterChangeRequest:
    return MasterChangeRequest(
        request_id=request_id,
        supplier_name=supplier,
        proposed=SupplierMasterRecord(supplier_name=supplier),
        proposed_by="Ana Ruiz",
        proposed_at=proposed_at,
        status=status,
    )


def _worklist(changes, **kw):
    return marker_worklist(
        [_run()], changes=changes, as_of=AS_OF, expiry_days=30, **kw
    )


def test_pending_change_ages_against_its_window():
    # Proposed 2026-09-24: 26 days waited at AS_OF, 4 left —
    # inside the default 7-day warning band.
    worklist = _worklist([_change("MCR-0007", NORTHBEAM, "2026-09-24T09:00:00")])
    row = next(i for i in worklist.items if i.supplier_name == NORTHBEAM)
    assert row.onboarding_state == "in_flight"
    assert row.in_flight_age_days == 26
    assert row.in_flight_days_to_expiry == 4
    assert row.in_flight_expiring_soon is True
    assert worklist.expiring_soon_count == 1


def test_a_young_change_is_in_flight_but_not_expiring():
    worklist = _worklist([_change("MCR-0008", NORTHBEAM, "2026-10-15T09:00:00")])
    row = next(i for i in worklist.items if i.supplier_name == NORTHBEAM)
    assert row.in_flight_age_days == 5
    assert row.in_flight_days_to_expiry == 25
    assert row.in_flight_expiring_soon is False
    assert worklist.expiring_soon_count == 0


def test_the_last_day_of_the_window_still_warns():
    # Proposed exactly 30 days before AS_OF: age == window, not
    # yet expired (expiry is age > window) — and 0 days left.
    worklist = _worklist([_change("MCR-0009", NORTHBEAM, "2026-09-20T09:00:00")])
    row = next(i for i in worklist.items if i.supplier_name == NORTHBEAM)
    assert row.onboarding_state == "in_flight"
    assert row.in_flight_days_to_expiry == 0
    assert row.in_flight_expiring_soon is True


def test_an_approved_change_never_expires_and_carries_no_warning():
    change = _change("MCR-0010", CASCADE, "2026-09-01T09:00:00", status="approved")
    worklist = _worklist([change])
    row = next(i for i in worklist.items if i.supplier_name == CASCADE)
    assert row.onboarding_state == "in_flight"
    assert row.in_flight_days_to_expiry is None
    assert row.in_flight_expiring_soon is False


def test_the_warning_band_is_configurable():
    changes = [_change("MCR-0007", NORTHBEAM, "2026-09-24T09:00:00")]
    worklist = _worklist(changes, expiry_warning_days=3)
    row = next(i for i in worklist.items if i.supplier_name == NORTHBEAM)
    assert row.in_flight_days_to_expiry == 4
    assert row.in_flight_expiring_soon is False  # 4 left, band is 3


def test_digest_names_the_closing_window():
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        supplier_master=InMemorySupplierMaster(),
    )
    store = svc._get_store()
    store.save_payment_run(_run())
    store.save_master_change(
        _change("MCR-0007", NORTHBEAM, "2026-09-24T09:00:00")
    )
    digest = svc.escalation_digest(as_of="2026-10-20")
    assert digest.onboarding_expiring_count == 1
    entry = digest.onboarding_expiring[0]
    assert entry.supplier_name == NORTHBEAM
    assert entry.change_id == "MCR-0007"
    assert entry.days_to_expiry == 4
    assert entry.marked_value == 500.0
    assert "Onboarding decisions owed, window closing" in digest.summary
    assert "MCR-0007" in digest.summary
    # A younger change on the other supplier stays out of it.
    store.save_master_change(
        _change("MCR-0011", CASCADE, "2026-10-18T09:00:00")
    )
    digest = svc.escalation_digest(as_of="2026-10-20")
    assert digest.onboarding_expiring_count == 1
