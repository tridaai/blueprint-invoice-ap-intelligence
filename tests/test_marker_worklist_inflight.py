"""The marker worklist's flow state: the ranked rows are joined
with the governed master-change pipeline, so a supplier already
being onboarded reads as in flight — change id, proposer, age,
decision still owed — instead of open work to chase a second
time, and a supplier whose change was rejected wears the
rejection (decider, date, note) so re-proposing is deliberate.
An expired pending change decided nothing: the row is open
again."""

from __future__ import annotations

from datetime import date

from fastapi.testclient import TestClient

from ap_agent.api import create_app
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


def _run(run_id: str, exported_at: str, lines, entries, currency="USD") -> PaymentRun:
    return PaymentRun(
        run_id=run_id,
        currency=currency,
        status="confirmed",
        handoff_generated_at=exported_at,
        handoff_format="csv",
        lines=lines,
        invoice_count=len(lines),
        handoff_creditor_entries=entries,
    )


def _line(invoice_id: str, supplier: str, net: float) -> PaymentRunLine:
    return PaymentRunLine(
        invoice_id=invoice_id, supplier_name=supplier, gross_amount=net, net_amount=net
    )


def _marked(invoice_id: str, supplier: str) -> HandoffCreditorEntry:
    return HandoffCreditorEntry(
        invoice_id=invoice_id, supplier_name=supplier, source="at_import"
    )


def _runs() -> list[PaymentRun]:
    return [
        _run(
            "RUN-USD-A",
            "2026-10-07T09:00:00+00:00",
            [_line("INV-1", NORTHBEAM, 1156.40), _line("INV-2", CASCADE, 500.0)],
            [_marked("INV-1", NORTHBEAM), _marked("INV-2", CASCADE)],
        )
    ]


def _change(supplier: str, **overrides) -> MasterChangeRequest:
    fields = dict(
        request_id="MCR-0001",
        supplier_name=supplier,
        proposed=SupplierMasterRecord(supplier_name=supplier, iban="DE89370400440532013000"),
        proposed_by="Priya Nair",
        proposed_at="2026-10-15T09:00:00+00:00",
        status="pending",
    )
    fields.update(overrides)
    return MasterChangeRequest(**fields)


def _item(worklist, supplier: str):
    return next(i for i in worklist.items if i.supplier_name == supplier)


def test_a_pending_change_marks_the_supplier_in_flight():
    worklist = marker_worklist(
        _runs(),
        changes=[_change(NORTHBEAM)],
        as_of=AS_OF,
        expiry_days=30,
    )
    item = _item(worklist, NORTHBEAM)
    assert item.onboarding_state == "in_flight"
    assert item.in_flight_change_id == "MCR-0001"
    assert item.in_flight_proposed_by == "Priya Nair"
    assert item.in_flight_age_days == 5  # proposed 10-15, read on 10-20
    assert worklist.in_flight_count == 1
    assert worklist.rejected_count == 0
    # The supplier nobody proposed a change for stays plainly open.
    other = _item(worklist, CASCADE)
    assert other.onboarding_state == "open"
    assert other.in_flight_change_id == "" and other.rejected_change_id == ""


def test_an_expired_pending_change_reads_open_again():
    worklist = marker_worklist(
        _runs(),
        changes=[_change(NORTHBEAM, proposed_at="2026-09-01T09:00:00+00:00")],
        as_of=AS_OF,  # 49 days on, past the 30-day decision window
        expiry_days=30,
    )
    item = _item(worklist, NORTHBEAM)
    assert item.onboarding_state == "open"
    assert item.in_flight_change_id == ""
    assert worklist.in_flight_count == 0


def test_an_approved_change_awaiting_its_effective_date_is_in_flight():
    worklist = marker_worklist(
        _runs(),
        changes=[
            _change(
                NORTHBEAM,
                status="approved",
                decided_by="Tomas Reid",
                decided_at="2026-10-18T10:00:00+00:00",
                applied_at=None,  # effective date still ahead
            )
        ],
        as_of=AS_OF,
        expiry_days=30,
    )
    assert _item(worklist, NORTHBEAM).onboarding_state == "in_flight"


def test_a_rejected_change_shows_the_rejection():
    worklist = marker_worklist(
        _runs(),
        changes=[
            _change(
                NORTHBEAM,
                request_id="MCR-0004",
                status="rejected",
                decided_by="Tomas Reid",
                decided_at="2026-10-18T10:00:00+00:00",
                decision_note="Call-back number did not match the number on file",
            )
        ],
        as_of=AS_OF,
        expiry_days=30,
    )
    item = _item(worklist, NORTHBEAM)
    assert item.onboarding_state == "rejected"
    assert item.rejected_change_id == "MCR-0004"
    assert item.rejected_by == "Tomas Reid"
    assert item.rejected_at == "2026-10-18T10:00:00+00:00"
    assert "did not match" in item.rejection_note
    assert worklist.rejected_count == 1


def test_the_latest_change_wins():
    older_rejection = _change(
        NORTHBEAM,
        request_id="MCR-0001",
        status="rejected",
        decided_by="Tomas Reid",
        decided_at="2026-10-10T10:00:00+00:00",
        decision_note="stale identifiers",
    )
    newer_pending = _change(
        NORTHBEAM, request_id="MCR-0002", proposed_at="2026-10-19T09:00:00+00:00"
    )
    worklist = marker_worklist(
        _runs(),
        changes=[older_rejection, newer_pending],
        as_of=AS_OF,
        expiry_days=30,
    )
    item = _item(worklist, NORTHBEAM)
    assert item.onboarding_state == "in_flight"
    assert item.in_flight_change_id == "MCR-0002"


def _service_with_marked_run() -> InvoiceService:
    store = InMemoryStore()
    for run in _runs():
        store.save_payment_run(run)
    return InvoiceService(
        store=store,
        checkpointer=False,
        supplier_master=InMemorySupplierMaster(),
    )


def test_the_service_joins_the_change_pipeline_end_to_end():
    svc = _service_with_marked_run()
    change = svc.propose_master_change(
        SupplierMasterRecord(
            supplier_name=NORTHBEAM, iban="DE89370400440532013000"
        ),
        actor="Priya Nair",
        reason="onboard the marked supplier",
    )
    worklist = svc.marker_worklist()
    item = _item(worklist, NORTHBEAM)
    assert item.onboarding_state == "in_flight"
    assert item.in_flight_change_id == change.request_id
    assert item.in_flight_age_days == 0  # proposed today

    svc.decide_master_change(
        change.request_id,
        actor="Tomas Reid",
        approve=False,
        note="Verification call reached a number the document supplied",
    )
    worklist = svc.marker_worklist()
    item = _item(worklist, NORTHBEAM)
    assert item.onboarding_state == "rejected"
    assert item.rejected_by == "Tomas Reid"
    assert "document supplied" in item.rejection_note


def test_the_api_serves_the_flow_state():
    svc = _service_with_marked_run()
    svc.propose_master_change(
        SupplierMasterRecord(
            supplier_name=NORTHBEAM, iban="DE89370400440532013000"
        ),
        actor="Priya Nair",
        reason="onboard the marked supplier",
    )
    client = TestClient(create_app(svc))
    body = client.get("/suppliers/marker-worklist").json()
    item = next(i for i in body["items"] if i["supplier_name"] == NORTHBEAM)
    assert item["onboarding_state"] == "in_flight"
    assert item["in_flight_proposed_by"] == "Priya Nair"
    assert body["in_flight_count"] == 1
