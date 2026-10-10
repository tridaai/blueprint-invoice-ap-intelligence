"""Group-receipt ageing: an unanswered batch group joins the
queue, the digest, and the metrics.

The bank answers a batch-booked pain.001 per PmtInf group,
and the receipts made *answers* visible — but an answer that
never comes was still invisible: the file went out, the other
groups booked, and one group's members hung unbooked and
unbounced with no surface ageing them. Past
``GROUP_RECEIPT_AGING_DAYS`` the group becomes work like the
other waits: the queue names its run, members, value, and how
long the bank has had the file; the digest announces the
silence; ``/metrics`` counts the feedback loop's coverage. The
clock runs from the file the bank is currently answering — a
group re-presented alone ages from its solo re-export, not
from the original file it already answered by bouncing.
"""

from __future__ import annotations

from datetime import date

from ap_agent.handoff import group_file_sent_at
from ap_agent.paymentruns import group_answer_metrics
from ap_agent.queue import unanswered_group_items
from ap_agent.schemas import (
    HandoffCreditorEntry,
    HandoffGroupReceipt,
    HandoffGroupReexport,
    PaymentRun,
    PaymentRunLine,
)
from ap_agent.service import InvoiceService, group_receipt_aging_days
from ap_agent.store import InMemoryStore

SENT = "2026-10-11T09:00:00+00:00"


def _batch_run() -> PaymentRun:
    """A batch-booked EUR export sent 2026-10-11: groups
    2026-10-20 (INV-1001) and 2026-10-21 (INV-1008)."""
    return PaymentRun(
        run_id="RUN-EUR-1",
        currency="EUR",
        window_start="2026-10-05",
        window_end="2026-10-11",
        window_days=7,
        status="confirmed",
        lines=[
            PaymentRunLine(
                invoice_id="INV-1001",
                supplier_name="Northbeam Office Supplies",
                due_date="2026-10-20",
                gross_amount=1180.00,
                net_amount=1180.00,
            ),
            PaymentRunLine(
                invoice_id="INV-1008",
                supplier_name="Northbeam Office Supplies",
                due_date="2026-10-21",
                gross_amount=1180.00,
                net_amount=1180.00,
            ),
        ],
        invoice_count=2,
        total_gross=2360.00,
        total_net=2360.00,
        proposed_at="2026-10-05T09:00:00+00:00",
        confirmed_by="Tomas Reid",
        confirmed_at="2026-10-05T09:30:00+00:00",
        handoff_format="pain001",
        handoff_batch_booking=True,
        handoff_generated_at=SENT,
        handoff_key="payment-runs/RUN-EUR-1/handoff.pain001.xml",
        handoff_creditor_entries=[
            HandoffCreditorEntry(
                invoice_id="INV-1001",
                source="supplier_master",
                supplier_name="Northbeam Office Supplies",
                execution_date="2026-10-20",
            ),
            HandoffCreditorEntry(
                invoice_id="INV-1008",
                source="supplier_master",
                supplier_name="Northbeam Office Supplies",
                execution_date="2026-10-21",
            ),
        ],
    )


def _receipt(date_: str, status: str, **kw) -> HandoffGroupReceipt:
    return HandoffGroupReceipt(
        pmt_inf_id=f"RUN-EUR-1-{date_}",
        execution_date=date_,
        status=status,
        member_invoice_ids=kw.pop("members", []),
        recorded_by="Tomas Reid",
        recorded_at="2026-10-13T09:00:00+00:00",
        **kw,
    )


# -- the pure ageing read ---------------------------------------------------------


def test_unanswered_groups_age_onto_the_list():
    items = unanswered_group_items(
        [_batch_run()], as_of=date(2026, 10, 20), aging_days=5
    )
    assert [item.execution_date for item in items] == [
        "2026-10-20",
        "2026-10-21",
    ]
    first = items[0]
    assert first.run_id == "RUN-EUR-1"
    assert first.pmt_inf_id == "RUN-EUR-1-2026-10-20"
    assert first.member_invoice_ids == ["INV-1001"]
    assert first.member_count == 1
    assert first.currency == "EUR"
    assert first.group_value == 1180.00
    assert first.waiting_days == 9
    assert first.aging_threshold_days == 5
    assert first.reexported is False


def test_a_group_inside_its_threshold_is_just_a_bank_being_a_bank():
    assert (
        unanswered_group_items(
            [_batch_run()], as_of=date(2026, 10, 16), aging_days=5
        )
        == []
    )


def test_answered_groups_never_age():
    run = _batch_run()
    run.handoff_group_receipts = [
        _receipt("2026-10-20", "accepted", members=["INV-1001"]),
        _receipt(
            "2026-10-21",
            "rejected",
            detail="creditor account closed",
            members=["INV-1008"],
        ),
    ]
    # Accepted closes it; rejected is the re-export's work.
    assert (
        unanswered_group_items(
            [run], as_of=date(2026, 11, 20), aging_days=5
        )
        == []
    )


def test_a_reexported_group_ages_from_its_solo_file():
    run = _batch_run()
    rejected = _receipt(
        "2026-10-21",
        "rejected",
        detail="creditor account closed",
        members=["INV-1008"],
        superseded=True,
    )
    run.handoff_group_receipts = [rejected]
    run.handoff_group_reexports = [
        HandoffGroupReexport(
            pmt_inf_id="RUN-EUR-1-2026-10-21",
            execution_date="2026-10-21",
            reexport_execution_date="2026-10-21",
            rejection_reason="creditor account closed",
            member_invoice_ids=["INV-1008"],
            reexported_by="Tomas Reid",
            reexported_at="2026-10-18T09:00:00+00:00",
        )
    ]
    assert group_file_sent_at(run, "2026-10-21") == "2026-10-18T09:00:00+00:00"
    # Four days after the re-export: inside the threshold, even
    # though the original file is eleven days old.
    items = unanswered_group_items(
        [run], as_of=date(2026, 10, 22), aging_days=5
    )
    assert [item.execution_date for item in items] == ["2026-10-20"]
    # Eight days after the re-export: the silence is about the
    # corrected file, and the item says so.
    items = unanswered_group_items(
        [run], as_of=date(2026, 10, 26), aging_days=5
    )
    by_date = {item.execution_date: item for item in items}
    assert by_date["2026-10-21"].reexported is True
    assert by_date["2026-10-21"].waiting_days == 8
    assert by_date["2026-10-20"].waiting_days == 15


def test_a_flat_export_has_no_groups_to_age():
    run = _batch_run()
    run.handoff_batch_booking = False
    assert (
        unanswered_group_items(
            [run], as_of=date(2026, 11, 20), aging_days=5
        )
        == []
    )


# -- the queue, the digest, the metrics ---------------------------------------------


def _service_with(run: PaymentRun) -> InvoiceService:
    store = InMemoryStore()
    store.save_payment_run(run)
    return InvoiceService(store=store, checkpointer=False)


def test_the_queue_carries_the_group_waits():
    svc = _service_with(_batch_run())
    queue = svc.queue_view(as_of="2026-10-20")
    assert queue.group_receipt_wait_count == 2
    assert queue.group_receipt_aging_days == group_receipt_aging_days()
    assert {
        item.execution_date for item in queue.group_receipt_waits
    } == {"2026-10-20", "2026-10-21"}


def test_the_digest_announces_bank_silence():
    svc = _service_with(_batch_run())
    digest = svc.escalation_digest(as_of="2026-10-20")
    assert digest.group_receipt_wait_count == 2
    assert "Bank silence" in digest.summary
    assert "INV-1001" in digest.summary
    assert "INV-1008" in digest.summary
    assert "EUR 1,180.00" in digest.summary
    assert "All clear" not in digest.summary


def test_a_quiet_bank_stays_out_of_the_digest():
    svc = _service_with(_batch_run())
    digest = svc.escalation_digest(as_of="2026-10-13")
    assert digest.group_receipt_wait_count == 0
    assert "Bank silence" not in digest.summary


def test_the_metrics_count_the_feedback_loops_coverage():
    run = _batch_run()
    metrics = group_answer_metrics([run])
    assert metrics == {
        "groups_total": 2,
        "accepted": 0,
        "rejected": 0,
        "unanswered": 2,
        "reexported_groups": 0,
        "reexports": 0,
    }
    run.handoff_group_receipts = [
        _receipt("2026-10-20", "accepted", members=["INV-1001"]),
        _receipt(
            "2026-10-21",
            "rejected",
            detail="creditor account closed",
            members=["INV-1008"],
            superseded=True,
        ),
    ]
    run.handoff_group_reexports = [
        HandoffGroupReexport(
            pmt_inf_id="RUN-EUR-1-2026-10-21",
            execution_date="2026-10-21",
            member_invoice_ids=["INV-1008"],
            reexported_at="2026-10-18T09:00:00+00:00",
        )
    ]
    metrics = group_answer_metrics([run])
    assert metrics["accepted"] == 1
    # The rejected receipt stands superseded — the group reads
    # unanswered against its new file, and the re-export counts.
    assert metrics["rejected"] == 0
    assert metrics["unanswered"] == 1
    assert metrics["reexported_groups"] == 1
    assert metrics["reexports"] == 1
    svc = _service_with(run)
    section = svc.metrics_snapshot()["payment_runs"]["group_answers"]
    assert section == metrics


def test_the_aging_threshold_is_configurable(monkeypatch):
    monkeypatch.setenv("GROUP_RECEIPT_AGING_DAYS", "2")
    assert group_receipt_aging_days() == 2
    svc = _service_with(_batch_run())
    queue = svc.queue_view(as_of="2026-10-14")
    assert queue.group_receipt_aging_days == 2
    assert queue.group_receipt_wait_count == 2
