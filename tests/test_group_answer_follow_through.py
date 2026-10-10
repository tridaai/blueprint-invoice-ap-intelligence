"""Bank-silence follow-through: the aged group waits, graded.

The ageing read names the batch groups the bank has not
answered; the receipts name the answers that came. Neither
said whether the ageing itself did anything — a group could
cross the threshold, sit on the queue, and the bank's late
answer (or continued silence) read exactly like a group that
answered on day two. This grading reads each (group, file)
episode that stayed unanswered past the threshold against
what happened next: a late answer (accepted / rejected), a
re-presentation (the late answer was a rejection and the
group went out alone), or silence still standing — the
information-wait escalation grading's shape, applied to the
bank loop and carried in /metrics beside the coverage counts.
"""

from __future__ import annotations

from datetime import date

from ap_agent.paymentruns import group_answer_follow_through, handoff_metrics
from ap_agent.schemas import (
    HandoffCreditorEntry,
    HandoffGroupReceipt,
    HandoffGroupReexport,
    PaymentRun,
    PaymentRunLine,
)
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SENT = "2026-10-11T09:00:00+00:00"
GROUP_A = "2026-10-20"  # INV-1001
GROUP_B = "2026-10-21"  # INV-1008


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
                due_date=GROUP_A,
                gross_amount=1180.00,
                net_amount=1180.00,
            ),
            PaymentRunLine(
                invoice_id="INV-1008",
                supplier_name="Northbeam Office Supplies",
                due_date=GROUP_B,
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
                execution_date=GROUP_A,
            ),
            HandoffCreditorEntry(
                invoice_id="INV-1008",
                source="supplier_master",
                supplier_name="Northbeam Office Supplies",
                execution_date=GROUP_B,
            ),
        ],
    )


def _receipt(group: str, status: str, recorded_at: str, **kw) -> HandoffGroupReceipt:
    return HandoffGroupReceipt(
        pmt_inf_id=f"RUN-EUR-1-{group}",
        execution_date=group,
        status=status,
        detail=kw.pop("detail", ""),
        member_invoice_ids=kw.pop("members", []),
        recorded_by="Tomas Reid",
        recorded_at=recorded_at,
        **kw,
    )


def _reexport(group: str, reexported_at: str, reason: str = "account closed") -> HandoffGroupReexport:
    return HandoffGroupReexport(
        pmt_inf_id=f"RUN-EUR-1-{group}",
        execution_date=group,
        reexport_execution_date=group,
        rejection_reason=reason,
        member_invoice_ids=["INV-1001"] if group == GROUP_A else ["INV-1008"],
        reexported_by="Tomas Reid",
        reexported_at=reexported_at,
    )


def _grade(run: PaymentRun, as_of: date, aging_days: int = 5) -> dict:
    return group_answer_follow_through([run], as_of=as_of, aging_days=aging_days)


# -- the grading -------------------------------------------------------------------


def test_a_late_acceptance_grades_answered_and_the_rest_still_silent():
    run = _batch_run()
    run.handoff_group_receipts = [
        _receipt(GROUP_A, "accepted", "2026-10-20T09:00:00+00:00")
    ]
    grade = _grade(run, date(2026, 10, 20))
    assert grade["aging_threshold_days"] == 5
    assert grade["episodes_graded"] == 2
    assert grade["answered_after_ageing"] == 1
    assert grade["answered_accepted"] == 1
    assert grade["answered_rejected"] == 0
    assert grade["reexported_after_ageing"] == 0
    assert grade["still_silent"] == 1
    assert grade["still_silent_groups"] == [
        {"run_id": "RUN-EUR-1", "execution_date": GROUP_B}
    ]
    assert grade["silence_ended_rate"] == 0.5
    # Nine days from the file going out to the answer landing.
    assert grade["days_to_answer"] == {
        "count": 1,
        "median": 9,
        "mean": 9.0,
        "max": 9,
    }


def test_an_answer_inside_the_threshold_never_aged_and_is_not_graded():
    run = _batch_run()
    run.handoff_group_receipts = [
        _receipt(GROUP_A, "accepted", "2026-10-13T09:00:00+00:00"),
        _receipt(GROUP_B, "rejected", "2026-10-14T09:00:00+00:00", detail="bad date"),
    ]
    grade = _grade(run, date(2026, 10, 15))
    assert grade["episodes_graded"] == 0
    assert grade["answered_after_ageing"] == 0
    assert grade["still_silent"] == 0
    assert grade["silence_ended_rate"] is None
    assert grade["days_to_answer"] is None


def test_a_late_rejection_without_reexport_grades_answered_rejected():
    run = _batch_run()
    run.handoff_group_receipts = [
        _receipt(GROUP_A, "rejected", "2026-10-19T09:00:00+00:00", detail="limit")
    ]
    grade = _grade(run, date(2026, 10, 19))
    assert grade["answered_after_ageing"] == 1
    assert grade["answered_rejected"] == 1
    assert grade["reexported_after_ageing"] == 0


def test_a_late_rejection_then_reexport_grades_reexported_not_answered():
    run = _batch_run()
    run.handoff_group_receipts = [
        _receipt(
            GROUP_A,
            "rejected",
            "2026-10-19T09:00:00+00:00",
            detail="account closed",
            superseded=True,
        ),
        _receipt(GROUP_B, "accepted", "2026-10-12T09:00:00+00:00"),
    ]
    run.handoff_group_reexports = [
        _reexport(GROUP_A, "2026-10-19T15:00:00+00:00")
    ]
    grade = _grade(run, date(2026, 10, 21))
    # The whole-file episode aged (8 days) and ended in a
    # re-presentation; the solo file is two days old — inside
    # its threshold, not graded.
    assert grade["episodes_graded"] == 1
    assert grade["reexported_after_ageing"] == 1
    assert grade["answered_after_ageing"] == 0
    assert grade["days_to_answer"]["count"] == 1
    assert grade["days_to_answer"]["median"] == 8


def test_a_group_can_age_twice_once_per_file():
    run = _batch_run()
    run.handoff_group_receipts = [
        _receipt(
            GROUP_A,
            "rejected",
            "2026-10-19T09:00:00+00:00",
            detail="account closed",
            superseded=True,
        ),
        _receipt(GROUP_B, "accepted", "2026-10-12T09:00:00+00:00"),
        _receipt(
            GROUP_A,
            "rejected",
            "2026-10-29T09:00:00+00:00",
            detail="account still closed",
        ),
    ]
    run.handoff_group_reexports = [
        _reexport(GROUP_A, "2026-10-19T15:00:00+00:00")
    ]
    grade = _grade(run, date(2026, 10, 29))
    # Episode one: aged, rejection, re-presented. Episode two:
    # aged (10 days), rejection, no further file — answered.
    assert grade["episodes_graded"] == 2
    assert grade["reexported_after_ageing"] == 1
    assert grade["answered_rejected"] == 1
    assert grade["silence_ended_rate"] == 1.0
    assert grade["days_to_answer"]["count"] == 2
    assert grade["days_to_answer"]["max"] == 10


def test_a_solo_file_left_silent_ages_in_its_own_episode():
    run = _batch_run()
    run.handoff_group_receipts = [
        _receipt(
            GROUP_A,
            "rejected",
            "2026-10-12T09:00:00+00:00",
            detail="account closed",
            superseded=True,
        ),
        _receipt(GROUP_B, "accepted", "2026-10-12T09:00:00+00:00"),
    ]
    run.handoff_group_reexports = [
        _reexport(GROUP_A, "2026-10-12T15:00:00+00:00")
    ]
    grade = _grade(run, date(2026, 10, 25))
    # The first file was answered in a day — never aged. The
    # solo file has stood silent for 13 days: still silent.
    assert grade["episodes_graded"] == 1
    assert grade["still_silent"] == 1
    assert grade["still_silent_groups"] == [
        {"run_id": "RUN-EUR-1", "execution_date": GROUP_A}
    ]


def test_non_batch_runs_grade_nothing():
    run = _batch_run()
    run.handoff_batch_booking = False
    grade = _grade(run, date(2026, 11, 1))
    assert grade["episodes_graded"] == 0
    assert grade["still_silent_groups"] == []


# -- the metrics wiring -------------------------------------------------------------


def test_the_grading_rides_the_handoff_metrics_beside_the_coverage():
    run = _batch_run()
    run.handoff_group_receipts = [
        _receipt(GROUP_A, "accepted", "2026-10-20T09:00:00+00:00")
    ]
    section = handoff_metrics(
        [run], as_of=date(2026, 10, 20), group_aging_days=5
    )
    assert section["group_answers"]["accepted"] == 1
    grade = section["group_answer_follow_through"]
    assert grade["answered_after_ageing"] == 1
    assert grade["still_silent"] == 1


def test_the_service_metrics_carry_the_grading_with_the_configured_threshold():
    store = InMemoryStore()
    store.save_payment_run(_batch_run())
    svc = InvoiceService(store=store, checkpointer=False)
    grade = svc.metrics_snapshot()["payment_runs"][
        "group_answer_follow_through"
    ]
    assert grade["aging_threshold_days"] == 5  # GROUP_RECEIPT_AGING_DAYS default
    assert set(grade) >= {
        "episodes_graded",
        "answered_after_ageing",
        "reexported_after_ageing",
        "still_silent",
        "days_to_answer",
    }
