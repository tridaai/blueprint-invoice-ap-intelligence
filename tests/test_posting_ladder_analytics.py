"""Escalation-ladder analytics: which rung the posting followed.

The posting ladder's ledger records that a crossing was
pushed — rung 1 past the threshold, rung 2 past twice it —
but a firing is not an effect. This grading reads each
escalated episode against the record as it stands: a posted
receipt after rung 1 (and before rung 2) credits rung 1; one
after rung 2 credits rung 2, because rung 1 already had its
window; an episode closed by a failure that started a newer
episode, or posted without following the push, credits
neither; and an episode with no resolution stands open,
counted. The two waiting loops — the supplier's information
waits and the ERP's postings — now carry the same grading in
one /metrics read.
"""

from __future__ import annotations

from ap_agent.posting import posting_ladder_analytics
from ap_agent.schemas import (
    AgentResult,
    PostingEscalation,
    PostingReceipt,
)
from ap_agent.service import InvoiceService
from ap_agent.store import ApprovalRecord, InMemoryStore


def _record(invoice_id: str, escalations=(), receipt=None) -> ApprovalRecord:
    result = AgentResult(
        invoice_id=invoice_id,
        status="approved",
        posting_escalations=list(escalations),
        posting_receipt=receipt,
    )
    return ApprovalRecord(result=result, invoice={"invoice_id": invoice_id})


def _rung(level: int, dispatched_at: str, state: str = "none",
          since: str = "2026-09-20") -> PostingEscalation:
    return PostingEscalation(
        receipt_state=state,
        episode_since=since,
        escalation_days=21 if level == 1 else 42,
        level=level,
        dispatched_at=dispatched_at,
        dispatch_status="sent",
    )


def _posted(recorded_at: str) -> PostingReceipt:
    return PostingReceipt(
        invoice_id="INV-X",
        status="posted",
        external_reference="ERP-1",
        recorded_by="Tomas Reid",
        recorded_at=recorded_at,
    )


def _failed(recorded_at: str) -> PostingReceipt:
    return PostingReceipt(
        invoice_id="INV-X",
        status="failed",
        detail="cost centre closed",
        recorded_by="Tomas Reid",
        recorded_at=recorded_at,
    )


# -- the grading ---------------------------------------------------------------------


def test_rung_one_resolves_when_the_posting_follows_it():
    record = _record(
        "INV-1",
        escalations=[_rung(1, "2026-10-11T09:00:00+00:00")],
        receipt=_posted("2026-10-14T09:00:00+00:00"),
    )
    analytics = posting_ladder_analytics([record])
    assert analytics["episodes_graded"] == 1
    assert analytics["firings"] == 1
    assert analytics["firings_by_rung"] == {"1": 1, "2": 0}
    assert analytics["resolved_after_rung_1"] == 1
    assert analytics["resolved_after_rung_2"] == 0
    assert analytics["resolution_rate"] == 1.0
    assert analytics["days_to_post"] == {
        "count": 1,
        "median": 3,
        "mean": 3.0,
        "max": 3,
    }


def test_a_posting_after_rung_two_credits_rung_two_not_rung_one():
    record = _record(
        "INV-2",
        escalations=[
            _rung(1, "2026-10-11T09:00:00+00:00"),
            _rung(2, "2026-11-01T09:00:00+00:00"),
        ],
        receipt=_posted("2026-11-04T09:00:00+00:00"),
    )
    analytics = posting_ladder_analytics([record])
    assert analytics["firings_by_rung"] == {"1": 1, "2": 1}
    assert analytics["resolved_after_rung_1"] == 0
    assert analytics["resolved_after_rung_2"] == 1
    # Days run from the resolving rung, not the first shout.
    assert analytics["days_to_post"]["median"] == 3


def test_an_unanswered_ladder_stands_open_and_names_the_invoice():
    record = _record(
        "INV-3",
        escalations=[
            _rung(1, "2026-10-11T09:00:00+00:00"),
            _rung(2, "2026-11-01T09:00:00+00:00"),
        ],
    )
    analytics = posting_ladder_analytics([record])
    assert analytics["still_open"] == 1
    assert analytics["still_open_invoice_ids"] == ["INV-3"]
    assert analytics["resolution_rate"] == 0.0


def test_a_failed_receipt_closes_the_old_episode_without_crediting_it():
    record = _record(
        "INV-4",
        escalations=[
            _rung(1, "2026-10-11T09:00:00+00:00", state="none",
                  since="2026-09-20"),
            _rung(1, "2026-11-05T09:00:00+00:00", state="failed",
                  since="2026-10-15"),
        ],
        receipt=_failed("2026-10-15T09:00:00+00:00"),
    )
    analytics = posting_ladder_analytics([record])
    assert analytics["episodes_graded"] == 2
    # Episode one ended when the failure started episode two —
    # closed, but not by the ladder. Episode two stands open.
    assert analytics["closed_otherwise"] == 1
    assert analytics["still_open"] == 1
    assert analytics["resolved_after_rung_1"] == 0


def test_a_posting_that_predates_the_push_credits_no_rung():
    record = _record(
        "INV-5",
        escalations=[_rung(1, "2026-10-11T09:00:00+00:00")],
        receipt=_posted("2026-10-05T09:00:00+00:00"),
    )
    analytics = posting_ladder_analytics([record])
    assert analytics["resolved_after_rung_1"] == 0
    assert analytics["closed_otherwise"] == 1
    assert analytics["days_to_post"] is None


def test_records_without_a_ladder_grade_nothing():
    analytics = posting_ladder_analytics([_record("INV-6")])
    assert analytics["episodes_graded"] == 0
    assert analytics["firings"] == 0
    assert analytics["resolution_rate"] is None
    assert analytics["still_open_invoice_ids"] == []


# -- the metrics wiring ---------------------------------------------------------------


def test_the_ladder_rides_metrics_beside_the_information_follow_through():
    store = InMemoryStore()
    store.save(
        _record(
            "INV-1",
            escalations=[_rung(1, "2026-10-11T09:00:00+00:00")],
            receipt=_posted("2026-10-14T09:00:00+00:00"),
        )
    )
    svc = InvoiceService(store=store, checkpointer=False)
    metrics = svc.metrics_snapshot()
    ladder = metrics["posting_open_items"]["escalation_ladder"]
    assert ladder["resolved_after_rung_1"] == 1
    # The other ladder's grading sits in the same read.
    assert "escalation_follow_through" in metrics["information_waits"]
