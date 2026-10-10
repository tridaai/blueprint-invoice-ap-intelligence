"""Wait-escalation follow-through: the push is graded, not just
recorded.

Round 17's sweep fires the signed ``information_wait_escalation``
webhook when a wait outlasts its second window, and the ledger
on the record says it fired. Nothing read whether it WORKED.
The follow-through grades each firing against the record as it
now stands: a reply that arrived after the dispatch (the chase
landed — whole or partial, either is the supplier moving), a
wait closed without one (decided at the gate; the push claims
no credit), and a wait still open exactly as it was (a chase
failing in public, counted, with the invoice named). Answered
episodes carry their days-to-answer: the push's price in days.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ap_agent.matching import _parse_date
from ap_agent.queue import information_escalation_follow_through
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import (
    AgentResult,
    InformationEscalation,
    InformationRequest,
    Triage,
)
from ap_agent.service import InvoiceService
from ap_agent.store import ApprovalRecord, InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


@pytest.fixture()
def service() -> InvoiceService:
    return InvoiceService(store=InMemoryStore(), checkpointer=False)


def _plus(iso: str, days: int) -> str:
    return (_parse_date(iso) + timedelta(days=days)).isoformat()


def _escalate_inv_1005(service: InvoiceService, monkeypatch) -> object:
    """Analyse INV-1005 (blocked: no PO), age its wait past the
    escalation threshold with the webhook on, and fire."""
    result = service.analyze(SAMPLES["INV-1005"])
    assert result.needs_information
    requested = result.information_request.requested_at
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    monkeypatch.setenv("ACTION_WEBHOOK_SECRET", "test-secret")
    escalated = service.escalate_information_waits(as_of=_plus(requested, 7))
    assert [item.invoice_id for item in escalated] == ["INV-1005"]
    return service._get_store().get("INV-1005").result


# -- the pure grade, over fabricated ledgers ---------------------------------


def _record(
    invoice_id: str,
    *,
    status: str = "awaiting_approver",
    needs_information: bool = True,
    reply_status: str = "awaiting_reply",
    reply_received_at: str = "",
    dispatched_at: str = "2026-10-01T09:00:00+00:00",
    with_request: bool = True,
) -> ApprovalRecord:
    result = AgentResult(
        invoice_id=invoice_id,
        status=status,
        triage=Triage(outcome="missing_po"),
        needs_information=needs_information,
        information_request=(
            InformationRequest(
                reply_status=reply_status,
                reply_received_at=reply_received_at,
            )
            if with_request
            else None
        ),
        information_escalations=[
            InformationEscalation(
                episode_since="2026-09-25",
                escalation_days=6,
                dispatched_at=dispatched_at,
                dispatch_status="sent",
            )
        ],
    )
    return ApprovalRecord(result=result, invoice={"invoice_id": invoice_id})


def test_no_escalations_grade_as_an_honest_empty():
    block = information_escalation_follow_through([])
    assert block["escalations_graded"] == 0
    assert block["answer_rate"] is None
    assert block["days_to_answer"] is None


def test_a_reply_after_the_push_is_answered():
    record = _record(
        "INV-A",
        reply_status="answered",
        reply_received_at="2026-10-04T10:00:00+00:00",
    )
    block = information_escalation_follow_through([record])
    assert block["answered_after_push"] == 1
    assert block["answer_rate"] == 1.0
    # Dispatch 2026-10-01 → reply 2026-10-04: three days.
    assert block["days_to_answer"] == {
        "count": 1,
        "median": 3,
        "mean": 3.0,
        "max": 3,
    }


def test_a_partial_reply_after_the_push_still_counts_as_answered():
    # The episode moved: the supplier answered, the ball rolled.
    record = _record(
        "INV-A",
        reply_status="partially_answered",
        reply_received_at="2026-10-02T10:00:00+00:00",
    )
    block = information_escalation_follow_through([record])
    assert block["answered_after_push"] == 1
    assert block["still_open"] == 0


def test_a_reply_before_the_push_is_not_the_pushs_doing():
    record = _record(
        "INV-A",
        reply_status="partially_answered",
        reply_received_at="2026-09-30T10:00:00+00:00",
    )
    block = information_escalation_follow_through([record])
    assert block["answered_after_push"] == 0
    assert block["still_open"] == 1
    assert block["still_open_invoice_ids"] == ["INV-A"]


def test_a_decided_case_closed_otherwise():
    record = _record("INV-A", status="rejected", needs_information=False)
    block = information_escalation_follow_through([record])
    assert block["closed_otherwise"] == 1
    assert block["answered_after_push"] == 0
    assert block["still_open"] == 0


def test_mixed_population_aggregates(monkeypatch):
    records = [
        _record(
            "INV-A",
            reply_status="answered",
            reply_received_at="2026-10-03T10:00:00+00:00",
        ),
        _record("INV-B"),  # untouched since the push
        _record("INV-C", status="approved", needs_information=False),
        _record(
            "INV-D",
            reply_status="answered",
            reply_received_at="2026-10-06T10:00:00+00:00",
        ),
    ]
    block = information_escalation_follow_through(records)
    assert block["escalations_graded"] == 4
    assert block["answered_after_push"] == 2
    assert block["still_open"] == 1
    assert block["closed_otherwise"] == 1
    assert block["answer_rate"] == 0.5
    # Answers at 2 and 5 days: the median of an even pair averages.
    assert block["days_to_answer"]["count"] == 2
    assert block["days_to_answer"]["median"] == 3.5
    assert block["days_to_answer"]["max"] == 5


# -- end to end, over the real sweep -----------------------------------------


def test_metrics_grade_an_unanswered_push_as_still_open(service, monkeypatch):
    _escalate_inv_1005(service, monkeypatch)
    block = service.metrics_snapshot()["information_waits"][
        "escalation_follow_through"
    ]
    assert block["escalations_graded"] == 1
    assert block["still_open"] == 1
    assert block["still_open_invoice_ids"] == ["INV-1005"]
    assert block["answer_rate"] == 0.0


def test_metrics_grade_a_later_reply_as_answered(service, monkeypatch):
    _escalate_inv_1005(service, monkeypatch)
    after = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat(
        timespec="seconds"
    )
    service.record_information_reply(
        "INV-1005",
        actor="Sam (AP)",
        supplied_items=["purchase_order"],
        received_at=after,
    )
    block = service.metrics_snapshot()["information_waits"][
        "escalation_follow_through"
    ]
    assert block["answered_after_push"] == 1
    assert block["still_open"] == 0
    assert block["answer_rate"] == 1.0
    assert block["days_to_answer"]["count"] == 1
    assert block["days_to_answer"]["max"] >= 1


def test_metrics_grade_a_decision_without_reply_as_closed_otherwise(
    service, monkeypatch
):
    _escalate_inv_1005(service, monkeypatch)
    service.reject("INV-1005", reviewer="Ruth (AP lead)", reason="not ours")
    block = service.metrics_snapshot()["information_waits"][
        "escalation_follow_through"
    ]
    assert block["closed_otherwise"] == 1
    assert block["answered_after_push"] == 0
    assert block["still_open"] == 0
