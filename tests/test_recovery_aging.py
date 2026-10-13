"""Recovery-case aging: open duplicate-payment cases age in the
queue and metrics — total age, per-status age, age buckets, and an
SLA flag when a case sits in one status too long (money leaking
while it sits). Closed cases drop out; moving a case resets its
status clock."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.queue import recovery_case_aging
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import RecoveryCaseEvent, RecoveryPack
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
AS_OF = date(2026, 10, 10)


def _pack(pack_id: str, created: str, events: list[tuple[str, str]], status: str):
    return RecoveryPack(
        pack_id=pack_id,
        duplicate_invoice_id=f"INV-{pack_id[-4:]}",
        supplier_name="Northbeam Office Supplies",
        amount=1180.0,
        created_at=created,
        case_status=status,
        case_history=[
            RecoveryCaseEvent(status=s, actor="operator", at=at) for s, at in events
        ],
    )


# A: opened Sep 1, contacted Sep 20 — 39 days old, 20 days stuck.
PACK_A = _pack(
    "REC-INV-9001",
    "2026-09-01T09:00:00+00:00",
    [("opened", "2026-09-01T09:00:00+00:00"), ("supplier_contacted", "2026-09-20T09:00:00+00:00")],
    "supplier_contacted",
)
# B: opened Oct 1, never contacted — 9 days old, 9 days stuck.
PACK_B = _pack(
    "REC-INV-9002",
    "2026-10-01T09:00:00+00:00",
    [("opened", "2026-10-01T09:00:00+00:00")],
    "opened",
)
# C: closed — done, drops out of the aging view entirely.
PACK_C = _pack(
    "REC-INV-9003",
    "2026-08-01T09:00:00+00:00",
    [("opened", "2026-08-01T09:00:00+00:00"), ("closed", "2026-08-20T09:00:00+00:00")],
    "closed",
)
# D: opened Oct 9 — fresh.
PACK_D = _pack(
    "REC-INV-9004",
    "2026-10-09T09:00:00+00:00",
    [("opened", "2026-10-09T09:00:00+00:00")],
    "opened",
)


def test_aging_buckets_status_age_and_sla():
    items, buckets, breaches = recovery_case_aging(
        [PACK_B, PACK_C, PACK_D, PACK_A], as_of=AS_OF, sla_days=14
    )
    assert [i.pack_id for i in items] == ["REC-INV-9001", "REC-INV-9002", "REC-INV-9004"]
    by_id = {i.pack_id: i for i in items}
    a = by_id["REC-INV-9001"]
    assert (a.age_days, a.status_age_days) == (39, 20)
    assert a.bucket == "age_30_plus" and a.sla_breached is True
    assert a.case_status == "supplier_contacted"  # the step it is stuck on
    b = by_id["REC-INV-9002"]
    assert (b.age_days, b.status_age_days) == (9, 9)
    assert b.bucket == "age_8_30" and b.sla_breached is False
    d = by_id["REC-INV-9004"]
    assert d.bucket == "age_0_7" and d.sla_breached is False
    assert buckets == {"age_30_plus": 1, "age_8_30": 1, "age_0_7": 1}
    assert breaches == 1


def test_sla_is_on_status_age_not_total_age():
    # A long-lived case that moved yesterday is moving, not stuck.
    moved = _pack(
        "REC-INV-9005",
        "2026-08-01T09:00:00+00:00",
        [("opened", "2026-08-01T09:00:00+00:00"), ("credit_received", "2026-10-09T09:00:00+00:00")],
        "credit_received",
    )
    items, _, breaches = recovery_case_aging([moved], as_of=AS_OF, sla_days=14)
    assert items[0].age_days == 70
    assert items[0].status_age_days == 1
    assert items[0].sla_breached is False and breaches == 0
    # A wider SLA also clears pack A (20 days stuck < 25).
    _, _, breaches = recovery_case_aging([PACK_A], as_of=AS_OF, sla_days=25)
    assert breaches == 0


def test_queue_view_carries_the_case_aging(monkeypatch):
    store = InMemoryStore()
    for pack in (PACK_A, PACK_B, PACK_C, PACK_D):
        store.save_recovery_pack(pack)
    svc = InvoiceService(store=store, checkpointer=False)
    monkeypatch.delenv("RECOVERY_CASE_SLA_DAYS", raising=False)
    view = svc.queue_view(as_of="2026-10-10")
    assert view.recovery_open_count == 3
    assert view.recovery_sla_breaches == 1
    assert view.recovery_sla_days == 14
    assert view.recovery_buckets == {"age_30_plus": 1, "age_8_30": 1, "age_0_7": 1}
    monkeypatch.setenv("RECOVERY_CASE_SLA_DAYS", "25")
    view = svc.queue_view(as_of="2026-10-10")
    assert view.recovery_sla_days == 25
    assert view.recovery_sla_breaches == 0


def test_real_case_ages_from_its_draft(service=None):
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="approved in test")
    svc.analyze(SAMPLES["INV-1013"])
    svc.draft_duplicate_recovery("INV-1013")
    future = (date.today() + timedelta(days=40)).isoformat()
    view = svc.queue_view(as_of=future)
    assert view.recovery_open_count == 1
    case = view.recovery_cases[0]
    assert case.pack_id == "REC-INV-1013"
    assert case.age_days == 40 and case.status_age_days == 40
    assert case.bucket == "age_30_plus" and case.sla_breached is True
    metrics = svc.metrics_snapshot()["recovery_cases"]
    assert metrics["total"] == 1 and metrics["open"] == 1
    assert metrics["by_status"] == {"opened": 1}
    assert metrics["sla_breaches"] == 0  # as of today it is fresh
    assert metrics["age_buckets"] == {"age_0_7": 1}


def test_queue_endpoint_includes_recovery_aging():
    store = InMemoryStore()
    store.save_recovery_pack(PACK_A)
    svc = InvoiceService(store=store, checkpointer=False)
    client = TestClient(create_app(svc))
    body = client.get("/queue", params={"as_of": "2026-10-10"}).json()
    assert body["recovery_open_count"] == 1
    assert body["recovery_sla_breaches"] == 1
    assert body["recovery_cases"][0]["case_status"] == "supplier_contacted"
    assert body["recovery_cases"][0]["status_age_days"] == 20
