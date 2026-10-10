"""The escalation digest: the queue is a pull surface, but
escalations deserve a push counterpart — and this agent sends
nothing, so the digest is composed deterministically from the
stored records and left for a human to read or forward. One
summary for the day: recovery cases past their SLA, escalated
open postings, invoices stuck at the gate, four-eyes stalls, and
the day's run news — discounts forecast to lapse on runs
confirmed today, drift realised on runs exported today."""

from __future__ import annotations

from datetime import date, timedelta

from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.digest import build_digest
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import (
    FourEyesAgingItem,
    OpenPostingItem,
    PaymentRun,
    PaymentRunDrift,
    PaymentRunForecast,
    QueueItem,
    QueueView,
    RecoveryCaseAgingItem,
    RecoveryCaseEvent,
    RecoveryPack,
)
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
AS_OF = date(2026, 11, 15)


def _pack() -> RecoveryPack:
    # Opened Sep 1, contacted Sep 20: at the digest date the
    # current status is 56 days old against a 14-day SLA.
    return RecoveryPack(
        pack_id="REC-INV-9001",
        duplicate_invoice_id="INV-9001",
        supplier_name="Northbeam Office Supplies",
        amount=1180.0,
        created_at="2026-09-01T09:00:00+00:00",
        case_status="supplier_contacted",
        case_history=[
            RecoveryCaseEvent(status="opened", actor="op", at="2026-09-01T09:00:00+00:00"),
            RecoveryCaseEvent(status="supplier_contacted", actor="op", at="2026-09-20T09:00:00+00:00"),
        ],
    )


def _forecast_run() -> PaymentRun:
    return PaymentRun(
        run_id="RUN-USD-FORECAST",
        currency="USD",
        status="confirmed",
        confirmed_by="Tomas Reid",
        confirmed_at="2026-11-15T09:00:00+00:00",
        drift_forecast=[
            PaymentRunForecast(
                invoice_id="INV-1001",
                supplier_name="Northbeam Office Supplies",
                discount_amount=23.60,
                discount_deadline="2026-11-18",
                expected_export_date="2026-11-21",
            )
        ],
        drift_forecast_basis="forecast from the measured proposal→export lag",
    )


def _drifted_run() -> PaymentRun:
    return PaymentRun(
        run_id="RUN-USD-DRIFTED",
        currency="USD",
        status="confirmed",
        confirmed_at="2026-11-10T09:00:00+00:00",
        handoff_generated_at="2026-11-15T10:00:00+00:00",
        handoff_format="csv",
        handoff_drift=[
            PaymentRunDrift(
                invoice_id="INV-1003",
                discount_deadline="2026-11-12",
                discount_at_proposal=14.16,
                discount_at_export=0.0,
                discount_lost=14.16,
                reason="window closed",
            )
        ],
        handoff_drift_total=14.16,
    )


def _four_eyes_run() -> PaymentRun:
    return PaymentRun(
        run_id="RUN-USD-FOUREYES",
        currency="USD",
        status="confirmed",
        total_net=30000.0,
        four_eyes_required=True,
        four_eyes_threshold=25000.0,
        confirmed_by="Priya Nair",
        confirmed_at="2026-11-05T09:00:00+00:00",
    )


def test_pure_composition_names_every_section():
    queue = QueueView(
        as_of=AS_OF.isoformat(),
        recovery_sla_days=14,
        posting_escalation_days=21,
        four_eyes_aging_days=2,
        recovery_cases=[
            RecoveryCaseAgingItem(
                pack_id="REC-INV-9001",
                duplicate_invoice_id="INV-9001",
                supplier_name="Northbeam Office Supplies",
                amount=1180.0,
                case_status="supplier_contacted",
                age_days=75,
                status_age_days=56,
                sla_breached=True,
                bucket="age_30_plus",
            )
        ],
        recovery_open_count=1,
        recovery_sla_breaches=1,
        open_postings=[
            OpenPostingItem(
                invoice_id="INV-1002",
                supplier_name="Cascade IT Hardware",
                total_amount=1274.40,
                decided_by="Priya Nair",
                age_days=36,
                receipt_state="failed",
                receipt_detail="ERP refused: cost centre blocked",
            )
        ],
        open_posting_count=1,
        items=[
            QueueItem(
                invoice_id="INV-1001",
                supplier_name="Northbeam Office Supplies",
                total_amount=1180.0,
                due_date="2026-10-28",
                days_overdue=18,
                bucket="overdue_8_30",
            ),
            QueueItem(  # at the gate but not yet due — not stuck
                invoice_id="INV-1010",
                supplier_name="Bluepine Catering Co",
                due_date="2026-12-01",
                days_overdue=-16,
                bucket="due_later",
            ),
        ],
        four_eyes_pending=[
            FourEyesAgingItem(
                run_id="RUN-USD-FOUREYES",
                total_net=30000.0,
                four_eyes_threshold=25000.0,
                confirmed_by="Priya Nair",
                confirmed_at="2026-11-05T09:00:00+00:00",
                age_days=10,
            )
        ],
        four_eyes_pending_count=1,
    )
    digest = build_digest(as_of=AS_OF, queue=queue, runs=[_forecast_run(), _drifted_run()])
    assert digest.recovery_breach_count == 1
    assert digest.recovery_amount_by_currency == {"USD": 1180.0}
    assert digest.posting_escalation_count == 1
    assert digest.stuck_invoice_count == 1  # only the overdue one
    assert digest.most_overdue_days == 18
    assert digest.four_eyes_pending_count == 1
    assert [e.invoice_id for e in digest.forecast_lapses] == ["INV-1001"]
    assert digest.forecast_value_by_currency == {"USD": 23.60}
    assert [e.run_id for e in digest.drift_today] == ["RUN-USD-DRIFTED"]
    assert digest.drift_value_by_currency == {"USD": 14.16}
    summary = digest.summary
    assert "the agent sends nothing" in summary
    assert "Recovery cases past their SLA: 1" in summary
    assert "posting failed: ERP refused: cost centre blocked" in summary
    assert "18d overdue" in summary
    assert "confirmed by Priya Nair, waiting 10d" in summary
    assert "Forecast on runs confirmed today: 1" in summary
    assert "Drift realised on runs exported today" in summary
    assert "All clear" not in summary


def test_runs_from_other_days_are_not_the_days_news():
    digest = build_digest(
        as_of=AS_OF + timedelta(days=1),
        queue=QueueView(as_of=(AS_OF + timedelta(days=1)).isoformat()),
        runs=[_forecast_run(), _drifted_run()],
    )
    assert digest.forecast_lapses == []
    assert digest.drift_today == []


def test_a_quiet_day_says_all_clear():
    digest = build_digest(
        as_of=AS_OF, queue=QueueView(as_of=AS_OF.isoformat()), runs=[]
    )
    assert digest.recovery_breach_count == 0
    assert digest.stuck_invoice_count == 0
    assert digest.most_overdue_days is None
    assert "All clear: nothing crossed a line today" in digest.summary


def _service() -> InvoiceService:
    store = InMemoryStore()
    store.save_recovery_pack(_pack())
    store.save_payment_run(_forecast_run())
    store.save_payment_run(_drifted_run())
    store.save_payment_run(_four_eyes_run())
    svc = InvoiceService(store=store, checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])  # left awaiting; due 2026-10-28
    return svc


def test_service_digest_composes_the_whole_day():
    digest = _service().escalation_digest(as_of=AS_OF.isoformat())
    assert digest.as_of == "2026-11-15"
    assert digest.recovery_breach_count == 1
    assert digest.stuck_invoice_count == 1
    assert digest.stuck_items[0].invoice_id == "INV-1001"
    assert digest.four_eyes_pending_count == 1
    assert digest.four_eyes_pending[0].confirmed_by == "Priya Nair"
    assert len(digest.forecast_lapses) == 1
    assert len(digest.drift_today) == 1


def test_digest_rejects_a_garbage_date():
    svc = _service()
    try:
        svc.escalation_digest(as_of="not-a-date")
    except ValueError as exc:
        assert "not an ISO date" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_digest_api():
    client = TestClient(create_app(_service()))
    resp = client.get("/digest", params={"as_of": "2026-11-15"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["as_of"] == "2026-11-15"
    assert body["recovery_breach_count"] == 1
    assert body["four_eyes_pending_count"] == 1
    assert "Escalation digest for 2026-11-15" in body["summary"]
    resp = client.get("/digest", params={"as_of": "junk"})
    assert resp.status_code == 422
