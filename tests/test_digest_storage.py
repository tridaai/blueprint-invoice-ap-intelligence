"""Stored digest artifacts: the computed digest describes the
records as they are *now*, so yesterday's digest cannot be
re-derived once the records move — a case transitions, a posting
closes, and the day re-composes differently. An operator can pin
the day's first-computed digest in the store: read back verbatim
by date, labelled with its provenance, first-computed wins. The
stored series forms a history — headline counts with day-over-day
changes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.digest import build_digest_history, headline_counts
from ap_agent.schemas import (
    DigestDriftEntry,
    DigestForecastLapse,
    EscalationDigest,
    RecoveryCaseEvent,
    RecoveryPack,
)
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore


def _digest(as_of: str, **counts) -> EscalationDigest:
    digest = EscalationDigest(
        as_of=as_of,
        provenance="stored",
        stored_at=f"{as_of}T18:00:00+00:00",
        stored_by="Priya Nair",
        forecast_lapses=[
            DigestForecastLapse(run_id="RUN-1", invoice_id="INV-1")
        ]
        * counts.pop("forecast_lapse_count", 0),
        drift_today=[
            DigestDriftEntry(run_id="RUN-1")
        ]
        * counts.pop("drift_run_count", 0),
        **counts,
    )
    return digest


def test_history_orders_by_date_and_diffs_against_the_previous_stored_day():
    later = _digest(
        "2026-11-16",
        recovery_breach_count=3,
        stuck_invoice_count=1,
        forecast_lapse_count=2,
    )
    earlier = _digest(
        "2026-11-15",
        recovery_breach_count=1,
        stuck_invoice_count=1,
        forecast_lapse_count=1,
        drift_run_count=1,
    )
    history = build_digest_history([later, earlier])  # unordered input
    assert history.count == 2
    first, second = history.entries
    assert first.as_of == "2026-11-15"
    assert first.changes == {}  # no predecessor: no dressed-up deltas
    assert first.forecast_lapse_count == 1 and first.drift_run_count == 1
    assert second.as_of == "2026-11-16"
    # Moved counts appear; the count that held (stuck) is absent,
    # and drift falling to zero is a change like any other.
    assert second.changes == {
        "recovery_breach_count": 2,
        "forecast_lapse_count": 1,
        "drift_run_count": -1,
    }
    assert headline_counts(later)["recovery_breach_count"] == 3


def _pack(pack_id: str, invoice_id: str) -> RecoveryPack:
    # Opened Sep 1, contacted Sep 20: breached against the 14-day
    # SLA at the November dates used below.
    return RecoveryPack(
        pack_id=pack_id,
        duplicate_invoice_id=invoice_id,
        supplier_name="Northbeam Office Supplies",
        amount=1180.0,
        created_at="2026-09-01T09:00:00+00:00",
        case_status="supplier_contacted",
        case_history=[
            RecoveryCaseEvent(status="opened", actor="op", at="2026-09-01T09:00:00+00:00"),
            RecoveryCaseEvent(status="supplier_contacted", actor="op", at="2026-09-20T09:00:00+00:00"),
        ],
    )


def _service() -> InvoiceService:
    store = InMemoryStore()
    store.save_recovery_pack(_pack("REC-INV-9001", "INV-9001"))
    return InvoiceService(store=store, checkpointer=False)


def test_store_then_read_back_verbatim_even_after_the_records_move():
    svc = _service()
    assert svc.get_stored_digest("2026-11-15") is None
    stored = svc.store_digest(as_of="2026-11-15", actor="Priya Nair")
    assert stored.provenance == "stored"
    assert stored.stored_by == "Priya Nair" and stored.stored_at
    assert stored.recovery_breach_count == 1
    # The records move: a second case breaches too.
    svc._get_store().save_recovery_pack(_pack("REC-INV-9002", "INV-9002"))
    assert svc.escalation_digest(as_of="2026-11-15").recovery_breach_count == 2
    # The stored copy is the day as it was — verbatim, labelled.
    read_back = svc.get_stored_digest("2026-11-15")
    assert read_back.recovery_breach_count == 1
    assert read_back.provenance == "stored"
    assert read_back.summary == stored.summary


def test_first_computed_wins_a_second_store_never_overwrites():
    svc = _service()
    first = svc.store_digest(as_of="2026-11-15", actor="Priya Nair")
    svc._get_store().save_recovery_pack(_pack("REC-INV-9002", "INV-9002"))
    second = svc.store_digest(as_of="2026-11-15", actor="Tomas Reid")
    assert second.stored_by == "Priya Nair"  # the original storer
    assert second.stored_at == first.stored_at
    assert second.recovery_breach_count == 1


def test_the_computed_digest_never_wears_the_stored_label():
    svc = _service()
    digest = svc.escalation_digest(as_of="2026-11-15")
    assert digest.provenance == "computed"
    assert digest.stored_at == "" and digest.stored_by == ""


def test_history_service_view_and_garbage_dates():
    svc = _service()
    assert svc.digest_history().count == 0
    svc.store_digest(as_of="2026-11-15", actor="Priya Nair")
    svc.store_digest(as_of="2026-11-16", actor="Priya Nair")
    history = svc.digest_history()
    assert [e.as_of for e in history.entries] == ["2026-11-15", "2026-11-16"]
    for call in (
        lambda: svc.get_stored_digest("junk"),
        lambda: svc.store_digest(as_of="junk", actor="op"),
    ):
        try:
            call()
        except ValueError as exc:
            assert "not an ISO date" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


def test_digest_storage_api():
    client = TestClient(create_app(_service()))
    missing = client.get("/digest/2026-11-15")
    assert missing.status_code == 404
    stored = client.post("/digest/2026-11-15/store", json={"actor": "Priya Nair"})
    assert stored.status_code == 200
    body = stored.json()
    assert body["provenance"] == "stored" and body["stored_by"] == "Priya Nair"
    read_back = client.get("/digest/2026-11-15")
    assert read_back.status_code == 200
    assert read_back.json()["summary"] == body["summary"]
    history = client.get("/digest/history")
    assert history.status_code == 200
    assert history.json()["count"] == 1
    assert history.json()["entries"][0]["as_of"] == "2026-11-15"
    # Storing needs a named operator, like every attributable act.
    assert client.post("/digest/2026-11-16/store", json={}).status_code == 422
    assert client.get("/digest/not-a-date").status_code == 422
