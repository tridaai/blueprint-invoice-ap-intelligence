"""The digest auto-store sweep: the stored-digest history used
to depend on somebody remembering to press Store every morning.
The sweep pins every unstored day inside a bounded lookback
through the same first-computed-wins path a human uses —
attributed to ``system``, so a backfilled day never masquerades
as a morning ritual — and reports stored vs already-stored per
day. Days a human stored keep their copy and their name."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
ANCHOR = "2026-10-10"  # a Saturday; the lookback reaches back a week


@pytest.fixture()
def service() -> InvoiceService:
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    return svc


def test_the_sweep_stores_every_unstored_day_in_the_lookback(service):
    report = service.store_due_digests(lookback_days=3, as_of=ANCHOR)
    assert report.through_date == ANCHOR
    assert report.lookback_days == 3
    assert report.stored_count == 3 and report.already_stored_count == 0
    assert [e.date for e in report.entries] == [
        "2026-10-08",
        "2026-10-09",
        "2026-10-10",
    ]
    assert all(e.outcome == "stored" for e in report.entries)
    assert all(e.stored_by == "system" for e in report.entries)
    # Each pinned day reads back verbatim, carrying the sweep's
    # attribution in its own provenance.
    for day in ("2026-10-08", "2026-10-09", "2026-10-10"):
        stored = service.get_stored_digest(day)
        assert stored is not None
        assert stored.provenance == "stored" and stored.stored_by == "system"
        assert stored.stored_at  # pinned at sweep time, not backdated


def test_a_human_stored_day_is_never_overwritten(service):
    human = service.store_digest(as_of="2026-10-09", actor="Priya Nair")
    report = service.store_due_digests(lookback_days=3, as_of=ANCHOR)
    entry = next(e for e in report.entries if e.date == "2026-10-09")
    assert entry.outcome == "already_stored"
    assert entry.stored_by == "Priya Nair"
    assert report.stored_count == 2 and report.already_stored_count == 1
    # The human's copy survives the sweep byte for byte.
    assert service.get_stored_digest("2026-10-09") == human


def test_the_sweep_is_idempotent(service):
    service.store_due_digests(lookback_days=3, as_of=ANCHOR)
    again = service.store_due_digests(lookback_days=3, as_of=ANCHOR)
    assert again.stored_count == 0 and again.already_stored_count == 3
    history = service.digest_history()
    assert len(history.entries) == 3  # no duplicates, no rewrites


def test_the_lookback_is_bounded(service):
    report = service.store_due_digests(lookback_days=2, as_of=ANCHOR)
    assert [e.date for e in report.entries] == ["2026-10-09", "2026-10-10"]
    # A day outside the window stays unstored — the sweep is not
    # a licence to rewrite ancient history.
    assert service.get_stored_digest("2026-10-08") is None


def test_the_default_lookback_comes_from_configuration(service, monkeypatch):
    monkeypatch.setenv("DIGEST_STORE_LOOKBACK_DAYS", "4")
    report = service.store_due_digests(as_of=ANCHOR)
    assert report.lookback_days == 4 and report.stored_count == 4
    monkeypatch.delenv("DIGEST_STORE_LOOKBACK_DAYS")
    report = service.store_due_digests(as_of="2026-10-20")
    assert report.lookback_days == 7  # the shipped default


def test_a_bad_anchor_is_refused(service):
    with pytest.raises(ValueError, match="not an ISO date"):
        service.store_due_digests(as_of="not-a-date")


def test_the_sweep_endpoint(service):
    client = TestClient(create_app(service))
    res = client.post(
        "/digest/store-due", params={"as_of": ANCHOR, "lookback_days": 2}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["stored_count"] == 2 and body["already_stored_count"] == 0
    assert [e["date"] for e in body["entries"]] == ["2026-10-09", "2026-10-10"]
    bad = client.post("/digest/store-due", params={"as_of": "nope"})
    assert bad.status_code == 422
    # And the history endpoint now shows the backfilled series.
    history = client.get("/digest/history").json()
    assert len(history["entries"]) == 2
    assert all(e["stored_by"] == "system" for e in history["entries"])
