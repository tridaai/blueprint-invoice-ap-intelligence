"""Lag-watch announcements in the escalation digest.

The drift watch demotes a decaying lag statistic inside the
forecast's selection — automatically, silently, run by run.
But a forecast quietly degrading is an operations event: the
statistic that prices tomorrow's confirmations changed, and
nobody was told. The digest announces it now: the degrading
segments named with both windows' recalls and graded-member
counts, the segments standing excluded by governed decision
named too (an exclusion nobody can see is its own kind of
silence), and the day's confirmations the demotions actually
shaped, counted. A quiet watch announces nothing.
"""

from __future__ import annotations

from ap_agent.schemas import PaymentRun
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore
from test_lag_drift_watch import _history


def _service_with_history() -> InvoiceService:
    store = InMemoryStore()
    for run in _history():
        store.save_payment_run(run)
    return InvoiceService(store=store, checkpointer=False)


def test_a_degrading_statistic_is_announced_with_its_evidence():
    svc = _service_with_history()
    digest = svc.escalation_digest(as_of="2026-10-20")
    by_segment = {entry.segment: entry for entry in digest.lag_watch}
    assert set(by_segment) == {"currency:EUR", "global"}
    eur = by_segment["currency:EUR"]
    assert eur.degrading is True
    assert eur.excluded is False
    assert eur.earlier_recall == 1.0
    assert eur.recent_recall == 0.0
    assert eur.earlier_graded_members > 0
    assert eur.recent_graded_members > 0
    assert digest.lag_watch_degrading_count == 2
    assert digest.lag_watch_excluded_segments == []
    assert "Lag watch" in digest.summary
    assert "currency:EUR" in digest.summary
    assert "All clear" not in digest.summary


def test_demotions_that_shaped_the_days_confirmations_are_counted():
    svc = _service_with_history()
    store = svc._get_store()
    store.save_payment_run(
        PaymentRun(
            run_id="USD-TODAY",
            currency="USD",
            status="confirmed",
            proposed_at="2026-10-20T09:00:00+00:00",
            confirmed_at="2026-10-20T10:00:00+00:00",
            confirmed_by="Tomas Reid",
            invoice_count=1,
            drift_forecast_basis=(
                "the currency:USD statistic was demoted by the "
                "drift watch (recent recall 0.0 against 1.0 "
                "earlier); the global statistic priced the run"
            ),
        )
    )
    # A confirmation the watch did not shape does not count.
    store.save_payment_run(
        PaymentRun(
            run_id="USD-PLAIN",
            currency="USD",
            status="confirmed",
            proposed_at="2026-10-20T09:00:00+00:00",
            confirmed_at="2026-10-20T11:00:00+00:00",
            confirmed_by="Tomas Reid",
            invoice_count=1,
            drift_forecast_basis="the global statistic priced the run",
        )
    )
    digest = svc.escalation_digest(as_of="2026-10-20")
    assert digest.lag_watch_demotions_today == 1
    assert "shaped 1 of today's confirmation(s)" in digest.summary
    # Yesterday's digest does not inherit today's count.
    digest = svc.escalation_digest(as_of="2026-10-21")
    assert digest.lag_watch_demotions_today == 0


def test_an_excluded_segment_is_announced_as_excluded():
    svc = _service_with_history()
    drafted = svc.draft_lag_segment_proposals(actor="Ana Ruiz")
    eur = next(p for p in drafted if p.segment == "currency:EUR")
    svc.decide_lag_segment_proposal(
        eur.proposal_id, actor="Tomas Reid", approve=True, note="retire it"
    )
    digest = svc.escalation_digest(as_of="2026-10-20")
    assert digest.lag_watch_excluded_segments == ["currency:EUR"]
    by_segment = {entry.segment: entry for entry in digest.lag_watch}
    assert by_segment["currency:EUR"].excluded is True
    assert "excluded from the candidate set" in digest.summary


def test_a_quiet_watch_announces_nothing():
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    digest = svc.escalation_digest(as_of="2026-10-20")
    assert digest.lag_watch == []
    assert digest.lag_watch_degrading_count == 0
    assert digest.lag_watch_demotions_today == 0
    assert "Lag watch" not in digest.summary
    assert "All clear" in digest.summary


def test_a_stored_digest_keeps_its_watch_announcements():
    svc = _service_with_history()
    svc.store_digest(as_of="2026-10-20", actor="Ana Ruiz")
    stored = svc.get_stored_digest("2026-10-20")
    assert stored is not None
    assert stored.provenance == "stored"
    assert stored.lag_watch_degrading_count == 2
    assert {entry.segment for entry in stored.lag_watch} == {
        "currency:EUR",
        "global",
    }
