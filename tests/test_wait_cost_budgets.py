"""Wait-cost budgets: the priced four-eyes wait, owned.

Round 11 priced the second-confirmation wait per run. A price
nobody owns is still just a number, so the wait now carries a
per-currency monthly budget (FOUR_EYES_WAIT_BUDGETS): /metrics
reports the month's destroyed discounts against the ceiling in
force, and the digest names a breached budget every morning
until the month turns. A currency with no budget configured is
unbudgeted — reported, never breached.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ap_agent.paymentruns import wait_cost_budget_status
from ap_agent.schemas import PaymentRun
from ap_agent.service import InvoiceService, four_eyes_wait_budgets
from ap_agent.storage import InMemoryObjectStore
from ap_agent.store import InMemoryStore
from ap_agent.supplier_master import InMemorySupplierMaster


def _run(
    run_id: str,
    currency: str,
    *,
    second_confirmed_at: str | None,
    wait_cost: float,
) -> PaymentRun:
    return PaymentRun(
        run_id=run_id,
        currency=currency,
        status="confirmed",
        confirmed_by="Ana Ruiz",
        confirmed_at="2026-10-02T09:00:00",
        second_confirmed_by="Tomas Reid" if second_confirmed_at else None,
        second_confirmed_at=second_confirmed_at,
        four_eyes_wait_cost=wait_cost,
    )


def _service() -> InvoiceService:
    return InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        object_store=InMemoryObjectStore(),
        supplier_master=InMemorySupplierMaster(),
    )


# --- the pure read model ----------------------------------------------------


def test_status_aggregates_the_month_and_marks_the_breach():
    runs = [
        _run("R1", "EUR", second_confirmed_at="2026-10-03T10:00:00", wait_cost=700.0),
        _run("R2", "EUR", second_confirmed_at="2026-10-20T10:00:00", wait_cost=550.0),
        # Another month: not this month's cost.
        _run("R3", "EUR", second_confirmed_at="2026-09-20T10:00:00", wait_cost=900.0),
        # Never second-confirmed: the wait never ended, no price.
        _run("R4", "EUR", second_confirmed_at=None, wait_cost=0.0),
        _run("R5", "USD", second_confirmed_at="2026-10-05T10:00:00", wait_cost=80.0),
    ]
    status = wait_cost_budget_status(
        runs, {"EUR": 1000.0}, month="2026-10"
    )
    assert status["month"] == "2026-10"
    assert status["budgets_configured"] is True
    rows = {row["currency"]: row for row in status["currencies"]}
    assert rows["EUR"]["wait_cost"] == 1250.0
    assert rows["EUR"]["runs_waited"] == 2
    assert rows["EUR"]["budget"] == 1000.0
    assert rows["EUR"]["over_budget"] is True
    assert rows["EUR"]["excess"] == 250.0
    # USD has cost but no budget: reported, unbudgeted, unbreached.
    assert rows["USD"]["budget"] is None
    assert rows["USD"]["over_budget"] is False
    assert status["breach_currencies"] == ["EUR"]


def test_a_budgeted_currency_with_no_cost_reports_zero_not_a_breach():
    status = wait_cost_budget_status([], {"JPY": 5000.0}, month="2026-10")
    rows = {row["currency"]: row for row in status["currencies"]}
    assert rows["JPY"]["wait_cost"] == 0.0
    assert rows["JPY"]["over_budget"] is False
    assert status["breach_currencies"] == []


def test_exactly_at_budget_is_not_over():
    runs = [
        _run("R1", "EUR", second_confirmed_at="2026-10-03T10:00:00", wait_cost=1000.0)
    ]
    status = wait_cost_budget_status(runs, {"EUR": 1000.0}, month="2026-10")
    assert status["currencies"][0]["over_budget"] is False


# --- configuration ------------------------------------------------------------


def test_budgets_parse_like_the_thresholds(monkeypatch):
    monkeypatch.setenv(
        "FOUR_EYES_WAIT_BUDGETS", "EUR=1000, usd = 250.5 ,broken,JPY=abc"
    )
    assert four_eyes_wait_budgets() == {"EUR": 1000.0, "USD": 250.5}
    monkeypatch.delenv("FOUR_EYES_WAIT_BUDGETS")
    assert four_eyes_wait_budgets() == {}


# --- service surfaces ---------------------------------------------------------


def test_metrics_report_the_current_month_against_the_budget(monkeypatch):
    monkeypatch.setenv("FOUR_EYES_WAIT_BUDGETS", "EUR=100")
    svc = _service()
    now = datetime.now(timezone.utc)
    run = _run(
        "R-NOW",
        "EUR",
        second_confirmed_at=now.isoformat(timespec="seconds"),
        wait_cost=140.0,
    )
    svc._get_store().save_payment_run(run)
    section = svc.metrics_snapshot()["four_eyes_wait_budget"]
    assert section["month"] == now.strftime("%Y-%m")
    assert section["breach_currencies"] == ["EUR"]
    row = section["currencies"][0]
    assert row["wait_cost"] == 140.0 and row["excess"] == 40.0


def test_digest_names_the_breach_and_stays_quiet_without_budgets(monkeypatch):
    svc = _service()
    run = _run(
        "R1", "EUR",
        second_confirmed_at="2026-10-03T10:00:00", wait_cost=1250.0,
    )
    svc._get_store().save_payment_run(run)
    monkeypatch.setenv("FOUR_EYES_WAIT_BUDGETS", "EUR=1000")
    digest = svc.escalation_digest(as_of="2026-10-21")
    assert len(digest.wait_budget_breaches) == 1
    breach = digest.wait_budget_breaches[0]
    assert breach.currency == "EUR"
    assert breach.wait_cost == 1250.0 and breach.excess == 250.0
    assert "Four-eyes wait budget breached" in digest.summary
    assert "EUR 1,250.00" in digest.summary
    # No budgets configured: no breach, no budget line — an
    # unbudgeted month is not dressed up as a governed one.
    monkeypatch.delenv("FOUR_EYES_WAIT_BUDGETS")
    digest = svc.escalation_digest(as_of="2026-10-21")
    assert digest.wait_budget_breaches == []
    assert "wait budget" not in digest.summary
