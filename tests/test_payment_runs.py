"""Payment-run proposals: approved invoices grouped by currency +
due window into a draft pay list a finance lead confirms. The agent
proposes; it never pays (paymentruns.py + service payment-run
methods)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


@pytest.fixture()
def service() -> InvoiceService:
    """INV-1001 (USD 1180, due 2026-10-28, 2/10 net 30) and INV-1007
    (EUR 357, due 2026-10-23) approved; INV-1008 left awaiting."""
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    for invoice_id in ("INV-1001", "INV-1007", "INV-1008"):
        svc.analyze(SAMPLES[invoice_id])
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    svc.approve("INV-1007", approver="Priya Nair", reason="clean match")
    return svc


def test_proposal_groups_by_currency_and_window(service):
    runs = service.propose_payment_runs(window_days=7, as_of="2026-10-20")
    assert len(runs) == 2
    by_currency = {run.currency: run for run in runs}
    eur = by_currency["EUR"]
    assert eur.invoice_count == 1
    assert eur.lines[0].invoice_id == "INV-1007"
    assert eur.window_start == "2026-10-23"  # earliest due date in window 0
    assert eur.window_end == "2026-10-26"
    assert eur.total_gross == 357.0 and eur.total_net == 357.0
    usd = by_currency["USD"]
    # Due 2026-10-28 is 8 days out: the second weekly window.
    assert usd.window_start == "2026-10-27"
    assert usd.window_end == "2026-11-02"
    assert usd.lines[0].invoice_id == "INV-1001"
    assert usd.status == "proposed"


def test_awaiting_and_rejected_invoices_are_not_proposed(service):
    service.reject("INV-1008", reviewer="Priya Nair", reason="price variance")
    runs = service.propose_payment_runs(as_of="2026-10-20")
    member_ids = {line.invoice_id for run in runs for line in run.lines}
    assert member_ids == {"INV-1001", "INV-1007"}


def test_discount_counts_while_its_deadline_is_reachable(service):
    runs = service.propose_payment_runs(window_days=7, as_of="2026-10-05")
    usd = next(run for run in runs if run.currency == "USD")
    line = usd.lines[0]
    # INV-1001: 2/10 net 30, deadline 2026-10-08 >= as_of, gross 23.60.
    assert line.discount_amount == 23.60
    assert line.discount_deadline == "2026-10-08"
    assert line.net_amount == 1180.0 - 23.60
    assert usd.total_discount == 23.60
    assert usd.total_net == usd.total_gross - 23.60


def test_discount_lapsed_by_as_of_is_not_counted(service):
    runs = service.propose_payment_runs(window_days=7, as_of="2026-10-20")
    usd = next(run for run in runs if run.currency == "USD")
    assert usd.lines[0].discount_amount == 0.0
    assert usd.total_discount == 0.0
    assert usd.total_net == usd.total_gross == 1180.0


def test_overdue_approved_invoice_lands_in_the_first_window(service):
    runs = service.propose_payment_runs(window_days=7, as_of="2026-11-15")
    eur = next(run for run in runs if run.currency == "EUR")
    assert eur.lines[0].invoice_id == "INV-1007"  # due 2026-10-23, overdue
    assert eur.window_end == "2026-11-21"


def test_confirm_records_the_human_and_refuses_twice(service):
    runs = service.propose_payment_runs(as_of="2026-10-20")
    run = runs[0]
    confirmed = service.confirm_payment_run(run.run_id, actor="Tomas Reid")
    assert confirmed.status == "confirmed"
    assert confirmed.confirmed_by == "Tomas Reid"
    assert confirmed.confirmed_at
    with pytest.raises(ValueError, match="already confirmed"):
        service.confirm_payment_run(run.run_id, actor="Tomas Reid")
    with pytest.raises(KeyError):
        service.confirm_payment_run("RUN-XXX-1-2", actor="Tomas Reid")


def test_confirmed_members_are_not_reproposed(service):
    runs = service.propose_payment_runs(as_of="2026-10-20")
    eur = next(run for run in runs if run.currency == "EUR")
    service.confirm_payment_run(eur.run_id, actor="Tomas Reid")
    again = service.propose_payment_runs(as_of="2026-10-20")
    member_ids = {line.invoice_id for run in again for line in run.lines}
    assert "INV-1007" not in member_ids
    assert "INV-1001" in member_ids  # the unconfirmed USD run refreshes
    stored = {run.run_id: run for run in service.list_payment_runs()}
    assert stored[eur.run_id].status == "confirmed"


def test_proposing_and_confirming_change_no_invoice_state(service):
    runs = service.propose_payment_runs(as_of="2026-10-20")
    service.confirm_payment_run(runs[0].run_id, actor="Tomas Reid")
    result = service.get("INV-1001")
    assert result is not None
    assert result.status == "approved"  # untouched by the run lifecycle
    assert result.external_action_taken is False


def test_payment_run_endpoints():
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="approved in test")
    client = TestClient(create_app(svc))
    proposed = client.post(
        "/payment-runs/propose", json={"window_days": 7, "as_of": "2026-10-20"}
    )
    assert proposed.status_code == 200
    runs = proposed.json()
    assert len(runs) == 1 and runs[0]["currency"] == "USD"
    run_id = runs[0]["run_id"]
    listed = client.get("/payment-runs")
    assert [r["run_id"] for r in listed.json()] == [run_id]
    fetched = client.get(f"/payment-runs/{run_id}")
    assert fetched.status_code == 200
    confirmed = client.post(
        f"/payment-runs/{run_id}/confirm", json={"actor": "Tomas Reid"}
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["status"] == "confirmed"
    again = client.post(
        f"/payment-runs/{run_id}/confirm", json={"actor": "Tomas Reid"}
    )
    assert again.status_code == 422
    missing = client.get("/payment-runs/RUN-NOPE")
    assert missing.status_code == 404
