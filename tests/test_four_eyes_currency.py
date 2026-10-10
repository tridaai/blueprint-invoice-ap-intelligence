"""Per-currency four-eyes thresholds: 25,000 USD and 25,000 JPY
are not the same risk, so PAYMENT_RUN_FOUR_EYES_THRESHOLDS maps
currencies to their own figures and the global
PAYMENT_RUN_FOUR_EYES_THRESHOLD is only the fallback. The figure
that applied is stamped on the run at proposal and named in its
note — the gate a run was judged against is part of its record."""

from __future__ import annotations

import pytest

from ap_agent.samples import sample_invoice_models
from ap_agent.service import (
    InvoiceService,
    payment_run_four_eyes_threshold_for,
    payment_run_four_eyes_thresholds,
)
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


@pytest.fixture()
def service() -> InvoiceService:
    """INV-1001 (USD, net 1156.40 in the Oct 5 window) and INV-1007
    (EUR 357.00) approved."""
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    for invoice_id in ("INV-1001", "INV-1007"):
        svc.analyze(SAMPLES[invoice_id])
        svc.approve(invoice_id, approver="Priya Nair", reason="clean match")
    return svc


def _by_currency(svc: InvoiceService, as_of: str = "2026-10-05"):
    runs = svc.propose_payment_runs(window_days=7, as_of=as_of)
    return {run.currency: run for run in runs}


def test_map_parsing_skips_malformed_pairs(monkeypatch):
    monkeypatch.setenv(
        "PAYMENT_RUN_FOUR_EYES_THRESHOLDS", "EUR=50000, jpy = 3000000 ,USD=oops,FR"
    )
    assert payment_run_four_eyes_thresholds() == {
        "EUR": 50000.0,
        "JPY": 3000000.0,
    }
    monkeypatch.setenv("PAYMENT_RUN_FOUR_EYES_THRESHOLD", "25000")
    assert payment_run_four_eyes_threshold_for("eur") == 50000.0
    assert payment_run_four_eyes_threshold_for("GBP") == 25000.0  # fallback


def test_currency_threshold_gates_only_its_own_currency(service, monkeypatch):
    monkeypatch.setenv("PAYMENT_RUN_FOUR_EYES_THRESHOLD", "25000")
    monkeypatch.setenv("PAYMENT_RUN_FOUR_EYES_THRESHOLDS", "EUR=300")
    runs = _by_currency(service)
    eur, usd = runs["EUR"], runs["USD"]
    # EUR 357 is below the global 25000 but above EUR's own 300.
    assert eur.total_net == 357.0
    assert eur.four_eyes_required is True
    assert eur.four_eyes_threshold == 300.0
    assert "EUR 300.00" in eur.note and "above that threshold" in eur.note
    # USD is judged against the global fallback, and the run says so.
    assert usd.four_eyes_required is False
    assert usd.four_eyes_threshold == 25000.0
    assert "USD 25,000.00" in usd.note


def test_currency_threshold_can_also_relax_the_gate(service, monkeypatch):
    monkeypatch.setenv("PAYMENT_RUN_FOUR_EYES_THRESHOLD", "1000")
    monkeypatch.setenv("PAYMENT_RUN_FOUR_EYES_THRESHOLDS", "USD=2000")
    runs = _by_currency(service)
    usd = runs["USD"]
    # Net 1156.40 is above the global 1000 but below USD's own 2000:
    # the per-currency figure wins in both directions.
    assert usd.total_net == 1156.40
    assert usd.four_eyes_required is False
    assert usd.four_eyes_threshold == 2000.0
    # And the second-confirmation refusal names the figure applied.
    svc_confirm = service.confirm_payment_run(usd.run_id, actor="Tomas Reid")
    assert svc_confirm.status == "confirmed"
    with pytest.raises(ValueError, match="2,000.00"):
        service.confirm_payment_run_second(usd.run_id, actor="Amara Chen")


def test_no_thresholds_configured_stamps_the_global_default(service, monkeypatch):
    monkeypatch.delenv("PAYMENT_RUN_FOUR_EYES_THRESHOLD", raising=False)
    monkeypatch.delenv("PAYMENT_RUN_FOUR_EYES_THRESHOLDS", raising=False)
    runs = _by_currency(service)
    assert runs["USD"].four_eyes_threshold == 25000.0
    assert runs["EUR"].four_eyes_threshold == 25000.0
    assert runs["USD"].four_eyes_required is False
