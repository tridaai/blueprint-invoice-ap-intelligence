"""Four-eyes on large payment runs: a run whose net total is above
PAYMENT_RUN_FOUR_EYES_THRESHOLD (stamped at proposal time) cannot
be handed off until a second confirmer — different from the first
— has also confirmed it. The boundary is exact: a total at the
threshold is not above it."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.storage import InMemoryObjectStore
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}

# The INV-1001 run as of 2026-10-05: gross 1180.00, capturable
# discount 23.60, net 1156.40 (pinned by test_handoff.py).
NET = 1156.40


def _service_with_run(monkeypatch, threshold: str | None) -> InvoiceService:
    if threshold is None:
        monkeypatch.delenv("PAYMENT_RUN_FOUR_EYES_THRESHOLD", raising=False)
    else:
        monkeypatch.setenv("PAYMENT_RUN_FOUR_EYES_THRESHOLD", threshold)
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        object_store=InMemoryObjectStore(),
    )
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    svc.propose_payment_runs(window_days=7, as_of="2026-10-05")
    return svc


def _run(svc: InvoiceService):
    return svc.list_payment_runs()[0]


def test_run_at_the_threshold_is_not_above_it(monkeypatch):
    svc = _service_with_run(monkeypatch, f"{NET:.2f}")
    run = _run(svc)
    assert run.total_net == NET
    assert run.four_eyes_required is False
    svc.confirm_payment_run(run.run_id, actor="Tomas Reid")
    exported = svc.export_payment_run_handoff(run.run_id, actor="Tomas Reid")
    assert exported.handoff_key is not None  # single confirmation suffices


def test_run_above_the_threshold_needs_a_second_confirmer(monkeypatch):
    svc = _service_with_run(monkeypatch, f"{NET - 0.01:.2f}")
    run = _run(svc)
    assert run.four_eyes_required is True
    assert "four-eyes" in run.note

    svc.confirm_payment_run(run.run_id, actor="Tomas Reid")
    with pytest.raises(ValueError, match="second confirmer"):
        svc.export_payment_run_handoff(run.run_id, actor="Tomas Reid")

    with pytest.raises(ValueError, match="must differ from the first"):
        svc.confirm_payment_run_second(run.run_id, actor="Tomas Reid")

    second = svc.confirm_payment_run_second(run.run_id, actor="Amara Chen")
    assert second.second_confirmed_by == "Amara Chen"
    assert second.second_confirmed_at

    exported = svc.export_payment_run_handoff(run.run_id, actor="Tomas Reid")
    assert exported.handoff_key is not None


def test_second_confirmation_rules(monkeypatch):
    svc = _service_with_run(monkeypatch, "1000")
    run = _run(svc)
    # Before the first confirmation there is nothing to second.
    with pytest.raises(ValueError, match="first confirmation comes before"):
        svc.confirm_payment_run_second(run.run_id, actor="Amara Chen")
    svc.confirm_payment_run(run.run_id, actor="Tomas Reid")
    svc.confirm_payment_run_second(run.run_id, actor="Amara Chen")
    with pytest.raises(ValueError, match="already second-confirmed"):
        svc.confirm_payment_run_second(run.run_id, actor="Priya Nair")
    with pytest.raises(KeyError):
        svc.confirm_payment_run_second("RUN-NOPE", actor="Amara Chen")


def test_small_run_takes_no_second_confirmation(monkeypatch):
    svc = _service_with_run(monkeypatch, None)  # default 25000
    run = _run(svc)
    assert run.four_eyes_required is False
    svc.confirm_payment_run(run.run_id, actor="Tomas Reid")
    with pytest.raises(ValueError, match="no second confirmation is required"):
        svc.confirm_payment_run_second(run.run_id, actor="Amara Chen")


def test_four_eyes_endpoints(monkeypatch):
    svc = _service_with_run(monkeypatch, "1000")
    run = _run(svc)
    client = TestClient(create_app(svc))
    client.post(f"/payment-runs/{run.run_id}/confirm", json={"actor": "Tomas Reid"})
    blocked = client.post(
        f"/payment-runs/{run.run_id}/handoff", json={"actor": "Tomas Reid"}
    )
    assert blocked.status_code == 422
    assert "second confirmer" in blocked.json()["detail"]
    same = client.post(
        f"/payment-runs/{run.run_id}/confirm-second", json={"actor": "Tomas Reid"}
    )
    assert same.status_code == 422
    second = client.post(
        f"/payment-runs/{run.run_id}/confirm-second", json={"actor": "Amara Chen"}
    )
    assert second.status_code == 200
    assert second.json()["second_confirmed_by"] == "Amara Chen"
    exported = client.post(
        f"/payment-runs/{run.run_id}/handoff", json={"actor": "Tomas Reid"}
    )
    assert exported.status_code == 200
