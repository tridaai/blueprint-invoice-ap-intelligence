"""Budgets in force, on the record: run telemetry stamps the
completion budgets the run actually ran with.

Round 15 made every completion budget env-tunable and wrote the
audit down — but nothing on a run's record said which budgets
that run used, so a starved-reply post-mortem (a reasoning model
truncated mid-thought, read downstream as a malformed reply)
had to re-derive the budget from the code and the environment
the run happened to start in. Now the telemetry carries the
resolved budgets per call site — the provider-call registry
plus the OCR/vision ports — and the LLM eval pack records the
same stamp beside the judge budget.
"""

from __future__ import annotations

from ap_agent.model_backends import (
    COMPLETION_BUDGETS,
    run_completion_budgets,
)
from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore


def _run_one() -> object:
    service = InvoiceService(store=InMemoryStore(), checkpointer=False)
    invoice = sample_invoice_models()[0]
    return service.analyze(invoice)


def test_run_telemetry_stamps_every_budget_in_force():
    result = _run_one()
    budgets = result.telemetry.completion_budgets
    assert budgets is not None
    # Every provider call site in the registry, plus the two
    # document ports — nothing a run can spend is missing.
    assert set(budgets) == set(COMPLETION_BUDGETS) | {"ocr", "vision"}
    assert budgets == run_completion_budgets()


def test_telemetry_budgets_carry_the_audited_defaults():
    budgets = _run_one().telemetry.completion_budgets
    assert budgets["draft"] == COMPLETION_BUDGETS["draft"][1]
    assert budgets["classify"] == COMPLETION_BUDGETS["classify"][1]
    assert budgets["ocr"] == 2000
    assert budgets["vision"] == 1000


def test_telemetry_budgets_record_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_DRAFT_MAX_TOKENS", "4321")
    monkeypatch.setenv("OCR_MAX_TOKENS", "777")
    budgets = _run_one().telemetry.completion_budgets
    # The stamp is what the run ran with, not the defaults: an
    # operator retune is visible on the record it affected.
    assert budgets["draft"] == 4321
    assert budgets["ocr"] == 777
    # An override retunes one call only.
    assert budgets["classify"] == COMPLETION_BUDGETS["classify"][1]
