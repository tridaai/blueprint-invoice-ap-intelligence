"""The LLM eval pack's judge budget — sized for reasoning models.

In the 2026-10-10 live slice, 4 of 6 judge verdicts came back
"judge reply was not a JSON object". The judge was not refusing
or rambling: it was a reasoning model working under a 300-token
completion budget, and the budget ran out mid-thought — the
answer channel returned empty, and the pack recorded a harness
failure in the drafts' column. The budget is now the judge's own
(``JUDGE_MAX_TOKENS`` / ``--judge-max-tokens``, default 2000),
and these tests pin that the configured budget is what the judge
call actually carries.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
sys.path.insert(0, str(EVALS_DIR))

import run_llm_evals  # noqa: E402

from ap_agent.schemas import Drafts, MatchFacts  # noqa: E402


class _JudgeBackend:
    """Records the budget each judge call arrives with and answers
    with a canned verdict."""

    name = "judge-stub"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.budgets: list[int] = []

    def complete_with_usage(self, system, user, max_tokens=None, model=None):
        self.budgets.append(max_tokens)
        return self.reply, {"input_tokens": 10, "output_tokens": 5}


def _result() -> SimpleNamespace:
    return SimpleNamespace(
        invoice_id="INV-1001",
        supplier_name="Northbeam Office Supplies",
        invoice_number="NB-4471",
        triage=SimpleNamespace(outcome="clean_match", severity="low"),
        facts=MatchFacts(match_status="full_match"),
        policies=[],
        drafts=Drafts(
            subject="Invoice INV-1001 — received",
            supplier_letter="Dear Northbeam team,\n\nNext step: review.",
            ap_packet="AP PACKET",
        ),
    )


_VERDICT = '{"grounded": true, "clean": true, "notes": "fine"}'


def test_judge_draft_carries_the_default_reasoning_budget():
    backend = _JudgeBackend(_VERDICT)
    verdict = run_llm_evals.judge_draft(backend, _result(), None)
    assert verdict == {"grounded": True, "clean": True, "notes": "fine"}
    assert backend.budgets == [run_llm_evals.DEFAULT_JUDGE_MAX_TOKENS]
    assert run_llm_evals.DEFAULT_JUDGE_MAX_TOKENS >= 1500  # room to think + answer


def test_judge_draft_carries_an_explicit_budget():
    backend = _JudgeBackend(_VERDICT)
    run_llm_evals.judge_draft(backend, _result(), None, max_tokens=4500)
    assert backend.budgets == [4500]


def test_judge_budget_resolution_order(monkeypatch):
    # Default, then the env var, then the CLI flag over both.
    monkeypatch.delenv("JUDGE_MAX_TOKENS", raising=False)
    assert run_llm_evals.judge_max_tokens() == run_llm_evals.DEFAULT_JUDGE_MAX_TOKENS
    monkeypatch.setenv("JUDGE_MAX_TOKENS", "3200")
    assert run_llm_evals.judge_max_tokens() == 3200
    assert run_llm_evals.judge_max_tokens(900) == 900
    # A broken env value falls back to the default, never to zero.
    monkeypatch.setenv("JUDGE_MAX_TOKENS", "not-a-number")
    assert run_llm_evals.judge_max_tokens() == run_llm_evals.DEFAULT_JUDGE_MAX_TOKENS
