"""AI-layer machinery: budget guard, resilience, toolbox, retrieval."""

from __future__ import annotations

from ap_agent.graph import _BudgetGuard
from ap_agent.model_backends import MockModelBackend
from ap_agent.retriever import KeywordRetriever, rerank_fused
from ap_agent.schemas import MatchFacts, RetrievedPolicy
from ap_agent.tools_agent import DiagnosisToolBox, diagnosis_max_tool_calls
from ap_agent.samples import sample_invoice_models


class _RichBackend(MockModelBackend):
    def classify_with_llm(self, context):
        return {"outcome": "clean_match", "severity": "low", "confidence": 0.9, "rationale": "x", "backend": "rich"}

    def usage_totals(self):
        return {"input_tokens": 10_000, "output_tokens": 10_000, "calls": 3}


def test_budget_guard_hides_capabilities_over_budget():
    guard = _BudgetGuard(_RichBackend(), MockModelBackend(), limit=100)
    assert getattr(guard, "classify_with_llm", None) is None  # degraded
    status = guard.status()
    assert status["exceeded"] is True


def test_budget_guard_passes_through_under_budget():
    backend = _RichBackend()
    backend.usage_totals = lambda: {"input_tokens": 1, "output_tokens": 1, "calls": 1}
    guard = _BudgetGuard(backend, MockModelBackend(), limit=100_000)
    assert getattr(guard, "classify_with_llm", None) is not None


def test_budget_guard_no_limit_never_degrades():
    guard = _BudgetGuard(_RichBackend(), MockModelBackend(), limit=None)
    assert getattr(guard, "classify_with_llm", None) is not None
    assert guard.status() is None


def test_rrf_fusion_prefers_policies_found_by_both():
    kw = [
        RetrievedPolicy(policy_id="AP-MATCH-01", title="a", text="x", score=0.9, sources=["keyword"]),
        RetrievedPolicy(policy_id="AP-PO-01", title="b", text="x", score=0.8, sources=["keyword"]),
    ]
    sem = [
        RetrievedPolicy(policy_id="AP-PO-01", title="b", text="x", score=0.7, sources=["semantic"]),
        RetrievedPolicy(policy_id="AP-TAX-01", title="c", text="x", score=0.6, sources=["semantic"]),
    ]
    fused = rerank_fused(kw, sem, top_k=3)
    assert fused[0].policy_id == "AP-PO-01"  # in both lists
    assert fused[0].sources == ["keyword", "semantic"]


def test_keyword_retriever_finds_match_policy():
    results = KeywordRetriever().retrieve("three way match tolerance price variance", top_k=4)
    ids = [r.policy_id for r in results]
    assert "AP-MATCH-01" in ids or "AP-MATCH-02" in ids


def test_toolbox_dispatch_and_errors():
    inv = sample_invoice_models()[0]
    facts = MatchFacts(match_status="full_match", match_score=1.0)
    box = DiagnosisToolBox(
        invoice=inv,
        triage={"outcome": "clean_match", "severity": "low", "confidence": 0.95},
        facts=facts,
        retriever=KeywordRetriever(),
        priors=[],
    )
    text, summary = box.dispatch("match_facts", {})
    assert "full_match" in text
    text, summary = box.dispatch("po_lookup", {})
    assert "PO-9001" in text
    text, summary = box.dispatch("supplier_history", {})
    assert "No prior invoices" in text
    try:
        box.dispatch("nonsense", {})
        raise AssertionError("unknown tool must raise")
    except ValueError:
        pass


def test_tool_call_budget_bounds(monkeypatch):
    monkeypatch.setenv("DIAGNOSIS_MAX_TOOL_CALLS", "99")
    assert diagnosis_max_tool_calls() == 6  # hard cap
    monkeypatch.setenv("DIAGNOSIS_MAX_TOOL_CALLS", "2")
    assert diagnosis_max_tool_calls() == 2
