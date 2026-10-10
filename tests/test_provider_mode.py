"""Provider mode with fakes: the LLM seams and the NIM ports.

A scripted fake backend stands in for a real provider (and for the
NVIDIA NIM profile, which speaks the same OpenAI-compatible shape):
it exercises extraction, the triage cross-check, the agentic
diagnosis tool loop, options, verify/review — plus the OCR and
vision ports via fake NIM engines. No network anywhere.
"""

from __future__ import annotations

import json

from ap_agent.crosscheck import resolve_triage
from ap_agent.graph import run_invoice
from ap_agent.model_backends import MockModelBackend
from ap_agent.retriever import KeywordRetriever
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import InvoiceInput, OcrResult, Triage, VisionResult
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLE = sample_invoice_models()[0]  # INV-1001, clean


class FakeLLMBackend(MockModelBackend):
    """A MockModelBackend plus the optional LLM capabilities, scripted."""

    name = "fake-llm"

    def __init__(self, *, classify_reply=None, tool_loop=True):
        super().__init__()
        self._classify_reply = classify_reply
        self._tool_loop = tool_loop
        self.tool_names_seen: list[str] = []

    def usage_totals(self):
        return {"input_tokens": 1200, "output_tokens": 300, "calls": 6}

    def classify_with_llm(self, context):
        return self._classify_reply

    def extract_document_fields(self, doc_type, document_id, raw_text, fields):
        return {
            f: {"value": None, "confidence": 0.0} for f in fields
        }

    def diagnose(self, context):
        return {"summary": "LLM diagnosis over the computed match.", "risk_factors": ["llm factor"]}

    def diagnose_with_tools(self, context, tool_specs, dispatch, max_tool_calls):
        if not self._tool_loop:
            return None
        text, summary = dispatch("match_facts", {})
        self.tool_names_seen.append("match_facts")
        text2, summary2 = dispatch("search_policies", {"query": "three-way match tolerance"})
        self.tool_names_seen.append("search_policies")
        return {
            "summary": "Agentic diagnosis composed after tool calls.",
            "risk_factors": ["tool-backed factor"],
            "tool_calls": [
                {"name": "match_facts", "summary": summary},
                {"name": "search_policies", "summary": summary2},
            ],
        }

    def propose_options(self, context):
        return [
            {"kind": "approve_for_payment_prep", "title": "Prepare for payment", "description": "Clean match."},
            {"kind": "hold_for_review", "title": "Hold", "description": "Hold for review."},
            {"kind": "not_a_real_kind", "title": "Bogus", "description": "Must be filtered."},
        ]

    def verify_draft(self, context):
        return {"grounded": True, "issues": [], "summary": "checked"}

    def review_draft(self, context):
        return {"verdict": "pass", "findings": ["reviewed by the fake"], "model": "fake-reviewer"}


def _run(backend, invoice=SAMPLE):
    return run_invoice(invoice, backend=backend, retriever=KeywordRetriever())


def test_crosscheck_agreement_keeps_rules():
    backend = FakeLLMBackend(
        classify_reply={
            "outcome": "clean_match", "severity": "low", "confidence": 0.9,
            "rationale": "matches", "backend": "fake-llm",
        }
    )
    result = _run(backend)
    assert result.cross_check.resolution == "agree"
    assert result.triage.outcome == "clean_match"


def test_crosscheck_disagreement_rules_authoritative_when_rules_strong():
    backend = FakeLLMBackend(
        classify_reply={
            "outcome": "price_variance", "severity": "high", "confidence": 0.99,
            "rationale": "the LLM disagrees", "backend": "fake-llm",
        }
    )
    result = _run(backend)
    # The rules triaged clean_match at 0.95 — strong — so they stand
    # even against a confident LLM disagreement.
    assert result.triage.outcome == "clean_match"
    assert result.cross_check.resolution == "rules_authoritative"


def test_crosscheck_adoption_when_rules_weak():
    rule = Triage(outcome="missing_po", severity="moderate", confidence=0.6, rationale="weak rule")
    llm = {
        "outcome": "price_variance", "severity": "high", "confidence": 0.95,
        "rationale": "prices disagree", "backend": "fake-llm",
    }
    final, cross = resolve_triage(rule, llm, "fake-llm")
    assert cross.resolution == "llm_adopted"
    assert final.outcome == "price_variance"
    assert any("llm-adopted" in s for s in final.signals)


def test_agentic_diagnosis_uses_tools():
    backend = FakeLLMBackend()
    result = _run(backend)
    assert result.diagnosis.source == "llm"
    assert backend.tool_names_seen == ["match_facts", "search_policies"]
    assert [c["name"] for c in result.diagnosis.tool_calls] == ["match_facts", "search_policies"]


def test_llm_options_are_validated_and_scored_in_code():
    backend = FakeLLMBackend()
    result = _run(backend)
    kinds = [o.kind for o in result.options]
    assert "not_a_real_kind" not in kinds
    assert all(o.score > 0 or o.kind != "approve_for_payment_prep" for o in result.options)
    recommended = next(o for o in result.options if o.recommended)
    assert recommended.kind == "approve_for_payment_prep"


def test_unusable_llm_diagnosis_falls_back_to_template():
    backend = FakeLLMBackend(tool_loop=False)
    backend.diagnose = lambda context: None
    result = _run(backend)
    assert result.diagnosis.source == "template"
    assert "unusable" in (result.diagnosis.note or "")


def test_telemetry_counts_fake_provider_tokens():
    backend = FakeLLMBackend()
    result = _run(backend)
    assert result.telemetry.input_tokens == 1200
    assert result.telemetry.output_tokens == 300


# -- NIM ports via fakes ---------------------------------------------------


class FakeNimOcr:
    name = "nim"

    def read(self, document):
        return OcrResult(
            document_id=document.document_id,
            doc_type=document.doc_type,
            engine="nim",
            mean_confidence=0.42,
            char_count=len(document.text_layer),
            low_confidence=True,
            note="fake NIM OCR read",
        )


class FakeNimVision:
    name = "vlm"

    def verify(self, invoice, ocr_results):
        return VisionResult(
            document_type_detected="invoice",
            is_invoice=True,
            legible=False,
            quality="unreadable",
            confidence=0.9,
            source="vlm",
            note="fake VLM verdict",
        )


def test_nim_ports_plug_into_the_pipeline():
    service = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        ocr_engine=FakeNimOcr(),
        vision_engine=FakeNimVision(),
    )
    result = service.analyze(SAMPLE)
    assert result.ocr_results[0].engine == "nim"
    assert result.vision.source == "vlm"
    # A VLM saying "unreadable" triages unreadable even on a clean scan.
    assert result.triage.outcome == "unreadable"


def test_extraction_fallback_when_llm_returns_nothing_useful():
    backend = FakeLLMBackend()
    result = _run(backend)
    # The fake returns value=None for every field: the extractor marks
    # them missing_in_text or falls back — the run must not fail and
    # the provided-fields path still produces extractions.
    assert result.extractions
    assert result.triage.outcome == "clean_match"


def test_invoice_payload_roundtrip():
    payload = json.loads(SAMPLE.model_dump_json())
    again = InvoiceInput.model_validate(payload)
    assert again.invoice_id == SAMPLE.invoice_id
