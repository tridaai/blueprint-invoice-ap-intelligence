"""Reasoning-budget audit: every provider call carries its audited
completion budget.

Reasoning models think inside the budget they answer in. The
pre-audit budgets were answer-sized — 300 for the classify call,
400 for extraction, 300 for the vision verdict (whose live
verdicts arrived with thin headroom, flagged in self-QA) — and a
starved call returns an empty or truncated reply that the pipeline
records as a malformed model reply. These tests pin, for every
provider call, the budget it actually carries: the audited default
(model_backends.COMPLETION_BUDGETS, docs/architecture.md §10) and
the env override that retunes it.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from ap_agent.config import completion_budget
from ap_agent.model_backends import (
    COMPLETION_BUDGETS,
    AnthropicBackend,
    DraftContext,
    OpenAIBackend,
    _BaseLLMBackend,
    completion_budgets_in_force,
)
from ap_agent.samples import load_sample_invoices
from ap_agent.schemas import InvoiceInput


class RecordingBackend(_BaseLLMBackend):
    """A backend whose _complete records the budget it was handed
    and answers from a queue of canned replies."""

    name = "recording"

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.budgets: list[int] = []
        self._model = "recording-model"

    def _complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        self.budgets.append(max_tokens)
        return self._replies.pop(0)


def _context() -> DraftContext:
    return DraftContext(
        invoice_id="INV-1",
        supplier_name="Northbeam Office Supplies",
        invoice_number="NB-1",
        missing_items=["the purchase order for this invoice"],
    )


def _run_every_call(backend: RecordingBackend) -> None:
    ctx = _context()
    backend.classify_with_llm(ctx)
    backend.extract_document_fields(
        "invoice", "DOC-1", "raw text", ["invoice_number"]
    )
    backend.draft_documents(ctx)
    backend.diagnose(ctx)
    backend.propose_options(ctx)
    backend.compose_information_request(ctx)
    backend.verify_draft(dict(ctx))
    backend.review_draft(dict(ctx))


_REPLIES = [
    '{"outcome": "missing_po", "severity": "moderate", '
    '"confidence": 0.9, "rationale": "no PO attached"}',
    '{"invoice_number": {"value": "NB-1", "confidence": 0.9}}',
    "Subject: Invoice INV-1 — under review\n\nDear supplier.\n"
    "---PACKET---\npacket",
    '{"summary": "A summary.", "risk_factors": ["a factor"]}',
    '[{"kind": "hold_for_review", "title": "Hold", '
    '"description": "Hold for review."}]',
    "Dear supplier, please send the purchase order.",
    '{"grounded": true, "issues": [], "summary": "ok"}',
    '{"verdict": "pass", "findings": []}',
]

_CALL_ORDER = [
    "classify",
    "extract",
    "draft",
    "diagnose",
    "options",
    "info_request",
    "verify",
    "review",
]


def test_every_call_carries_its_audited_default():
    backend = RecordingBackend(_REPLIES)
    _run_every_call(backend)
    expected = [COMPLETION_BUDGETS[call][1] for call in _CALL_ORDER]
    assert backend.budgets == expected
    # The audited defaults themselves: no call is left at a
    # pre-audit answer-sized budget.
    assert completion_budgets_in_force() == {
        call: COMPLETION_BUDGETS[call][1] for call in COMPLETION_BUDGETS
    }
    assert min(completion_budgets_in_force().values()) >= 1000


def test_env_override_retunes_one_call_only(monkeypatch):
    monkeypatch.setenv("LLM_CLASSIFY_MAX_TOKENS", "432")
    monkeypatch.setenv("LLM_DRAFT_MAX_TOKENS", "999")
    backend = RecordingBackend(_REPLIES)
    _run_every_call(backend)
    by_call = dict(zip(_CALL_ORDER, backend.budgets))
    assert by_call["classify"] == 432
    assert by_call["draft"] == 999
    assert by_call["extract"] == COMPLETION_BUDGETS["extract"][1]
    assert completion_budgets_in_force()["classify"] == 432


def test_budget_floor_and_invalid_values(monkeypatch):
    assert completion_budget("UNSET_BUDGET_VAR", 700) == 700
    monkeypatch.setenv("LLM_VERIFY_MAX_TOKENS", "0")
    assert completion_budgets_in_force()["verify"] == 1
    monkeypatch.setenv("LLM_VERIFY_MAX_TOKENS", "not-a-number")
    assert completion_budgets_in_force()["verify"] == (
        COMPLETION_BUDGETS["verify"][1]
    )


# -- the diagnosis tool loops (one budget per turn, both providers) --


class _FakeOpenAIClient:
    def __init__(self):
        self.calls: list[dict] = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                message = SimpleNamespace(content="composed", tool_calls=None)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message)], usage=None
                )

        self.chat = SimpleNamespace(completions=_Completions())


class _FakeAnthropicClient:
    def __init__(self):
        self.calls: list[dict] = []

        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                block = SimpleNamespace(type="text", text="composed")
                return SimpleNamespace(content=[block], usage=None)

        self.messages = _Messages()


def test_tool_loops_carry_the_diagnose_budget():
    openai_backend = OpenAIBackend.__new__(OpenAIBackend)
    openai_backend._model = "fake-model"
    openai_backend._client = _FakeOpenAIClient()
    text, _log = openai_backend._tool_loop(
        "system", "user", [], lambda name, args: ("", ""), 0
    )
    assert text == "composed"
    assert openai_backend._client.calls[0]["max_tokens"] == (
        COMPLETION_BUDGETS["diagnose"][1]
    )

    anthropic_backend = AnthropicBackend.__new__(AnthropicBackend)
    anthropic_backend._model = "fake-model"
    anthropic_backend._client = _FakeAnthropicClient()
    text, _log = anthropic_backend._tool_loop(
        "system", "user", [], lambda name, args: ("", ""), 0
    )
    assert text == "composed"
    assert anthropic_backend._client.calls[0]["max_tokens"] == (
        COMPLETION_BUDGETS["diagnose"][1]
    )


# -- the NIM adapters: vision verdict + OCR read ----------------------


class _FakeNimClient:
    instances: list = []

    def __init__(self, **kwargs):
        self.calls: list[dict] = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                content = outer.reply
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=content)
                        )
                    ]
                )

        self.chat = SimpleNamespace(completions=_Completions())
        self.reply = ""
        _FakeNimClient.instances.append(self)


def _install_fake_openai(monkeypatch, reply: str):
    _FakeNimClient.instances = []

    def factory(**kwargs):
        client = _FakeNimClient(**kwargs)
        client.reply = reply
        return client

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=factory))
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")


def test_vision_adapter_carries_the_audited_budget(monkeypatch):
    from ap_agent.vision import DEFAULT_VISION_MAX_TOKENS, NimVision

    _install_fake_openai(
        monkeypatch,
        '{"document_type": "invoice", "is_invoice": true, '
        '"legible": true, "quality": "good", "confidence": 0.9}',
    )
    invoice = InvoiceInput.model_validate(load_sample_invoices()[0])
    result = NimVision().verify(invoice, [])
    assert result.source == "vlm"
    client = _FakeNimClient.instances[-1]
    assert client.calls[0]["max_tokens"] == DEFAULT_VISION_MAX_TOKENS == 1000

    monkeypatch.setenv("VISION_MAX_TOKENS", "777")
    NimVision().verify(invoice, [])
    assert _FakeNimClient.instances[-1].calls[0]["max_tokens"] == 777


def test_ocr_adapter_carries_the_audited_budget(monkeypatch):
    from ap_agent.ocr import DEFAULT_OCR_MAX_TOKENS, NimOcrEngine

    _install_fake_openai(
        monkeypatch, '{"text": "INVOICE text", "confidence": 0.9}'
    )
    invoice = InvoiceInput.model_validate(load_sample_invoices()[0])
    result = NimOcrEngine().read(invoice.documents[0])
    assert result.engine == "nim"
    client = _FakeNimClient.instances[-1]
    assert client.calls[0]["max_tokens"] == DEFAULT_OCR_MAX_TOKENS == 2000

    monkeypatch.setenv("OCR_MAX_TOKENS", "1500")
    NimOcrEngine().read(invoice.documents[0])
    assert _FakeNimClient.instances[-1].calls[0]["max_tokens"] == 1500


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
