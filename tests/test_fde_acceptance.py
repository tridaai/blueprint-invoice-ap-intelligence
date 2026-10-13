"""POC acceptance tests — the failure modes a customer's AP lead tries first.

Each test names one way the agent could cost money, mislead an
approver, or let a supplier's text past a control. They run offline
(mock backend or scripted fakes) and are the gate the POC hardening
round is measured against:

- money: duplicates and fraud-held invoices never reach a payment run
  unacknowledged; overbilling split across lines or hidden in a
  stated line total is caught; an approval carries a reason.
- grounding: the LLM self-check and reviewer see the policies the
  drafts were grounded in, and judge the drafts the approver reads.
- guardrails: payment promises, invented figures and invented
  policies are caught however the text is punctuated.
"""

from __future__ import annotations

import copy
import sys

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.extractor import _cross_check
from ap_agent.graph import run_invoice
from ap_agent.guardrails import validate_drafts
from ap_agent.model_backends import MockModelBackend
from ap_agent.retriever import KeywordRetriever
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import DocumentInput, Drafts, InvoiceInput
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
CLEAN = SAMPLES["INV-1001"]  # 10 x A4 @ 45 + 5 x toner @ 110, tax 180, total 1180
ALLOWED = [1180.0, 1000.0, 180.0, 450.0, 550.0, 45.0, 110.0]
RETRIEVED = ["AP-MATCH-01", "AP-APPR-01"]


@pytest.fixture()
def service() -> InvoiceService:
    return InvoiceService(store=InMemoryStore(), checkpointer=False)


def _variant(invoice: InvoiceInput, **changes) -> InvoiceInput:
    data = copy.deepcopy(invoice.model_dump())
    data.update(changes)
    return InvoiceInput.model_validate(data)


def _validate(letter: str, packet: str = "AP packet INV-1001 [AP-MATCH-01]. Next step: approver review."):
    return validate_drafts(
        CLEAN,
        Drafts(subject="Invoice INV-1001", supplier_letter=letter, ap_packet=packet),
        RETRIEVED,
        allowed_amounts=ALLOWED,
        outcome="clean_match",
        match_status="full_match",
    )


def _failed(validation) -> set[str]:
    return {c.name for c in validation.checks if not c.passed}


# ---------------------------------------------------------------------------
# Money: what reaches a payment run, and what an approval must carry
# ---------------------------------------------------------------------------


def test_approval_requires_a_reason(service):
    service.analyze(CLEAN)
    with pytest.raises(ValueError, match="reason"):
        service.approve("INV-1001", approver="Priya Nair", reason="")
    with pytest.raises(ValueError, match="reason"):
        service.approve("INV-1001", approver="Priya Nair", reason="   ")


def test_api_refuses_an_approval_without_a_reason(service):
    client = TestClient(create_app(service))
    service.analyze(CLEAN)
    response = client.post("/invoices/INV-1001/approve", json={"actor": "priya"})
    assert response.status_code == 422
    assert "reason" in response.json()["detail"]


def test_a_suspected_duplicate_cannot_be_approved_without_acknowledging_the_hold(service):
    service.analyze(CLEAN)
    service.approve("INV-1001", approver="Priya Nair", reason="clean match")
    held = service.analyze(SAMPLES["INV-1004"])
    assert held.facts.duplicate_suspected
    with pytest.raises(ValueError, match="duplicate"):
        service.approve("INV-1004", approver="Priya Nair", reason="looks fine")


def test_a_fraud_held_invoice_cannot_be_approved_without_acknowledging_the_hold(service):
    held = service.analyze(SAMPLES["INV-1010"])
    assert held.bank_change_blocked
    with pytest.raises(ValueError, match="bank"):
        service.approve("INV-1010", approver="Priya Nair", reason="looks fine")


def test_an_acknowledged_hold_is_approvable_and_recorded(service):
    service.analyze(SAMPLES["INV-1010"])
    decided = service.approve(
        "INV-1010",
        approver="Priya Nair",
        reason="bank details verified by phone with Harborline's controller",
        acknowledge_holds=True,
    )
    assert decided.status == "approved"
    assert decided.holds_acknowledged == ["bank_change_blocked"]


def test_unacknowledged_holds_never_reach_a_payment_run(service):
    service.analyze(CLEAN)
    service.approve("INV-1001", approver="Priya Nair", reason="clean match")
    service.analyze(SAMPLES["INV-1004"])
    service.analyze(SAMPLES["INV-1010"])
    # A record approved before this rule existed (or written straight to
    # the store) must still be kept out of the pay list.
    store = service._get_store()
    for invoice_id in ("INV-1004", "INV-1010"):
        record = store.get(invoice_id)
        record.result.status = "approved"
        store.save(record)
    runs = service.propose_payment_runs(window_days=60, as_of="2026-10-01")
    members = {line.invoice_id for run in runs for line in run.lines}
    assert members == {"INV-1001"}


def test_split_line_overbilling_is_not_a_clean_match():
    # 10 boxes ordered and received; billed as two lines of 10.
    lines = [
        {"sku": "PAP-A4", "description": "A4 paper case", "quantity": 10, "unit_price": 45.0},
        {"sku": "PAP-A4", "description": "A4 paper case", "quantity": 10, "unit_price": 45.0},
        {"sku": "TON-220", "description": "Toner cartridge", "quantity": 5, "unit_price": 110.0},
    ]
    invoice = _variant(CLEAN, invoice_id="INV-SPLIT", line_items=lines,
                       total_amount=1711.0, tax_amount=261.0)
    result = run_invoice(invoice)
    assert result.triage.outcome == "quantity_variance"
    assert result.facts.match_status != "full_match"


def test_a_stated_line_total_that_disagrees_with_qty_times_price_is_caught():
    lines = copy.deepcopy([line.model_dump() for line in CLEAN.line_items])
    lines[0]["line_total"] = 4500.0  # 10 x 45 is 450
    invoice = _variant(CLEAN, invoice_id="INV-LINETOTAL", line_items=lines,
                       total_amount=5230.0, tax_amount=180.0)
    result = run_invoice(invoice)
    assert result.triage.outcome == "price_variance"
    assert result.facts.match_status != "full_match"
    assert any("line total" in m.detail for m in result.facts.line_matches)


# ---------------------------------------------------------------------------
# Grounding: the self-check and the reviewer judge what the approver reads
# ---------------------------------------------------------------------------


class _RecordingBackend(MockModelBackend):
    """Template drafts, plus LLM verify/review hooks that record what
    they were shown. ``flawed_first`` makes the first draft fail a
    guardrail so the repair loop replaces it."""

    name = "recording"

    def __init__(self, flawed_first: bool = False):
        super().__init__()
        self.flawed_first = flawed_first
        self.verify_contexts: list[dict] = []
        self.review_contexts: list[dict] = []

    def draft_documents(self, context):
        subject, letter, packet = super().draft_documents(context)
        if self.flawed_first and not context.get("repair_feedback"):
            letter += "\nWe will pay this invoice."
        return subject, letter, packet

    def verify_draft(self, context):
        self.verify_contexts.append(dict(context))
        return {"grounded": True, "issues": [], "summary": "checked"}

    def review_draft(self, context):
        self.review_contexts.append(dict(context))
        return {"verdict": "pass", "findings": [], "model": "recording"}


def test_self_check_and_reviewer_see_the_retrieved_policies():
    backend = _RecordingBackend()
    result = run_invoice(CLEAN, backend=backend, retriever=KeywordRetriever())
    retrieved = [p.policy_id for p in result.policies]
    assert retrieved
    for context in (*backend.verify_contexts, *backend.review_contexts):
        shown = [p["policy_id"] for p in context["policy_details"]]
        assert shown == retrieved


def test_self_check_and_reviewer_judge_the_repaired_drafts():
    backend = _RecordingBackend(flawed_first=True)
    result = run_invoice(CLEAN, backend=backend, retriever=KeywordRetriever())
    assert result.repaired
    assert "We will pay" not in result.drafts.supplier_letter
    assert backend.verify_contexts and backend.review_contexts
    assert backend.verify_contexts[-1]["supplier_letter"] == result.drafts.supplier_letter
    assert backend.review_contexts[-1]["supplier_letter"] == result.drafts.supplier_letter


def test_extraction_compares_amounts_as_numbers():
    document = DocumentInput(doc_type="invoice", document_id="DOC-X",
                             fields={"subtotal": "1000.00", "total": "1,180.00"})
    extraction = _cross_check(document, {
        "subtotal": {"value": "1000.0", "confidence": 0.99},
        "total": {"value": 1180, "confidence": 0.99},
    })
    statuses = {f.field: f.status for f in extraction.fields}
    assert statuses["subtotal"] == "match"
    assert statuses["total"] == "match"


# ---------------------------------------------------------------------------
# Guardrails: punctuation and phrasing must not open a gap
# ---------------------------------------------------------------------------


def test_the_required_disclaimers_still_pass():
    validation = _validate(
        "Invoice INV-1001 received [AP-MATCH-01]. This letter is not a payment "
        "confirmation, and no payment is made or scheduled by this letter. "
        "Next step: approver review."
    )
    assert "no_payment_language" not in _failed(validation)


@pytest.mark.parametrize("promise", [
    "No further info needed; payment will be made on Friday.",
    "Nothing else is required. Payment is scheduled for 30 October.",
    "Your invoice is scheduled for payment next week.",
    "We will release payment once the approver signs.",
])
def test_payment_promises_are_caught_however_phrased(promise):
    validation = _validate(f"Invoice INV-1001 [AP-MATCH-01]. {promise} Next step: review.")
    assert "no_payment_language" in _failed(validation)


@pytest.mark.parametrize("citation", [
    "[AP-FAKE-99]",
    "[AP‑FAKE‑99]",  # non-breaking hyphens
    "policy AP-FAKE-99",       # no brackets
])
def test_invented_policies_are_caught_however_punctuated(citation):
    validation = _validate(f"Invoice INV-1001 per {citation}. Next step: review.")
    assert "citations_within_retrieved" in _failed(validation)


@pytest.mark.parametrize("figure", [
    "a total of 9999",
    "a variance of USD ‑9999.0",  # non-breaking hyphen as a minus
    "USD 9 999.00",               # narrow no-break space grouping
])
def test_invented_figures_are_caught_however_formatted(figure):
    validation = _validate(f"Invoice INV-1001 [AP-MATCH-01] states {figure}. Next step: review.")
    assert "figures_must_trace" in _failed(validation)


def test_payment_terms_are_not_mistaken_for_amounts():
    validation = _validate(
        "Invoice INV-1001 [AP-MATCH-01] on terms 2/10 net 30, total USD 1,180.00. Next step: review."
    )
    assert "figures_must_trace" not in _failed(validation)


def test_the_supplier_letter_itself_must_state_the_next_step():
    validation = _validate("Invoice INV-1001 received [AP-MATCH-01].")
    assert "states_next_step" in _failed(validation)


def test_the_drafts_must_cite_a_policy():
    validation = _validate("Invoice INV-1001 received. Next step: review.",
                           packet="AP packet INV-1001. Next step: approver review.")
    assert "policy_citations_present" in _failed(validation)


# ---------------------------------------------------------------------------
# Drafts never echo bank details; the governing policy is always citable
# ---------------------------------------------------------------------------


def test_bank_details_restated_by_the_supplier_are_never_echoed_and_the_case_is_approvable(service):
    # INV-1014's note restates the account on file — a legitimate case
    # the template used to quote word for word, digits included, which
    # left it unapprovable.
    result = service.analyze(SAMPLES["INV-1014"])
    drafts = f"{result.drafts.supplier_letter}\n{result.drafts.ap_packet}"
    assert "5520109873" not in drafts and "20-44-18" not in drafts
    assert result.validation.passed
    decided = service.approve("INV-1014", approver="Priya Nair", reason="details match the master record")
    assert decided.status == "approved"


class _NotAnInvoiceVision:
    name = "vlm"

    def verify(self, invoice, ocr_results):
        from ap_agent.schemas import VisionResult

        return VisionResult(document_type_detected="other", is_invoice=False, legible=True,
                            quality="good", confidence=0.9, source="vlm", note="fake verdict")


def test_a_non_invoice_cites_the_document_type_policy_and_passes_its_guardrails():
    photo = _variant(
        CLEAN, invoice_id="INV-BERRY", line_items=[], purchase_order=None, goods_receipt=None,
        total_amount=None, tax_amount=None,
        documents=[{"doc_type": "invoice", "document_id": "DOC-BERRY",
                    "ocr_text": "FRESH BLUEBERRIES\nProduct of Peru\n125 g", "ocr_confidence": 0.93}],
    )
    result = run_invoice(photo, vision_engine=_NotAnInvoiceVision())
    assert result.triage.outcome == "not_an_invoice"
    assert "AP-DOC-02" in [p.policy_id for p in result.policies]
    assert result.validation.passed, [c.detail for c in result.validation.checks if not c.passed]


# ---------------------------------------------------------------------------
# A busy hosted endpoint is retried; a refused one is not
# ---------------------------------------------------------------------------


class _StatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_transient_provider_errors_are_retried_with_backoff(monkeypatch):
    from ap_agent.resilience import call_with_transient_retry

    monkeypatch.setenv("NIM_PORT_MAX_RETRIES", "2")
    monkeypatch.setenv("NIM_PORT_RETRY_BACKOFF_SECONDS", "1")
    replies = [_StatusError(503), _StatusError(429), "ok"]
    slept: list[float] = []

    def call():
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    assert call_with_transient_retry(call, sleep=slept.append) == "ok"
    assert slept == [1.0, 2.0]


def test_auth_errors_are_never_retried(monkeypatch):
    from ap_agent.resilience import call_with_transient_retry

    calls = []

    def call():
        calls.append(1)
        raise _StatusError(403)

    with pytest.raises(_StatusError):
        call_with_transient_retry(call, sleep=lambda _: None)
    assert len(calls) == 1


def test_the_vision_port_survives_a_busy_endpoint(monkeypatch):
    from types import SimpleNamespace

    from ap_agent import resilience
    from ap_agent.vision import NimVision

    verdict = '{"document_type": "other", "is_invoice": false, "legible": true, "quality": "good", "confidence": 0.9}'
    replies = [_StatusError(503), verdict]

    def create(**_):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    def factory(**_):  # stands in for openai.OpenAI — no SDK, no network
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=factory))
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("NIM_PORT_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setattr(resilience.time, "sleep", lambda _: None)
    result = NimVision().verify(CLEAN, [])
    assert not replies  # the 503 was retried, and the verdict consumed
    assert result.source == "vlm" and result.is_invoice is False


# ---------------------------------------------------------------------------
# A decision stands: re-analysis never erases it; overrides say why
# ---------------------------------------------------------------------------


def test_re_analysing_a_decided_invoice_is_refused_and_the_decision_stands(service):
    service.analyze(CLEAN)
    service.approve("INV-1001", approver="Priya Nair", reason="clean three-way match")
    service.analyze(SAMPLES["INV-1004"])
    with pytest.raises(ValueError, match="already decided"):
        service.analyze(CLEAN)
    record = service._get_store().get("INV-1001").result
    assert record.status == "approved" and record.decided_by == "Priya Nair"
    assert not record.facts.duplicate_suspected


def test_api_answers_409_when_a_decided_invoice_is_re_analysed(service):
    client = TestClient(create_app(service))
    client.post("/invoices/analyze", json=CLEAN.model_dump(mode="json"))
    client.post("/invoices/INV-1001/approve", json={"actor": "priya", "reason": "clean match"})
    again = client.post("/invoices/analyze", json=CLEAN.model_dump(mode="json"))
    assert again.status_code == 409
    assert "already decided" in again.json()["detail"]


def test_re_analysing_an_original_never_makes_it_a_duplicate_of_its_own_copy(service):
    service.analyze(CLEAN)
    service.analyze(SAMPLES["INV-1004"])  # the copy arrives second
    again = service.analyze(CLEAN)        # the original, re-run while undecided
    assert not again.facts.duplicate_suspected
    assert service._get_store().get("INV-1004").result.facts.duplicate_of == "INV-1001"


def test_overriding_a_hold_needs_a_specific_reason(service):
    service.analyze(SAMPLES["INV-1010"])
    with pytest.raises(ValueError, match="how the hold was resolved"):
        service.approve("INV-1010", approver="madhav", reason="every thing is fine",
                        acknowledge_holds=True)
    decided = service.approve(
        "INV-1010", approver="madhav",
        reason="Bank change verified by phone with Harborline's finance controller",
        acknowledge_holds=True,
    )
    assert decided.status == "approved"
