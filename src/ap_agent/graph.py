"""LangGraph pipeline for the invoice / AP document intelligence agent.

    ocr → vision → screen → extract → match → classify → ┌ retrieve ┐
                                                          ├ discrepancies ┤ (fan-out)
                                                          └ history ┘
                          → diagnose → options → draft → verify → review
                          → validate → gate → approval_gate → END

The three evidence branches (policy retrieval, extraction cross-check
discrepancies, supplier memory + approver feedback) are independent
reads over state the earlier nodes already produced, so they run as a
parallel fan-out; the diagnosis joins them. The joined state is
identical to a sequential run — the fan-out is an execution detail,
not a behaviour change.

Responsibilities are split on purpose:

- **Code computes**: OCR confidence handling, the three-way match,
  duplicate detection, triage signals, option scores, guardrail
  checks. The model never invents these numbers.
- **The model reads and drafts**: document extraction, an independent
  triage (cross-checked against the rules), exception prose, option
  wording, letters, self-verification, independent review — in
  provider mode. In the default mode deterministic templates take
  those paths.
- **Humans decide**: the graph ends at a checkpointed gate. Nothing
  is approved, paid, sent, or filed by any node; approval is recorded
  by the service layer and only then may an opt-in webhook route the
  packet to the customer's system of choice.

Resilience: provider-backed nodes run under a per-node timeout with
bounded retries for the steps that are safe to repeat (see
``resilience.py``); drafting and the gate are never retried.

Token budget: with ``RUN_TOKEN_BUDGET`` set, provider steps degrade
to their deterministic paths once cumulative usage crosses the budget
— the run completes on templates instead of failing (see
``_BudgetGuard``).
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from .autonomy import recommend_autonomy
from .clarify import build_information_request
from .config import env_int, env_str, load_dotenv
from .crosscheck import resolve_triage
from .diagnosis import build_diagnosis, memory_evidence_lines
from .discounts import evaluate_discount, format_discount_line
from .errors import ProviderError
from .events import NullSink, RunEvent
from .extractor import extract_documents, extraction_discrepancies
from .guardrails import cited_policies, repair_directive, validate_drafts
from .policies_data import POLICIES as _POLICY_CORPUS
from .matching import compute_match, format_match_summary
from .model_backends import (
    DraftContext,
    estimate_cost_usd,
    run_completion_budgets,
)
from .ocr import get_ocr_engine, run_ocr
from .options import build_options
from .resilience import ResilientBackend
from .retriever import get_retriever
from .reviewer import review_drafts
from .schemas import (
    AgentResult,
    BankMasterCheck,
    CrossCheck,
    Diagnosis,
    DiscountEvaluation,
    DocumentExtraction,
    Drafts,
    MatchFacts,
    InvoiceInput,
    OcrResult,
    RecoveryOption,
    RetrievedPolicy,
    ReviewResult,
    TraceStep,
    Triage,
    Validation,
    VerifyResult,
    VisionResult,
)
from .screening import fraud_flag_label, sanitized_notes, screen_fraud, screen_invoice
from .store import find_duplicate
from .supplier_master import compare_bank_details, mismatch_flag
from .tools_agent import DiagnosisToolBox
from .triage import classify_invoice
from .verify import verify_drafts
from .vision import get_vision_engine, run_vision

# Backend method -> the pipeline node it serves. The budget guard
# degrades per node: once the budget is exceeded, that node's provider
# calls stop and its deterministic path runs instead.
_METHOD_NODE = {
    "extract_document_fields": "extract",
    "classify_with_llm": "classify",
    "diagnose": "diagnose",
    "diagnose_with_tools": "diagnose",
    "propose_options": "options",
    "draft_documents": "draft",
    "verify_draft": "verify",
    "review_draft": "review",
    "compose_information_request": "gate",
}

# Nodes whose provider steps may degrade under a token budget. The
# gate node itself (autonomy, the human decision) never degrades —
# only its optional model call (the information-request wording) does.
_PROVIDER_STEP_NODES = {
    "extract", "classify", "diagnose", "options", "draft", "verify", "review", "gate",
}


class _BudgetGuard:
    """Backend proxy enforcing ``RUN_TOKEN_BUDGET`` per run.

    Wraps the run's backend: before each provider call in a degradable
    node it checks the cumulative usage reported so far; once usage
    exceeds the budget, the optional capability methods (extraction,
    triage cross-check, diagnosis, options, verify, review, info-
    request wording) disappear — the nodes read a missing method as
    "use the deterministic path", exactly like the mock backend — and
    ``draft_documents`` degrades to the template backend. The run
    completes; it never fails for budget reasons. ``status()`` reports
    the final accounting for telemetry.
    """

    def __init__(self, backend, template_backend, limit: int | None) -> None:
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_template", template_backend)
        object.__setattr__(self, "_limit", limit)
        object.__setattr__(self, "_exceeded", False)

    def _over_budget(self) -> bool:
        if self._limit is None:
            return False
        if self._exceeded:
            return True
        usage = self._backend.usage_totals()
        used = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
        if used > self._limit:
            object.__setattr__(self, "_exceeded", True)
            return True
        return False

    def status(self) -> dict | None:
        if self._limit is None:
            return None
        usage = self._backend.usage_totals()
        used = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
        return {"used": used, "limit": self._limit, "exceeded": used > self._limit}

    def __getattr__(self, name: str):
        node = _METHOD_NODE.get(name)
        if node is not None and node in _PROVIDER_STEP_NODES and self._over_budget():
            if name == "draft_documents":
                return getattr(self._template, name)
            return None  # capability absent -> deterministic path
        return getattr(self._backend, name)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._backend, name, value)


def _token_budget() -> int | None:
    load_dotenv()
    value = env_int("RUN_TOKEN_BUDGET", 0)
    return value if value and value > 0 else None


def _wrap_backend(backend):
    """Apply the per-node resilience policy (timeout + safe retries)."""
    return ResilientBackend(backend)


def _guardrail_repair_enabled() -> bool:
    load_dotenv()
    return env_str("GUARDRAIL_REPAIR", "on").strip().lower() not in {"off", "0", "false", "no"}


def _guardrail_repair_max_attempts() -> int:
    load_dotenv()
    return min(max(env_int("GUARDRAIL_REPAIR_MAX_ATTEMPTS", 1), 0), 3)


class AgentState(TypedDict, total=False):
    invoice: dict[str, Any]
    duplicate_of: str
    ocr_results: list[dict[str, Any]]
    vision: Optional[dict[str, Any]]
    injection_flags: list[str]
    fraud_flags: list[str]
    bank_change_blocked: bool
    bank_master_check: Optional[dict[str, Any]]
    extractions: list[dict[str, Any]]
    facts: dict[str, Any]
    triage: dict[str, Any]
    rule_triage: dict[str, Any]
    cross_check: Optional[dict[str, Any]]
    policies: list[dict[str, Any]]
    retrieval_info: str
    discrepancies: list[str]
    history_evidence: list[str]
    history: Optional[dict[str, Any]]
    priors: list[dict[str, Any]]
    diagnosis: dict[str, Any]
    discount: Optional[dict[str, Any]]
    options: list[dict[str, Any]]
    options_notes: str
    drafts: dict[str, Any]
    verification: dict[str, Any]
    review: dict[str, Any]
    reviewer_blocked: bool
    validation: dict[str, Any]
    repair_attempted: bool
    repaired: bool
    repair_attempts: int
    original_validation: Optional[dict[str, Any]]
    autonomy: dict[str, Any]
    needs_information: bool
    information_request: Optional[dict[str, Any]]
    token_budget: Optional[dict[str, Any]]
    gate_status: str


# ---------------------------------------------------------------------------
# Shared views over state
# ---------------------------------------------------------------------------


def _screen_text(sub: InvoiceInput) -> str:
    """The supplier's text as prompts may see it: flagged sentences removed."""
    return sanitized_notes(sub)


def _document_lines(state: AgentState) -> list[str]:
    """OCR + vision outcomes as evidence lines (diagnosis, prompts, trace)."""
    lines: list[str] = []
    for r in state.get("ocr_results", []):
        lines.append(
            f"OCR {r.get('document_id') or r.get('doc_type')}: engine "
            f"{r.get('engine')}, mean confidence {r.get('mean_confidence')}"
            + (" — LOW CONFIDENCE" if r.get("low_confidence") else "")
        )
    vision = state.get("vision")
    if vision:
        lines.append(
            f"vision check: detected {vision.get('document_type_detected')}, "
            f"is_invoice={vision.get('is_invoice')}, legible={vision.get('legible')}, "
            f"quality {vision.get('quality')} (source: {vision.get('source')})"
        )
    return lines


def _facts_view(state: AgentState, sub: InvoiceInput) -> dict[str, Any]:
    """The flat facts dict the diagnosis / options / verify layers read."""
    facts = MatchFacts.model_validate(state["facts"])
    allowed: list[float] = []
    for value in (
        facts.invoice_total_stated,
        facts.invoice_lines_total,
        facts.tax_stated,
        facts.tax_expected,
        abs(facts.tax_variance) if facts.tax_variance else None,
        sub.total_amount,
        sub.tax_amount,
    ):
        if value is not None:
            allowed.append(float(value))
    for m in facts.line_matches:
        allowed.append(round(m.invoice_qty * m.invoice_price, 2))
        allowed.append(round(m.po_qty * m.po_price, 2))
        allowed.append(float(m.invoice_price))
        allowed.append(float(m.po_price))
    discount = state.get("discount")
    if discount:
        # The discount computation is code too: its figures are
        # grounded amounts the drafts may quote.
        allowed += [
            float(discount["gross_discount"]),
            float(discount["capital_cost"]),
            float(discount["net_saving"]),
        ]
    # Option scores are computed figures too, and the packet prints
    # them — the grounding registry covers every number code produced
    # for this invoice, so a draft quoting them is grounded by
    # construction. (Options exist by the time drafts are verified.)
    allowed += [float(o["score"]) for o in state.get("options", [])]
    return {
        "match_status": facts.match_status,
        "match_score": facts.match_score,
        "match_summary": format_match_summary(facts),
        "line_detail": [
            f"L{m.line_no} {m.description}: {m.status} — {m.detail}"
            for m in facts.line_matches
        ],
        "discrepancies": list(facts.discrepancies),
        "total_amount": sub.total_amount,
        "duplicate_of": facts.duplicate_of,
        "fraud_blocked": bool(state.get("bank_change_blocked")),
        "allowed_amounts": allowed,
    }


def _classify_context(state: AgentState, sub: InvoiceInput, facts: MatchFacts) -> DraftContext:
    return DraftContext(
        invoice_id=sub.invoice_id,
        supplier_name=sub.supplier_name,
        invoice_number=sub.invoice_number,
        currency=sub.currency,
        total_amount=sub.total_amount,
        match_summary=format_match_summary(facts),
        match_status=facts.match_status,
        discrepancies=list(facts.discrepancies),
        ocr_vision="; ".join(_document_lines(state)) or "not recorded",
        supplier_text=_screen_text(sub),
        policy_details=[],
    )


def _draft_context(
    state: AgentState, sub: InvoiceInput, *, repair_feedback: str | None = None
) -> DraftContext:
    triage = Triage.model_validate(state["triage"])
    facts = MatchFacts.model_validate(state["facts"])
    view = _facts_view(state, sub)
    policies = [RetrievedPolicy.model_validate(g) for g in state.get("policies", [])]
    options = [RecoveryOption.model_validate(o) for o in state.get("options", [])]
    diagnosis = Diagnosis.model_validate(state["diagnosis"])
    recommended = next((o for o in options if o.recommended), None)
    context = DraftContext(
        invoice_id=sub.invoice_id,
        supplier_name=sub.supplier_name,
        invoice_number=sub.invoice_number,
        currency=sub.currency,
        total_amount=sub.total_amount,
        outcome=triage.outcome,
        severity=triage.severity,
        confidence=triage.confidence,
        rationale=triage.rationale,
        match_summary=view["match_summary"],
        match_status=facts.match_status,
        line_matches=[m.model_dump() for m in facts.line_matches],
        discrepancies=facts.discrepancies,
        ocr_vision="; ".join(_document_lines(state)) or "not recorded",
        duplicate_text=(
            f"suspected duplicate of {facts.duplicate_of} (same supplier, "
            "invoice number and amount already processed)"
            if facts.duplicate_suspected
            else "no duplicate suspected"
        ),
        fraud_flags=[fraud_flag_label(f) for f in state.get("fraud_flags", [])],
        bank_change_blocked=bool(state.get("bank_change_blocked")),
        supplier_text=_screen_text(sub),
        diagnosis_summary=diagnosis.summary,
        diagnosis_risk_factors=diagnosis.risk_factors,
        discount_text=(
            format_discount_line(
                DiscountEvaluation.model_validate(state["discount"]), sub.currency
            )
            if state.get("discount")
            else ""
        ),
        policy_details=[g.model_dump() for g in policies],
        citations=[g.policy_id for g in policies],
        options=[o.model_dump() for o in options],
        recommended_option_kind=recommended.kind if recommended else "",
        recommended_option_text=(
            f"{recommended.title} — {recommended.description}" if recommended else ""
        ),
    )
    if repair_feedback:
        context["repair_feedback"] = repair_feedback
    return context


def _failed_checks_text(validation: Validation) -> str:
    """The repair prompt's account of a failed draft: WHICH
    guardrail checks failed, what each one found, and the fix each
    one requires — the redraft is told the failed checks by name
    and the required element for each, not left to infer the fix
    from the failure wording."""
    lines = []
    for check in validation.checks:
        if check.passed:
            continue
        line = f"- {check.name}: {check.detail}"
        directive = repair_directive(check.name)
        if directive:
            line += f" — fix: {directive}"
        lines.append(line)
    return "\n".join(lines)


_CITATION_IN_TEXT = re.compile(r"\[(AP-[A-Z]+-\d+)\]")


def _carry_policy_citations(packet: str, citations: list[str]) -> str:
    """Carry the diagnosis's policies into the AP packet —
    deterministically, from the retrieved set the diagnosis rests
    on, never from the drafting model's memory.

    The diagnosis cites the retrieved policies; the packet is the
    document the approver reads, so a packet that drops them is
    incomplete however good its prose is — on the 2026-10-10 live
    Nemotron run the model's packet carried NONE of them and the
    independent reviewer blocked the draft. Composition, not
    prompting, closes that hole: any retrieved policy the packet
    does not already cite is appended on a ``Policy citations:``
    line at draft time (and on repair candidates), in retrieval
    order. A packet that already carries them — the offline
    template's, by construction — is returned byte-identical, and
    only retrieved policies are ever added, so the composition
    cannot trip ``citations_within_retrieved``.
    """
    present = set(_CITATION_IN_TEXT.findall(packet or ""))
    missing = [c for c in citations if c not in present]
    if not missing:
        return packet
    line = "Policy citations: " + " ".join(f"[{c}]" for c in missing)
    body = (packet or "").rstrip()
    return f"{body}\n{line}" if body else line


def _build_graph(backend, retriever, sink, checkpointer=None, ocr_engine=None, vision_engine=None, object_store=None, supplier_master=None):
    """Compile the pipeline graph around one run's collaborators."""
    ocr_engine = ocr_engine or get_ocr_engine()
    vision_engine = vision_engine or get_vision_engine()

    def node(name: str, title: str):
        """Wrap a step function as a graph node that times itself,
        emits its run event, and accumulates ``duration_ms``."""

        def wrap(fn):
            def run(state: AgentState) -> dict[str, Any]:
                started = time.perf_counter()
                update = fn(state)
                duration_ms = round((time.perf_counter() - started) * 1000, 1)
                merged = dict(state.get("_durations") or {})
                merged[name] = duration_ms
                sink.emit(
                    RunEvent(
                        type="node_completed",
                        invoice_id=state["invoice"]["invoice_id"],
                        node=name,
                        duration_ms=duration_ms,
                    )
                )
                return {**update, "_durations": merged}

            run.__name__ = f"node_{name}"
            return run

        return wrap

    @node("ocr", "OCR — document text extraction")
    def _ocr(state: AgentState) -> dict[str, Any]:
        sub = InvoiceInput.model_validate(state["invoice"])
        if object_store is not None:
            # Documents that reference a blob instead of carrying a
            # text layer: pull the bytes through the object-store
            # port and decode them as the document text (text-bearing
            # formats); the OCR engine then reads them like any other
            # document. A missing blob degrades to an empty layer —
            # the vision node will call the document unreadable.
            changed = False
            for document in sub.documents:
                if document.object_key and not document.text_layer:
                    try:
                        data = object_store.get(document.object_key)
                        document.ocr_text = data.decode("utf-8", errors="replace")
                        changed = True
                    except Exception:
                        pass
            if changed:
                return {
                    "invoice": sub.model_dump(),
                    "ocr_results": [r.model_dump() for r in run_ocr(sub, ocr_engine)],
                }
        return {"ocr_results": [r.model_dump() for r in run_ocr(sub, ocr_engine)]}

    @node("vision", "Vision verify — document type + legibility")
    def _vision(state: AgentState) -> dict[str, Any]:
        sub = InvoiceInput.model_validate(state["invoice"])
        ocr_results = [OcrResult.model_validate(r) for r in state.get("ocr_results", [])]
        vision = run_vision(sub, ocr_results, vision_engine)
        return {"vision": vision.model_dump()}

    @node("screen", "Injection + fraud screening of supplier text")
    def _screen(state: AgentState) -> dict[str, Any]:
        sub = InvoiceInput.model_validate(state["invoice"])
        fraud = screen_fraud(sub)
        update: dict[str, Any] = {
            "injection_flags": screen_invoice(sub),
            "fraud_flags": fraud,
            "bank_change_blocked": bool(fraud),
        }
        if supplier_master is not None:
            # The master-data check (AP-BANK-01's out-of-band record):
            # document-stated bank details vs the approved master.
            # A mismatch is the fraud signal; a match downgrades a
            # pattern hit whose details are the ones already on file.
            record = supplier_master.get_supplier(sub.supplier_name)
            check = compare_bank_details(
                sub,
                record,
                source=getattr(supplier_master, "name", "") or "",
            )
            update["bank_master_check"] = check.model_dump()
            if check.status == "checked_mismatch":
                fraud = [*fraud, mismatch_flag(check)]
                update["fraud_flags"] = fraud
                update["bank_change_blocked"] = True
            elif check.status == "checked_match" and fraud:
                update["fraud_flags"] = []
                update["bank_change_blocked"] = False
        return update

    @node("extract", "Field extraction + cross-check")
    def _extract(state: AgentState) -> dict[str, Any]:
        sub = InvoiceInput.model_validate(state["invoice"])
        extractions = extract_documents(sub, backend)
        return {"extractions": [e.model_dump() for e in extractions]}

    @node("match", "Three-way match (invoice vs PO vs goods receipt)")
    def _match(state: AgentState) -> dict[str, Any]:
        sub = InvoiceInput.model_validate(state["invoice"])
        extractions = [DocumentExtraction.model_validate(e) for e in state["extractions"]]
        facts = compute_match(
            sub, extractions, duplicate_of=state.get("duplicate_of", "")
        )
        return {"facts": facts.model_dump()}

    @node("classify", "Triage: rules + LLM cross-check")
    def _classify(state: AgentState) -> dict[str, Any]:
        sub = InvoiceInput.model_validate(state["invoice"])
        facts = MatchFacts.model_validate(state["facts"])
        vision = (
            VisionResult.model_validate(state["vision"]) if state.get("vision") else None
        )
        ocr_results = [OcrResult.model_validate(r) for r in state.get("ocr_results", [])]
        rule = classify_invoice(
            sub,
            facts,
            vision=vision,
            ocr_results=ocr_results,
            fraud_flags=state.get("fraud_flags", []),
        )
        update: dict[str, Any] = {"rule_triage": rule.model_dump()}
        classify_fn = getattr(backend, "classify_with_llm", None)
        if classify_fn is None:
            update["triage"] = rule.model_dump()
            update["cross_check"] = None
            return update
        try:
            llm = classify_fn(_classify_context(state, sub, facts))
        except Exception as exc:
            # A provider failure in the cross-check degrades to
            # rules-only — the exception travels in state, not up.
            final, cross = resolve_triage(rule, None)
            cross.note = f"LLM cross-check failed ({exc}) — rules only."
            update["triage"] = final.model_dump()
            update["cross_check"] = cross.model_dump()
            return update
        final, cross = resolve_triage(rule, llm, getattr(backend, "name", ""))
        update["triage"] = final.model_dump()
        update["cross_check"] = cross.model_dump()
        return update

    @node("retrieve", "AP policy retrieval")
    def _retrieve(state: AgentState) -> dict[str, Any]:
        sub = InvoiceInput.model_validate(state["invoice"])
        triage = Triage.model_validate(state["triage"])
        facts = MatchFacts.model_validate(state["facts"])
        query = " ".join(
            part
            for part in [
                "accounts payable policy",
                triage.outcome.replace("_", " "),
                facts.match_status.replace("_", " "),
                "three way match",
                "tolerance" if facts.match_status != "full_match" else "approval threshold",
                "duplicate invoice" if facts.duplicate_suspected else "",
                "bank detail change fraud" if state.get("bank_change_blocked") else "",
                "goods receipt" if not facts.has_receipt else "",
                "purchase order" if not facts.has_po else "",
                "tax reconciliation" if facts.tax_variance else "",
                "currency" if not facts.currency_consistent else "",
                "unreadable document OCR confidence legible resubmission scan" if triage.outcome == "unreadable" else "",
                sub.supplier_name,
            ]
            if part
        )
        policies = retriever.retrieve(query, top_k=4)
        # The policies the triage rationale itself cites govern the case
        # by construction (the rule that fired names them): carry them
        # into the retrieved set when the search missed them, so the
        # drafts may cite what the rationale cites — a not-an-invoice
        # case is governed by AP-DOC-02 whether or not the keyword
        # search ranked it.
        have = {g.policy_id for g in policies}
        corpus = {p["id"]: p for p in _POLICY_CORPUS}
        for policy_id in cited_policies(triage.rationale):
            if policy_id not in have and policy_id in corpus:
                entry = corpus[policy_id]
                policies.append(
                    RetrievedPolicy(
                        policy_id=policy_id,
                        title=entry["title"],
                        text=entry["text"],
                        score=0.0,
                        sources=["triage"],
                    )
                )
                have.add(policy_id)
        return {
            "policies": [g.model_dump() for g in policies],
            "retrieval_info": getattr(retriever, "vector_store", "") or "",
        }

    @node("discrepancies", "Extraction cross-check discrepancies")
    def _discrepancies(state: AgentState) -> dict[str, Any]:
        extractions = [DocumentExtraction.model_validate(e) for e in state["extractions"]]
        return {"discrepancies": extraction_discrepancies(extractions)}

    @node("history", "Supplier memory + approver feedback")
    def _history(state: AgentState) -> dict[str, Any]:
        return {"history_evidence": memory_evidence_lines(state.get("history"))}

    @node("diagnose", "Exception diagnosis (bounded tool loop in provider mode)")
    def _diagnose(state: AgentState) -> dict[str, Any]:
        sub = InvoiceInput.model_validate(state["invoice"])
        facts = MatchFacts.model_validate(state["facts"])
        toolbox = DiagnosisToolBox(
            invoice=sub,
            triage=state["triage"],
            facts=facts,
            discrepancies=state.get("discrepancies", []),
            retriever=retriever,
            priors=state.get("priors") or [],
        )
        diagnosis = build_diagnosis(
            invoice=sub,
            triage=state["triage"],
            facts=_facts_view(state, sub),
            extractions=[DocumentExtraction.model_validate(e) for e in state["extractions"]],
            policies=state.get("policies", []),
            backend=backend,
            history=state.get("history"),
            toolbox=toolbox,
            discrepancies=state.get("discrepancies", []),
            history_lines=state.get("history_evidence", []),
            document_lines=_document_lines(state),
            fraud_flags=state.get("fraud_flags", []),
        )
        return {"diagnosis": diagnosis.model_dump()}

    @node("options", "Next-step options, scored in code")
    def _options(state: AgentState) -> dict[str, Any]:
        sub = InvoiceInput.model_validate(state["invoice"])
        # Early-payment discount: priced in code before scoring, so
        # the approve option's score (and its wording) can carry the
        # quantified saving. The evaluation rides on state to the
        # drafts and the final result.
        discount = evaluate_discount(sub)
        facts = _facts_view(state, sub)
        if discount is not None:
            facts = {
                **facts,
                "discount": {
                    **discount.model_dump(),
                    "line": format_discount_line(discount, sub.currency),
                },
            }
        options, notes = build_options(
            invoice=sub,
            triage=state["triage"],
            facts=facts,
            diagnosis_summary=state["diagnosis"]["summary"],
            backend=backend,
        )
        update: dict[str, Any] = {
            "options": [o.model_dump() for o in options],
            "discount": discount.model_dump() if discount is not None else None,
        }
        if notes:
            update["options_notes"] = notes
        return update

    @node("draft", "Draft supplier letter + AP packet")
    def _draft(state: AgentState) -> dict[str, Any]:
        sub = InvoiceInput.model_validate(state["invoice"])
        subject, letter, packet = backend.draft_documents(_draft_context(state, sub))
        # Composition, not memory: the packet carries the policies
        # the diagnosis rests on (the retrieved set) whether or not
        # the drafter remembered to cite them.
        packet = _carry_policy_citations(
            packet, [g["policy_id"] for g in state.get("policies", [])]
        )
        return {
            "drafts": Drafts(
                subject=subject, supplier_letter=letter, ap_packet=packet
            ).model_dump()
        }

    @node("verify", "Self-verification of the drafts")
    def _verify(state: AgentState) -> dict[str, Any]:
        sub = InvoiceInput.model_validate(state["invoice"])
        drafts = Drafts.model_validate(state["drafts"])
        retrieved_ids = [g["policy_id"] for g in state.get("policies", [])]
        result = verify_drafts(
            invoice=sub,
            triage=state["triage"],
            facts=_facts_view(state, sub),
            drafts=drafts,
            retrieved_ids=retrieved_ids,
            backend=backend,
            policies=list(state.get("policies", [])),
        )
        return {"verification": result.model_dump()}

    @node("review", "Independent review of the drafts")
    def _review(state: AgentState) -> dict[str, Any]:
        sub = InvoiceInput.model_validate(state["invoice"])
        drafts = Drafts.model_validate(state["drafts"])
        retrieved_ids = [g["policy_id"] for g in state.get("policies", [])]
        result = review_drafts(
            invoice=sub,
            triage=state["triage"],
            facts=_facts_view(state, sub),
            diagnosis=Diagnosis.model_validate(state["diagnosis"]),
            options=[RecoveryOption.model_validate(o) for o in state.get("options", [])],
            drafts=drafts,
            retrieved_ids=retrieved_ids,
            backend=backend,
            policies=list(state.get("policies", [])),
        )
        return {
            "review": result.model_dump(),
            "reviewer_blocked": bool(result.enabled and result.verdict == "block"),
        }

    @node("validate", "Guardrail checks (+ bounded repair)")
    def _validate(state: AgentState) -> dict[str, Any]:
        sub = InvoiceInput.model_validate(state["invoice"])
        citations = [g["policy_id"] for g in state.get("policies", [])]
        drafts = Drafts.model_validate(state["drafts"])
        allowed = _facts_view(state, sub)["allowed_amounts"]
        # The computed outcome the drafts' claims are traced
        # against (outcome_claim_must_trace) — from this run's own
        # triage + facts, never from the drafts under judgement.
        outcome = str((state.get("triage") or {}).get("outcome") or "")
        match_status = str((state.get("facts") or {}).get("match_status") or "")
        validation = validate_drafts(
            sub,
            drafts,
            citations,
            allowed_amounts=allowed,
            outcome=outcome,
            match_status=match_status,
        )
        update: dict[str, Any] = {
            "validation": validation.model_dump(),
            "repair_attempted": False,
            "repaired": False,
            "repair_attempts": 0,
            "original_validation": None,
        }
        if validation.passed or not _guardrail_repair_enabled():
            return update
        max_attempts = _guardrail_repair_max_attempts()
        original = validation
        attempts = 0
        while attempts < max_attempts and not validation.passed:
            attempts += 1
            feedback = _failed_checks_text(validation)
            try:
                subject, letter, packet = backend.draft_documents(
                    _draft_context(state, sub, repair_feedback=feedback)
                )
            except Exception:
                break  # a failed redraft keeps the original drafts + failures
            candidate = Drafts(
                subject=subject,
                supplier_letter=letter,
                ap_packet=_carry_policy_citations(packet, citations),
            )
            candidate_validation = validate_drafts(
                sub,
                candidate,
                citations,
                allowed_amounts=allowed,
                outcome=outcome,
                match_status=match_status,
            )
            if candidate_validation.passed or _failed_count(candidate_validation) < _failed_count(validation):
                drafts, validation = candidate, candidate_validation
        update.update(
            {
                "drafts": drafts.model_dump(),
                "validation": validation.model_dump(),
                "repair_attempted": attempts > 0,
                "repaired": validation.passed,
                "repair_attempts": attempts,
                "original_validation": original.model_dump(),
            }
        )
        return update

    @node("gate", "Autonomy recommendation + information request")
    def _gate(state: AgentState) -> dict[str, Any]:
        sub = InvoiceInput.model_validate(state["invoice"])
        triage = Triage.model_validate(state["triage"])
        facts = MatchFacts.model_validate(state["facts"])
        validation = Validation.model_validate(state["validation"])
        cross = (
            CrossCheck.model_validate(state["cross_check"])
            if state.get("cross_check")
            else None
        )
        review = (
            ReviewResult.model_validate(state["review"]) if state.get("review") else None
        )
        autonomy = recommend_autonomy(
            triage=triage,
            facts=facts,
            validation=validation,
            repair_attempted=bool(state.get("repair_attempted")),
            cross_check=cross,
            review=review,
            reviewer_blocked=bool(state.get("reviewer_blocked")),
            fraud_flags=state.get("fraud_flags", []),
            total_amount=sub.total_amount,
        )
        info = build_information_request(
            invoice=sub,
            triage_outcome=triage.outcome,
            facts=facts,
            extractions=[DocumentExtraction.model_validate(e) for e in state["extractions"]],
            backend=backend,
            allowed_amounts=_facts_view(state, sub)["allowed_amounts"],
        )
        return {
            "autonomy": autonomy.model_dump(),
            "needs_information": bool(info and info.needed),
            "information_request": info.model_dump() if info else None,
            "gate_status": "awaiting_approver",
        }

    def _approval_gate(state: AgentState) -> dict[str, Any]:
        """The human gate: pause the thread until a decision resumes it.

        With a checkpointer the run pauses here (``interrupt``) and the
        service's approve/reject resumes the same thread with the
        decision payload. Without one this node is a pass-through and
        the decision lives only in the store — same outcome, less
        ceremony. Either way no node after this one exists: END.
        """
        if checkpointer is None:
            return {"gate_status": state.get("gate_status", "awaiting_approver")}
        triage = state.get("triage") or {}
        validation = state.get("validation") or {}
        decision = interrupt(
            {
                "invoice_id": state["invoice"]["invoice_id"],
                "status": "awaiting_approver",
                "outcome": triage.get("outcome"),
                "severity": triage.get("severity"),
                "validation_passed": validation.get("passed"),
                "external_action_taken": False,
            }
        )
        if isinstance(decision, dict) and decision.get("decision") in (
            "approved",
            "rejected",
        ):
            return {"gate_status": decision["decision"]}
        return {"gate_status": "awaiting_approver"}

    builder = StateGraph(AgentState)
    for name, fn in [
        ("ocr", _ocr),
        ("vision", _vision),
        ("screen", _screen),
        ("extract", _extract),
        ("match", _match),
        ("classify", _classify),
        ("retrieve", _retrieve),
        ("discrepancies", _discrepancies),
        ("history", _history),
        ("diagnose", _diagnose),
        ("options", _options),
        ("draft", _draft),
        ("verify", _verify),
        ("review", _review),
        ("validate", _validate),
        ("gate", _gate),
        ("approval_gate", _approval_gate),
    ]:
        builder.add_node(name, fn)
    builder.set_entry_point("ocr")
    builder.add_edge("ocr", "vision")
    builder.add_edge("vision", "screen")
    builder.add_edge("screen", "extract")
    builder.add_edge("extract", "match")
    builder.add_edge("match", "classify")
    # Evidence fan-out: three independent branches off classify,
    # joined by diagnose.
    builder.add_edge("classify", "retrieve")
    builder.add_edge("classify", "discrepancies")
    builder.add_edge("classify", "history")
    builder.add_edge("retrieve", "diagnose")
    builder.add_edge("discrepancies", "diagnose")
    builder.add_edge("history", "diagnose")
    builder.add_edge("diagnose", "options")
    builder.add_edge("options", "draft")
    # Guardrails (and their bounded repair) run before the critics, so
    # the self-check and the independent reviewer judge the drafts the
    # approver will actually read — a repaired draft replaces the first
    # one, and a verdict about the replaced draft would be stale.
    builder.add_edge("draft", "validate")
    builder.add_edge("validate", "verify")
    builder.add_edge("verify", "review")
    builder.add_edge("review", "gate")
    builder.add_edge("gate", "approval_gate")
    builder.add_edge("approval_gate", END)
    return builder.compile(checkpointer=checkpointer)


def _failed_count(validation: Validation) -> int:
    return sum(1 for check in validation.checks if not check.passed)


def _durations(state: AgentState) -> dict[str, float]:
    return dict(state.get("_durations") or {})


def _build_trace(state: AgentState) -> list[TraceStep]:
    """Assemble the human-readable step trace from the final state."""
    d = _durations(state)
    sub = InvoiceInput.model_validate(state["invoice"])
    triage = Triage.model_validate(state["triage"])
    rule = Triage.model_validate(state["rule_triage"]) if state.get("rule_triage") else triage
    facts = MatchFacts.model_validate(state["facts"])
    diagnosis = Diagnosis.model_validate(state["diagnosis"])
    validation = Validation.model_validate(state["validation"])
    policies = [RetrievedPolicy.model_validate(g) for g in state.get("policies", [])]
    options = [RecoveryOption.model_validate(o) for o in state.get("options", [])]
    verification = (
        VerifyResult.model_validate(state["verification"])
        if state.get("verification")
        else None
    )
    review = ReviewResult.model_validate(state["review"]) if state.get("review") else None

    def step(name: str, title: str, details: list[str]) -> TraceStep:
        return TraceStep(name=name, title=title, details=details, duration_ms=d.get(name, 0.0))

    ocr_details = [
        f"{r.get('document_id') or r.get('doc_type')}: engine {r.get('engine')}, "
        f"mean confidence {r.get('mean_confidence')}, {r.get('char_count')} chars"
        + (" — LOW CONFIDENCE" if r.get("low_confidence") else "")
        for r in state.get("ocr_results", [])
    ] or ["no documents on the invoice"]
    vision = state.get("vision") or {}
    vision_details = [
        f"detected document type: {vision.get('document_type_detected')} · "
        f"is_invoice: {vision.get('is_invoice')} · legible: {vision.get('legible')} · "
        f"quality: {vision.get('quality')} (source: {vision.get('source')})"
    ] if vision else ["vision check not run"]
    screen_details = list(state.get("injection_flags") or [])
    screen_details += [f"FRAUD: {f}" for f in state.get("fraud_flags") or []]
    master_check = state.get("bank_master_check") or {}
    if master_check.get("status"):
        master_line = f"supplier master check: {master_check['status']}"
        if master_check.get("matched"):
            master_line += f" — on file: {', '.join(master_check['matched'])}"
        if master_check.get("mismatched"):
            master_line += f" — NOT on file: {', '.join(master_check['mismatched'])}"
        screen_details.append(master_line)
    if not screen_details:
        screen_details = ["no injection or fraud patterns found in supplier text"]
    if state.get("bank_change_blocked"):
        screen_details.append(
            "bank-detail change request BLOCKED — fraud hold applies [AP-BANK-01]"
        )

    extraction_details = [
        f"{e['document_id'] or e['doc_type']}: {len(e['fields'])} field(s) "
        f"({e['source']})"
        + (
            " — "
            + ", ".join(
                f"{f['field']}={f['status']}" for f in e["fields"] if f["status"] != "match"
            )
            if any(f["status"] != "match" for f in e["fields"])
            else ""
        )
        for e in state.get("extractions", [])
    ] or ["no documents on the invoice"]
    if state.get("discrepancies"):
        extraction_details.append(
            f"{len(state['discrepancies'])} cross-check discrepanc"
            f"{'y' if len(state['discrepancies']) == 1 else 'ies'} recorded"
        )

    match_details = [format_match_summary(facts)]
    match_details += [
        f"L{m.line_no} {m.description}: {m.status} — {m.detail}" for m in facts.line_matches
    ]
    if facts.duplicate_suspected:
        match_details.append(f"duplicate suspected of {facts.duplicate_of} [AP-DUP-01]")

    classify_details = [
        f"rules: {rule.outcome} / {rule.severity} (confidence {rule.confidence})"
    ]
    cross = state.get("cross_check")
    if cross:
        classify_details.append(f"cross-check: {cross['resolution']} — {cross['note']}")
        if cross.get("llm_outcome"):
            classify_details.append(
                f"LLM: {cross['llm_outcome']} / {cross['llm_severity']} "
                f"(confidence {cross['llm_confidence']})"
            )
    classify_details.append(
        f"final: {triage.outcome} / {triage.severity} (confidence {triage.confidence})"
    )
    classify_details.extend(f"signal: {s}" for s in triage.signals)

    steps = [
        step("ocr", "OCR — document text", ocr_details),
        step("vision", "Vision verify — type + legibility", vision_details),
        step("screen", "Injection + fraud screening", screen_details),
        step("extract", "Field extraction + cross-check", extraction_details),
        step("match", "Three-way match", match_details),
        step("classify", "Triage (rules + LLM cross-check)", classify_details),
        step(
            "retrieve",
            "AP policy retrieval",
            [
                f"[{g.policy_id}] {g.title} (score {g.score}, via "
                f"{'+'.join(g.sources) or 'keyword'})"
                for g in policies
            ]
            or ["no policies retrieved"],
        ),
        step(
            "diagnose",
            "Exception diagnosis",
            [
                f"{diagnosis.summary} (source: {diagnosis.source})",
                *([f"note: {diagnosis.note}"] if diagnosis.note else []),
                *[f"tool: {tc.get('name')} — {tc.get('summary')}" for tc in diagnosis.tool_calls],
                *diagnosis.evidence,
            ],
        ),
        step(
            "options",
            "Next-step options (scored in code)",
            [
                f"[{o.kind}] {o.title} — score {o.score}"
                + (" ← recommended" if o.recommended else "")
                for o in options
            ]
            + ([f"note: {state['options_notes']}"] if state.get("options_notes") else []),
        ),
        step(
            "draft",
            "Drafts composed",
            [
                "supplier query letter + AP packet drafted — nothing sent, "
                "nothing approved, nothing paid"
            ],
        ),
    ]
    # Execution order: guardrails (+ repair) run before the critics.
    validate_details = [
        f"[{'PASS' if c.passed else 'FAIL'}] {c.name} — {c.detail}"
        for c in validation.checks
    ]
    if state.get("repair_attempted"):
        validate_details.append(
            f"repair: {state.get('repair_attempts')} attempt(s) — "
            f"{'repaired' if state.get('repaired') else 'still failing'}"
        )
    steps.append(step("validate", "Guardrail checks", validate_details))
    if verification is not None:
        steps.append(
            step(
                "verify",
                "Self-verification",
                [
                    f"grounded: {verification.grounded} (source: {verification.source})",
                    *[f"issue: {i}" for i in verification.issues],
                    *([f"note: {verification.note}"] if verification.note else []),
                ],
            )
        )
    if review is not None:
        steps.append(
            step(
                "review",
                "Independent review",
                [
                    f"verdict: {review.verdict} (source: {review.source}, "
                    f"enabled: {review.enabled})",
                    *[f"finding: {f}" for f in review.findings],
                    *(
                        ["⛔ reviewer blocked — flagged for the approver"]
                        if state.get("reviewer_blocked")
                        else []
                    ),
                ],
            )
        )
    autonomy = state.get("autonomy") or {}
    gate_details = [
        f"autonomy recommendation: "
        f"{'eligible for auto-triage' if autonomy.get('eligible_for_auto_triage') else 'human decision required'}"
        " (recommendation only — never acted on)",
        *autonomy.get("reasons", []),
    ]
    if state.get("needs_information"):
        gate_details.append(
            "information request composed (not sent) — missing: "
            + "; ".join((state.get("information_request") or {}).get("missing_items", []))
        )
    steps.append(step("gate", "Approver gate", gate_details))
    steps.append(
        step(
            "approval_gate",
            "Human decision",
            [
                f"status: {state.get('gate_status', 'awaiting_approver')} · "
                "external action taken: False (nothing is approved, paid, sent, "
                "or filed by the agent)"
            ],
        )
    )
    return steps


def _run_telemetry(backend, started: float, budget_status: dict | None) -> dict:
    """Assemble the run's telemetry block from the backend's accounting.

    The mock backend reports zero tokens for its render calls — those
    are template renders, not model usage — so mock runs record token
    counts as None ("no model ran") rather than dressing them up.
    """
    usage = backend.usage_totals()
    is_mock = getattr(backend, "name", "") == "mock"
    input_tokens = None if is_mock else int(usage.get("input_tokens", 0))
    output_tokens = None if is_mock else int(usage.get("output_tokens", 0))
    model = getattr(backend, "_model", "") or ""
    return {
        "backend": getattr(backend, "name", "unknown"),
        "model": model,
        "model_calls": int(usage.get("calls", 0)),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": estimate_cost_usd(model, input_tokens, output_tokens),
        "latency_seconds": round(time.perf_counter() - started, 3),
        "budget": budget_status,
        # The budgets this run actually ran with, per call site,
        # resolved at run time (env overrides included) — so a
        # starved-reply post-mortem reads them off the record.
        "completion_budgets": run_completion_budgets(),
    }


def run_invoice(
    invoice: InvoiceInput,
    backend=None,
    retriever=None,
    history: dict | None = None,
    priors: list[dict] | None = None,
    event_sink=None,
    checkpointer: BaseCheckpointSaver | None = None,
    thread_id: str | None = None,
    ocr_engine=None,
    vision_engine=None,
    object_store=None,
    supplier_master=None,
    duplicate_of: str = "",
) -> AgentResult:
    """Run one invoice through the pipeline to the approver gate.

    ``backend``/``retriever`` default to the environment-configured
    ones; ``history``/``priors`` carry the store's memory for this
    supplier (the service layer supplies them); ``event_sink``
    receives structured run events; ``checkpointer`` + ``thread_id``
    pause the run at the gate for a later resume; ``ocr_engine`` /
    ``vision_engine`` / ``object_store`` are the document ports.

    The duplicate verdict (AP-DUP-01) is computed HERE, from the
    priors themselves — never taken on a caller's word. When
    ``priors`` are supplied, ``duplicate_of`` is recomputed from
    them and any caller-supplied verdict is discarded; the
    parameter is honoured only when no priors ride along (a caller
    holding a verdict but not the evidence). Before this, the
    screen existed only because the service computed the verdict
    and passed it in: every direct ``run_invoice`` caller ran with
    the screen silently off, and a live judged slice scored the
    INV-1004 resend as a clean match because nobody had screened
    it. A deterministic control that depends on the caller
    remembering to invoke it is not a control.
    """
    from .model_backends import MockModelBackend, get_backend

    started = time.perf_counter()
    sink = event_sink or NullSink()
    raw_backend = backend or get_backend()
    if hasattr(raw_backend, "reset_usage"):
        raw_backend.reset_usage()
    budget = _BudgetGuard(raw_backend, MockModelBackend(), _token_budget())
    wrapped = _wrap_backend(budget)
    retriever = retriever or get_retriever()
    if priors:
        # The screen, computed from the evidence (see the
        # docstring): with priors in hand the verdict is derived,
        # never inherited.
        duplicate_of = find_duplicate(priors, invoice)
    graph = _build_graph(
        wrapped,
        retriever,
        sink,
        checkpointer=checkpointer,
        ocr_engine=ocr_engine,
        vision_engine=vision_engine,
        object_store=object_store,
        supplier_master=supplier_master,
    )
    initial: AgentState = {
        "invoice": invoice.model_dump(),
        "history": history,
        "priors": priors or [],
        "duplicate_of": duplicate_of or "",
    }
    sink.emit(
        RunEvent(
            type="run_started",
            invoice_id=invoice.invoice_id,
            detail={"supplier_name": invoice.supplier_name},
        )
    )
    config = (
        {"configurable": {"thread_id": thread_id or invoice.invoice_id}}
        if checkpointer is not None
        else None
    )
    try:
        if config is not None:
            state = graph.invoke(initial, config)
            interrupts = state.get("__interrupt__") or []
            if interrupts:
                # Paused at the gate: read the working state back from
                # the checkpoint — it is the state the decision will
                # resume, and the store records it as awaiting.
                snapshot = graph.get_state(config)
                state = dict(snapshot.values)
        else:
            state = graph.invoke(initial)
    except ProviderError:
        raise
    except Exception as exc:
        sink.emit(
            RunEvent(
                type="run_failed",
                invoice_id=invoice.invoice_id,
                detail={"error": f"{type(exc).__name__}: {exc}"},
            )
        )
        raise
    result = AgentResult(
        invoice_id=invoice.invoice_id,
        status="awaiting_approver",
        supplier_name=invoice.supplier_name,
        invoice_number=invoice.invoice_number,
        ocr_results=[OcrResult.model_validate(r) for r in state.get("ocr_results", [])],
        vision=(
            VisionResult.model_validate(state["vision"]) if state.get("vision") else None
        ),
        extractions=[DocumentExtraction.model_validate(e) for e in state.get("extractions", [])],
        facts=MatchFacts.model_validate(state["facts"]),
        injection_flags=state.get("injection_flags", []),
        fraud_flags=state.get("fraud_flags", []),
        bank_change_blocked=bool(state.get("bank_change_blocked")),
        bank_master_check=(
            BankMasterCheck.model_validate(state["bank_master_check"])
            if state.get("bank_master_check")
            else None
        ),
        triage=Triage.model_validate(state["triage"]),
        cross_check=(
            CrossCheck.model_validate(state["cross_check"])
            if state.get("cross_check")
            else None
        ),
        policies=[RetrievedPolicy.model_validate(g) for g in state.get("policies", [])],
        vector_store=state.get("retrieval_info", ""),
        diagnosis=Diagnosis.model_validate(state["diagnosis"]),
        discount=(
            DiscountEvaluation.model_validate(state["discount"])
            if state.get("discount")
            else None
        ),
        options=[RecoveryOption.model_validate(o) for o in state.get("options", [])],
        drafts=Drafts.model_validate(state["drafts"]) if state.get("drafts") else None,
        verification=(
            VerifyResult.model_validate(state["verification"])
            if state.get("verification")
            else None
        ),
        review=ReviewResult.model_validate(state["review"]) if state.get("review") else None,
        reviewer_blocked=bool(state.get("reviewer_blocked")),
        validation=(
            Validation.model_validate(state["validation"])
            if state.get("validation")
            else None
        ),
        repair_attempted=bool(state.get("repair_attempted")),
        repaired=bool(state.get("repaired")),
        repair_attempts=int(state.get("repair_attempts", 0)),
        original_validation=(
            Validation.model_validate(state["original_validation"])
            if state.get("original_validation")
            else None
        ),
        information_request=None,
        needs_information=bool(state.get("needs_information")),
        telemetry=None,
        trace=_build_trace(state),
    )
    if state.get("information_request"):
        from .schemas import InformationRequest

        result.information_request = InformationRequest.model_validate(
            state["information_request"]
        )
    from .schemas import AutonomyRecommendation, RunTelemetry

    if state.get("autonomy"):
        result.autonomy = AutonomyRecommendation.model_validate(state["autonomy"])
    result.telemetry = RunTelemetry.model_validate(
        _run_telemetry(raw_backend, started, budget.status())
    )
    sink.emit(
        RunEvent(
            type="run_completed",
            invoice_id=invoice.invoice_id,
            detail={"status": result.status},
        )
    )
    return result


def resume_approval(
    invoice_id: str,
    decision: dict,
    checkpointer: BaseCheckpointSaver,
    backend=None,
    retriever=None,
) -> bool:
    """Resume a gate-paused thread with the approver's decision.

    Returns True when a paused thread was found and completed, False
    when there is nothing to resume (checkpoints off during analysis,
    thread already finished). The decision of record lives in the
    store; this only closes out the graph thread.
    """
    from .model_backends import get_backend

    sink = NullSink()
    graph = _build_graph(
        _wrap_backend(backend or get_backend()),
        retriever or get_retriever(),
        sink,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": invoice_id}}
    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.next:
        return False
    from langgraph.types import Command

    graph.invoke(Command(resume=decision), config)
    return True
