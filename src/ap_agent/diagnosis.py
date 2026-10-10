"""Diagnose node logic: an exception summary over the computed evidence.

The diagnosis reasons over computed evidence only — the three-way
match (line by line), the extraction cross-check discrepancies, the
OCR/vision document checks, the triage's own signals, the retrieved
AP policies, and (when the store has history) the supplier memory and
approver feedback lines. The evidence list is assembled
deterministically in BOTH modes, so the citations an approver sees are
always the real inputs. In provider mode the LLM composes the prose
from those facts — agentically, through a bounded tool loop when the
graph supplies a toolbox (``tools_agent.py``); in the default mode a
deterministic template composes it from the same evidence structure.
An unusable LLM reply falls back to the template, with the reason
recorded in ``note``.
"""

from __future__ import annotations

from .extractor import extraction_discrepancies
from .matching import format_match_summary
from .model_backends import DraftContext
from .schemas import Diagnosis, DocumentExtraction, InvoiceInput
from .screening import fraud_flag_label, sanitized_notes
from .store import format_outcome_counts
from .tools_agent import TOOL_SPECS, DiagnosisToolBox, diagnosis_max_tool_calls


def memory_evidence_lines(history: dict | None) -> list[str]:
    """History as evidence lines, in the diagnosis' citation style.

    Only real history produces lines — absence is not evidence.
    Feedback lines carry the human decisions (with their reasons) on
    earlier cases for this supplier, so the pattern of oversight
    itself becomes evidence.
    """
    if not history:
        return []
    lines: list[str] = []
    if history.get("supplier_count"):
        recent = ", ".join(history.get("supplier_recent_outcomes", [])) or "none recorded"
        breakdown = format_outcome_counts(history.get("supplier_outcome_counts", {}))
        lines.append(
            f"supplier history: {history['supplier_count']} prior invoice(s) from "
            f"{history.get('supplier', '')} in the stored history "
            f"(outcomes: {breakdown}; most recent: {recent})"
        )
    for index, feedback in enumerate(history.get("feedback", [])):
        when = "last" if index == 0 else "an earlier"
        lines.append(
            f"approver feedback: {when} decision for this supplier was "
            f"{feedback.get('decision', '')} — reason: '{feedback.get('reason', '')}'"
        )
    return lines


def build_evidence(
    *,
    triage: dict,
    facts: dict,
    extractions: list[DocumentExtraction],
    policies: list[dict],
    history: dict | None = None,
    discrepancies: list[str] | None = None,
    history_lines: list[str] | None = None,
    document_lines: list[str] | None = None,
) -> list[str]:
    """The deterministic evidence list every diagnosis cites.

    ``discrepancies`` / ``history_lines`` are the precomputed
    extraction cross-check and memory/feedback evidence (the graph's
    evidence fan-out computes them in parallel branches); when they
    are None they are computed here — the lines are identical either
    way, only *when* they are computed differs.
    """
    evidence: list[str] = []
    evidence.append(
        f"computed three-way match: {facts.get('match_summary')} "
        f"(status {facts.get('match_status')}, score {facts.get('match_score')})"
    )
    for line in facts.get("line_detail") or []:
        evidence.append(f"match line: {line}")
    for line in document_lines or []:
        evidence.append(f"document check: {line}")
    if discrepancies is None:
        discrepancies = extraction_discrepancies(extractions)
    for line in discrepancies:
        evidence.append(f"extraction cross-check: {line}")
    if triage.get("signals"):
        evidence.append("triage signals: " + "; ".join(triage["signals"]))
    for g in policies:
        evidence.append(f"policy {g['policy_id']}: {g['title']}")
    evidence.extend(
        history_lines if history_lines is not None else memory_evidence_lines(history)
    )
    return evidence


def _template_risk_factors(
    *, triage: dict, facts: dict, discrepancies: list[str], fraud_flags: list[str]
) -> list[str]:
    factors: list[str] = []
    for flag in fraud_flags:
        factors.append(f"FRAUD SCREEN: {fraud_flag_label(flag)}")
    factors.extend(f"match discrepancy: {d}" for d in facts.get("discrepancies") or [])
    factors.extend(f"extraction discrepancy: {d}" for d in discrepancies)
    for signal in triage.get("signals") or []:
        if signal not in factors:
            factors.append(f"triage signal: {signal}")
    return factors


def build_diagnosis(
    *,
    invoice: InvoiceInput,
    triage: dict,
    facts: dict,
    extractions: list[DocumentExtraction],
    policies: list[dict],
    backend,
    history: dict | None = None,
    toolbox: DiagnosisToolBox | None = None,
    discrepancies: list[str] | None = None,
    history_lines: list[str] | None = None,
    document_lines: list[str] | None = None,
    fraud_flags: list[str] | None = None,
) -> Diagnosis:
    """Compose the diagnosis: LLM prose in provider mode, template otherwise.

    In provider mode with a ``toolbox`` (the graph passes one), the LLM
    diagnosis is agentic: a bounded tool loop (``tools_agent.py``) lets
    the model pull policy search results, the computed match facts,
    supplier history, and the purchase order before composing. Without
    a toolbox the single-call diagnosis runs. Either path degrades to
    the template, with the reason recorded in ``note``.
    """
    # The diagnosis feeds the supplier's text to the model (provider
    # path); it sees the screened version — sentences the injection
    # and fraud screens flagged never reach the prompt.
    supplier_text = sanitized_notes(invoice)
    fraud_flags = fraud_flags or []
    if discrepancies is None:
        discrepancies = extraction_discrepancies(extractions)
    evidence = build_evidence(
        triage=triage,
        facts=facts,
        extractions=extractions,
        policies=policies,
        history=history,
        discrepancies=discrepancies,
        history_lines=history_lines,
        document_lines=document_lines,
    )
    risk_factors = _template_risk_factors(
        triage=triage, facts=facts, discrepancies=discrepancies, fraud_flags=fraud_flags
    )
    template = Diagnosis(
        summary=(
            f"{triage['outcome']} (severity {triage['severity']}) for "
            f"{invoice.supplier_name} invoice {invoice.invoice_number or invoice.invoice_id} — "
            f"{facts.get('match_summary')}; "
            f"{len(policies)} governing policy(s) retrieved."
            + (
                " A bank-detail change request was screened in the "
                "documents — fraud hold applies [AP-BANK-01]."
                if fraud_flags
                else ""
            )
        ),
        risk_factors=risk_factors,
        evidence=evidence,
        source="template",
    )
    diagnose_fn = getattr(backend, "diagnose", None)
    if diagnose_fn is None:
        return template
    context = DraftContext(
        invoice_id=invoice.invoice_id,
        supplier_name=invoice.supplier_name,
        invoice_number=invoice.invoice_number,
        currency=invoice.currency,
        total_amount=invoice.total_amount,
        outcome=triage["outcome"],
        severity=triage["severity"],
        rationale=triage["rationale"],
        match_summary=facts.get("match_summary"),
        match_status=facts.get("match_status"),
        discrepancies=discrepancies,
        ocr_vision="; ".join(document_lines or []) or "not recorded",
        supplier_text=supplier_text,
        policy_details=policies,
    )
    tools_fn = getattr(backend, "diagnose_with_tools", None)
    if tools_fn is not None and toolbox is not None:
        try:
            llm = tools_fn(
                context, TOOL_SPECS, toolbox.dispatch, diagnosis_max_tool_calls()
            )
        except Exception as exc:  # the diagnosis never fails the run
            llm = None
            template.note = (
                f"LLM agentic diagnosis failed ({exc}) — template diagnosis used"
            )
        if llm:
            return Diagnosis(
                summary=llm["summary"],
                risk_factors=llm["risk_factors"],
                evidence=evidence,
                source="llm",
                tool_calls=llm.get("tool_calls", []),
            )
        if not template.note:
            template.note = (
                "LLM agentic diagnosis reply was unusable — template diagnosis used"
            )
        return template
    try:
        llm = diagnose_fn(context)
    except Exception as exc:  # the diagnosis never fails the run
        llm = None
        template.note = f"LLM diagnosis failed ({exc}) — template diagnosis used"
    if not llm:
        if not template.note:
            template.note = "LLM diagnosis reply was unusable — template diagnosis used"
        return template
    return Diagnosis(
        summary=llm["summary"],
        risk_factors=llm["risk_factors"],
        evidence=evidence,
        source="llm",
    )
