"""One-command demo: ONE invoice, end to end, with a readable trace.

    ap-agent-demo            (console script)
    python -m ap_agent demo  (module form)
    make demo

Default: offline, deterministic mock backend, synthetic sample data. Set
MODEL_BACKEND=openai with the NVIDIA NIM endpoint (the lead profile —
see the README) or another provider, and the same trace runs against
the real model — the header line says which backend produced the
drafts. The trace shows every stage a reviewer needs to judge the
blueprint: input → OCR + vision verify → screening → extraction →
three-way match → triage with cross-check → retrieved policies →
diagnosis → scored options → drafts → self-verification → independent
review → guardrail checks → the approver gate it stops at.
"""

from __future__ import annotations

import argparse
import sys

from .config import silence_langchain_deprecation_warnings

# Must precede the graph import: the warning fires while langgraph loads.
silence_langchain_deprecation_warnings()

from .graph import run_invoice
from .matching import format_match_summary
from .ports import ModelBackend, Retriever
from .samples import sample_invoice_models
from .wiring import build_backend, build_retriever

_LINE = "=" * 74


def print_trace(
    index: int = 0,
    backend: ModelBackend | None = None,
    retriever: Retriever | None = None,
) -> None:
    backend = backend or build_backend()
    retriever = retriever or build_retriever()
    invoices = sample_invoice_models()
    invoice = invoices[index]
    result = run_invoice(invoice, backend=backend, retriever=retriever)
    t = result.triage
    backend_desc = (
        "offline mock backend"
        if backend.name == "mock"
        else f"{backend.name} backend via MODEL_BACKEND (real provider calls)"
    )

    print(_LINE)
    print("TRIDA AI BLUEPRINT DEMO — Invoice / AP Document Intelligence Agent")
    print(f"Reference prototype · synthetic data · {backend_desc} · no external action")
    print(_LINE)

    print("\n[1] INPUT INVOICE")
    print(f"    ID          : {invoice.invoice_id}")
    print(f"    Supplier    : {invoice.supplier_name} · invoice number {invoice.invoice_number or '—'}")
    print(f"    Amount      : {invoice.currency} {invoice.total_amount} (tax {invoice.tax_amount}) · due {invoice.due_date or '—'}")
    print(f"    Line items  : {len(invoice.line_items)} · PO: {invoice.purchase_order.po_number if invoice.purchase_order else 'none'} · receipt: {invoice.goods_receipt.receipt_id if invoice.goods_receipt else 'none'}")
    print(f"    Documents   : {[d.document_id or d.doc_type for d in invoice.documents] or 'none'}")

    print("\n[2] OCR + VISION VERIFY")
    for r in result.ocr_results:
        print(f"    OCR {r.document_id or r.doc_type}: engine {r.engine} · mean confidence {r.mean_confidence} · {r.char_count} chars" + (" · LOW CONFIDENCE" if r.low_confidence else ""))
    if result.vision is not None:
        v = result.vision
        print(f"    Vision: detected {v.document_type_detected} · is_invoice {v.is_invoice} · legible {v.legible} · quality {v.quality} (source: {v.source})")

    print("\n[3] SCREENING — injection + bank-change fraud")
    for flag in result.injection_flags:
        print(f"    ⚠ Injection screen: FLAG — {flag}")
    for flag in result.fraud_flags:
        print(f"    ⛔ Fraud screen: FLAG — {flag}")
    if result.bank_change_blocked:
        print("      (bank-detail change request: case blocked for payment prep; the new details are never echoed in drafts [AP-BANK-01])")
    if not result.injection_flags and not result.fraud_flags:
        print("    No injection or fraud patterns found in supplier text.")
    for ext in result.extractions:
        notable = [fl for fl in ext.fields if fl.status not in ("match", "source_provided")]
        if notable:
            for fl in notable:
                print(f"    Extraction  : {ext.document_id} field '{fl.field}' — {fl.status} (confidence {fl.confidence})")

    print("\n[4] THREE-WAY MATCH (computed in code)")
    f = result.facts
    print(f"    {format_match_summary(f)}")
    for m in f.line_matches:
        print(f"      L{m.line_no} {m.description}: {m.status} — {m.detail}")
    if f.duplicate_suspected:
        print(f"    Duplicate: suspected duplicate of {f.duplicate_of} [AP-DUP-01]")

    print("\n[5] TRIAGE")
    print(f"    Outcome   : {t.outcome}")
    print(f"    Severity  : {t.severity}   Confidence: {t.confidence}")
    print(f"    Rationale : {t.rationale}")
    print("    Evidence (signals):")
    for signal in t.signals or ["(none)"]:
        print(f"      - {signal}")
    if result.cross_check is not None:
        cc = result.cross_check
        print(f"    Rules       : {cc.rule_outcome}")
        if cc.llm_outcome is not None:
            print(f"    LLM check   : {cc.llm_outcome} · severity {cc.llm_severity} · confidence {cc.llm_confidence}")
        else:
            print("    LLM check   : no usable LLM triage — rules only")
        print(f"    Cross-check : {cc.resolution} — {cc.note}")

    print("\n[6] RETRIEVED AP POLICY CONTEXT")
    for g in result.policies:
        snippet = g.text if len(g.text) <= 160 else g.text[:157] + "..."
        print(f"    [{g.policy_id}] {g.title} (score {g.score}, via {'+'.join(g.sources) or 'keyword'})")
        print(f"        {snippet}")
    if not result.policies:
        print("    (no policies retrieved)")

    print("\n[7] DIAGNOSIS")
    print(f"    Summary   : {result.diagnosis.summary} (composed by: {result.diagnosis.source})")
    for factor in result.diagnosis.risk_factors:
        print(f"      - factor: {factor}")
    for call in result.diagnosis.tool_calls:
        print(f"      - tool call: {call.get('name')} — {call.get('summary')}")
    if result.diagnosis.note:
        print(f"    Note      : {result.diagnosis.note}")
    for line in result.diagnosis.evidence:
        print(f"      - evidence: {line}")

    print("\n[8] NEXT-STEP OPTIONS (scored by code — the model never does this arithmetic)")
    for option in result.options:
        mark = " <- recommended" if option.recommended else ""
        print(f"    [{option.kind}] {option.title} — score {option.score}{mark}")

    print("\n[9] DRAFT — SUPPLIER QUERY LETTER")
    if result.drafts is not None:
        print(f"    Subject: {result.drafts.subject}")
        for line in result.drafts.supplier_letter.splitlines():
            print(f"    {line}")

    print("\n[10] DRAFT — AP PACKET (internal, for the finance approver's decision)")
    if result.drafts is not None:
        for line in result.drafts.ap_packet.splitlines():
            print(f"    {line}")

    print("\n[11] SELF-VERIFICATION — the agent critiques its own drafts")
    if result.verification is not None:
        verdict = "GROUNDED" if result.verification.grounded else "NOT GROUNDED"
        source = "LLM critique" if result.verification.source == "llm" else "deterministic checklist"
        print(f"    verdict: {verdict} ({source})")
        for issue in result.verification.issues:
            print(f"    issue: {issue}")
        if result.verification.note:
            print(f"    note: {result.verification.note}")

    print("\n[12] INDEPENDENT REVIEW — a second pair of eyes (not the drafter)")
    if result.review is not None:
        r = result.review
        source = "LLM reviewer" if r.source == "llm" else "deterministic checklist"
        print(f"    verdict: {r.verdict.upper()} ({source})")
        for finding in r.findings:
            print(f"    finding: {finding}")
        if result.reviewer_blocked:
            print("    ⛔ REVIEWER BLOCKED this draft — flagged for the approver; the human still decides.")
    else:
        print("    (disabled — REVIEWER=off)")

    print("\n[13] GUARDRAIL CHECKS")
    if result.validation is not None:
        for check in result.validation.checks:
            mark = "PASS" if check.passed else "FAIL"
            print(f"    [{mark}] {check.name} — {check.detail}")
        print(f"    Overall: {'PASSED' if result.validation.passed else 'FAILED'}")
    if result.repair_attempted:
        outcome = "redraft passed" if result.repaired else "redraft still failed"
        print(f"    Repair: bounded repair attempted ({result.repair_attempts} attempt(s)) — {outcome}")
        if result.original_validation is not None:
            for failed in [c for c in result.original_validation.checks if not c.passed]:
                print(f"    Original failure: {failed.name} — {failed.detail}")

    print("\n[14] FINAL STATE")
    print("    AWAITING_APPROVER" if result.status == "awaiting_approver" else f"    {result.status}")
    print(f"    status = {result.status} · external_action_taken = {result.external_action_taken}")
    label = "eligible for auto-triage" if result.autonomy.eligible_for_auto_triage else "human decision required"
    print(f"    Autonomy recommendation: {label} (recommendation only — never acted on)")
    for reason in result.autonomy.reasons:
        print(f"      - {reason}")
    if result.needs_information and result.information_request is not None:
        print("    Information : NEEDED — a request was composed (not sent):")
        for item in result.information_request.missing_items:
            print(f"      - missing: {item}")
    if result.telemetry is not None:
        tel = result.telemetry
        budget_text = ""
        if tel.budget is not None:
            budget_text = f" · token budget {tel.budget['used']}/{tel.budget['limit']}" + (
                " — EXCEEDED (later provider steps degraded to their template paths)"
                if tel.budget["exceeded"]
                else ""
            )
        if tel.input_tokens is not None:
            cost = (
                f" · est. ${tel.estimated_cost_usd:.4f}"
                if tel.estimated_cost_usd is not None
                else " · cost n/a (model not in the price table)"
            )
            print(
                f"    Telemetry : {tel.backend} · {tel.model} · {tel.model_calls} model call(s) · "
                f"{tel.input_tokens} in / {tel.output_tokens} out tokens{cost} · {tel.latency_seconds}s{budget_text}"
            )
        else:
            print(
                f"    Telemetry : {tel.backend} (offline fallback — no model ran) · "
                f"{tel.model_calls} call(s) · tokens n/a · {tel.latency_seconds}s{budget_text}"
            )
    print("    Nothing was approved, paid, sent, or filed. A finance approver decides via the API/UI.")
    print(_LINE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Invoice / AP Document Intelligence Agent — traced demo (offline)")
    parser.add_argument("--index", type=int, default=0, help="Which bundled sample invoice to run (0-based)")
    args = parser.parse_args()
    try:
        print_trace(args.index)
    except (RuntimeError, ValueError) as exc:
        # e.g. MODEL_BACKEND=openai with no key — fail loudly and cleanly.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
