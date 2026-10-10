"""Deterministic triage rules over the computed three-way match.

Rules run first and independently. They read only computed facts
(``matching.py``), the OCR/vision verdicts, and the fraud screen —
never model output — so the same invoice always triages the same
way. Priority order is explicit: document problems first (you cannot
match what you cannot read), then fraud/duplicate holds, then
structural gaps (no PO, no receipt), then match variances, then
clean.

The LLM cross-check (see ``crosscheck.py``) can only change the
outcome under the one explicit resolution policy.
"""

from __future__ import annotations

from .schemas import InvoiceInput, MatchFacts, OcrResult, Triage, VisionResult

# An invoice at or above this total is high severity when it carries
# any exception (approval-threshold policy AP-APPR-01).
HIGH_VALUE_THRESHOLD = 10_000.0


def classify_invoice(
    invoice: InvoiceInput,
    facts: MatchFacts,
    vision: VisionResult | None = None,
    ocr_results: list[OcrResult] | None = None,
    fraud_flags: list[str] | None = None,
) -> Triage:
    amount = facts.invoice_total_stated or facts.invoice_lines_total or 0.0
    value_band = "high" if amount >= HIGH_VALUE_THRESHOLD else "moderate"

    if vision is not None and not vision.is_invoice:
        return Triage(
            outcome="not_an_invoice",
            severity="moderate",
            confidence=round(vision.confidence or 0.8, 2),
            rationale=(
                f"The primary document verified as '{vision.document_type_detected}', "
                "not an invoice — statements and remittance advices are not payable (AP-DOC-02)."
            ),
            signals=[f"vision: document type {vision.document_type_detected}"],
        )

    low_ocr = [r for r in (ocr_results or []) if r.low_confidence]
    if (vision is not None and vision.quality == "unreadable") or (
        low_ocr and vision is not None and not vision.legible
    ):
        return Triage(
            outcome="unreadable",
            severity="moderate",
            confidence=0.9,
            rationale=(
                "The invoice document cannot be read reliably (OCR confidence "
                "below threshold / illegible scan) — a legible resubmission is "
                "required; amounts are never processed on a guess (AP-DOC-01)."
            ),
            signals=[f"ocr low confidence: {r.document_id}" for r in low_ocr]
            or ["vision: quality unreadable"],
        )

    # Note: bank-change fraud flags deliberately do NOT change the
    # triage outcome — the match result is the match result. The fraud
    # screen acts on the case around the triage: it forces the
    # hold-for-review option, blocks the autonomy recommendation, and
    # arms a guardrail on the drafts (see graph/options/autonomy).
    _ = fraud_flags

    if facts.duplicate_suspected:
        return Triage(
            outcome="duplicate_suspected",
            severity="high",
            confidence=0.93,
            rationale=(
                f"Same supplier, invoice number and amount as {facts.duplicate_of} "
                "already in the system — a suspected duplicate is held, never "
                "paid twice (AP-DUP-01)."
            ),
            signals=[f"duplicate of: {facts.duplicate_of}"],
        )

    if not facts.has_po:
        return Triage(
            outcome="missing_po",
            severity=value_band,
            confidence=0.94,
            rationale=(
                "No purchase order is attached, so the three-way match cannot "
                "run. No PO, no payment (AP-PO-01): route to purchasing."
            ),
            signals=["match: no purchase order"],
        )

    if not facts.has_receipt:
        return Triage(
            outcome="missing_receipt",
            severity=value_band,
            confidence=0.9,
            rationale=(
                "A purchase order exists but no goods receipt is recorded — "
                "receipt must be confirmed before payment preparation (AP-PO-02)."
            ),
            signals=["match: no goods receipt"],
        )

    statuses = [m.status for m in facts.line_matches]
    if "price_variance" in statuses:
        worst = max(
            (abs(m.price_variance_pct) for m in facts.line_matches if m.status == "price_variance"),
            default=0.0,
        )
        return Triage(
            outcome="price_variance",
            severity="high" if worst > 5 else value_band,
            confidence=0.92,
            rationale=(
                f"At least one line's unit price exceeds the PO price beyond "
                f"tolerance (worst {worst:+.2f}%). Variances above 5% route to "
                "purchasing (AP-MATCH-02)."
            ),
            signals=[f"line {m.line_no}: price variance {m.price_variance_pct:+.2f}%"
                     for m in facts.line_matches if m.status == "price_variance"],
        )

    if any(s in statuses for s in ("quantity_variance", "short_received", "missing_on_po")):
        return Triage(
            outcome="quantity_variance",
            severity=value_band,
            confidence=0.92,
            rationale=(
                "Invoiced quantities exceed what was ordered or received. "
                "Payment is never prepared for quantities not received (AP-MATCH-03)."
            ),
            signals=[f"line {m.line_no}: {m.status} ({m.detail})"
                     for m in facts.line_matches if m.status != "matched"],
        )

    if facts.match_status != "full_match" or abs(facts.tax_variance) > 1.00 \
            or not facts.currency_consistent:
        reasons = []
        if not facts.currency_consistent:
            reasons.append("currency differs between invoice and purchase order (AP-CURR-01)")
        if abs(facts.tax_variance) > 1.00:
            reasons.append(
                f"tax does not reconcile (stated {facts.tax_stated}, expected "
                f"{facts.tax_expected}) (AP-TAX-01)"
            )
        if facts.match_status not in {"full_match"}:
            reasons.append(f"match status {facts.match_status}")
        return Triage(
            outcome="price_variance",
            severity=value_band,
            confidence=0.85,
            rationale="; ".join(reasons) + ".",
            signals=reasons,
        )

    return Triage(
        outcome="clean_match",
        severity="low",
        confidence=0.95,
        rationale=(
            "Every line matches the PO and the receipt within tolerance, totals "
            "and tax reconcile, currency agrees — a clean three-way match (AP-MATCH-01)."
        ),
        signals=[f"match: {facts.matched_lines}/{facts.total_lines} lines matched"],
    )
