"""Duplicate-payment recovery packs — drafted, never executed.

Duplicate detection (AP-DUP-01) holds the second copy at the gate.
But sometimes the sequence runs the other way: the original was
already approved — already on its way to payment — when the
duplicate arrives. Holding the copy is no longer enough; AP has a
recovery case: one amount, two approvals in flight.

This module builds that case from the two stored records:

- The evidence is computed: same supplier, same invoice number,
  same total, both submissions' dates, and the exposure — the
  original approved with the copy still at the gate, or (worst
  case) both copies approved.
- The drafts are a supplier recovery letter (issue a credit note
  for the duplicated amount, or refund it) and an internal memo
  (what happened, what to stop, what to chase). They are composed
  deterministically from the computed evidence — in this surface a
  letter that invents a figure is worse than a plain one — and then
  put through the standard path: the shared self-verification
  (``verify_drafts``) and the guardrails as code
  (``validate_drafts``), exactly as pipeline drafts are.
- The pack is stored for a human to send and act on. The agent
  recovers nothing itself: no credit is booked, no payment is
  reversed, no letter is sent.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .guardrails import validate_drafts
from .matching import format_match_summary
from .schemas import Drafts, InvoiceInput, RecoveryCaseEvent, RecoveryPack
from .store import ApprovalRecord
from .verify import verify_drafts


def _allowed_amounts(*records: ApprovalRecord) -> list[float]:
    """The grounding registry for the recovery drafts: every figure
    the two runs computed, so a draft quoting them is grounded by
    construction (the same rule the pipeline's facts view applies)."""
    allowed: list[float] = []
    for record in records:
        facts = record.result.facts
        invoice = record.invoice or {}
        for value in (
            facts.invoice_total_stated,
            facts.invoice_lines_total,
            facts.tax_stated,
            facts.tax_expected,
            abs(facts.tax_variance) if facts.tax_variance else None,
            invoice.get("total_amount"),
            invoice.get("tax_amount"),
        ):
            if value is not None:
                allowed.append(float(value))
        for m in facts.line_matches:
            allowed.append(round(m.invoice_qty * m.invoice_price, 2))
            allowed.append(round(m.po_qty * m.po_price, 2))
            allowed.append(float(m.invoice_price))
            allowed.append(float(m.po_price))
    return allowed


def recovery_evidence(
    duplicate: ApprovalRecord, twin: ApprovalRecord
) -> tuple[list[str], str, float]:
    """(Evidence lines, exposure status, duplicated amount) computed
    from the two stored records."""
    dup_result = duplicate.result
    twin_result = twin.result
    dup_invoice = duplicate.invoice or {}
    twin_invoice = twin.invoice or {}
    amount = float(
        dup_invoice.get("total_amount")
        or dup_result.facts.invoice_total_stated
        or 0.0
    )
    both_approved = dup_result.status == "approved"
    exposure = "both_approved" if both_approved else "twin_approved"
    evidence = [
        f"same supplier: {dup_result.supplier_name}",
        f"same supplier invoice number: {dup_result.invoice_number}",
        f"same amount: {amount:.2f} {dup_invoice.get('currency') or 'USD'}",
        f"original {twin_result.invoice_id}: submitted "
        f"{twin_invoice.get('submitted_at') or 'unknown'}, status "
        f"{twin_result.status}"
        + (
            f", approved by {twin_result.decided_by}"
            if twin_result.decided_by
            else ""
        ),
        f"duplicate {dup_result.invoice_id}: submitted "
        f"{dup_invoice.get('submitted_at') or 'unknown'}, triage "
        f"{dup_result.triage.outcome} (duplicate of {dup_result.facts.duplicate_of})",
        "exposure: "
        + (
            "both copies are approved — two payments are in flight "
            "for one invoice"
            if both_approved
            else "the original is approved and the duplicate is "
            "still at the gate — one payment in flight, one to stop"
        ),
    ]
    return evidence, exposure, amount


def _compose_drafts(
    duplicate: ApprovalRecord,
    twin: ApprovalRecord,
    evidence: list[str],
    exposure: str,
    amount: float,
    citations: list[str],
) -> Drafts:
    """The recovery letter + internal memo, composed from the
    computed evidence. Wording discipline is the guardrails':
    nothing here states or promises a payment movement — the letter
    asks the supplier to credit or refund, the memo tells the human
    what to do."""
    dup = duplicate.result
    currency = (duplicate.invoice or {}).get("currency") or "USD"
    citation_text = " ".join(f"[{c}]" for c in citations) or "[AP-DUP-01]"
    # The letter names the duplicate control specifically when the
    # run retrieved it; the memo lists every cited policy.
    letter_citation = (
        "[AP-DUP-01]" if "AP-DUP-01" in citations else citation_text
    )
    subject = (
        f"Duplicate of invoice {dup.invoice_number} — recovery for "
        f"{dup.invoice_id} (original {twin.result.invoice_id})"
    )
    letter = (
        f"Dear {dup.supplier_name} accounts team,\n\n"
        f"Our records show that invoice {dup.invoice_number} for "
        f"{currency} {amount:.2f} was processed by us twice: first as "
        f"{twin.result.invoice_id}, which was approved here, and again "
        f"as {dup.invoice_id}. The second copy is a duplicate of the "
        f"same invoice — same supplier, same invoice number, same "
        f"amount — and is held under our duplicate invoice control "
        f"{letter_citation}.\n\n"
        f"Next step: please issue a credit note for {currency} "
        f"{amount:.2f} against invoice {dup.invoice_number}, or refund "
        f"the duplicated amount, and confirm which route you have "
        f"taken. If you believe the two submissions cover two distinct "
        f"deliveries, please reissue the second delivery under a "
        f"distinct invoice number with its delivery evidence, and we "
        f"will process it as a new invoice.\n\n"
        f"This letter records the duplicate and opens the recovery; "
        f"it makes no payment and changes no bank details."
    )
    memo_lines = [
        f"INTERNAL MEMO — duplicate-payment recovery {dup.invoice_id}",
        f"Supplier: {dup.supplier_name} · invoice {dup.invoice_number} "
        f"· amount {currency} {amount:.2f}",
        "Evidence (computed from the two stored runs):",
        *[f"- {line}" for line in evidence],
        f"Duplicate match: {format_match_summary(dup.facts)}",
        "Actions for the approver:",
        (
            "- Both copies are approved: escalate to the finance lead "
            "today; ask the supplier for a credit note or refund of "
            f"{currency} {amount:.2f} using the recovery letter, and "
            "place a hold on further payments to this supplier until "
            "the credit lands."
            if exposure == "both_approved"
            else "- The duplicate is still at the gate: reject "
            f"{dup.invoice_id} with the reason 'duplicate of "
            f"{twin.result.invoice_id}' so it can never be prepared "
            "for payment, then send the recovery letter if the "
            "original has already been settled in the AP system."
        ),
        "- Do not offset the amount against other invoices from this "
        "supplier without a credit note on file.",
        f"Policy: {citation_text}",
        "Next step: the recovery letter is a draft — a human sends "
        "it; this agent takes no external action.",
    ]
    return Drafts(
        subject=subject,
        supplier_letter=letter,
        ap_packet="\n".join(memo_lines),
    )


def build_recovery_pack(
    duplicate: ApprovalRecord, twin: ApprovalRecord, backend
) -> RecoveryPack:
    """Build the recovery pack for one duplicate record and its
    approved twin: evidence in code, drafts composed, then the
    standard verify + guardrail path over them."""
    dup_result = duplicate.result
    invoice = InvoiceInput.model_validate(duplicate.invoice or {})
    evidence, exposure, amount = recovery_evidence(duplicate, twin)
    citations = [p.policy_id for p in dup_result.policies]
    drafts = _compose_drafts(duplicate, twin, evidence, exposure, amount, citations)
    facts_view = {
        "match_status": dup_result.facts.match_status,
        "match_summary": format_match_summary(dup_result.facts),
        "allowed_amounts": _allowed_amounts(duplicate, twin),
        "duplicate_of": twin.result.invoice_id,
    }
    verification = verify_drafts(
        invoice=invoice,
        triage=dup_result.triage.model_dump(),
        facts=facts_view,
        drafts=drafts,
        retrieved_ids=citations,
        backend=backend,
    )
    validation = validate_drafts(
        invoice,
        drafts,
        citations,
        allowed_amounts=facts_view["allowed_amounts"],
        outcome=dup_result.triage.outcome,
        match_status=dup_result.facts.match_status,
    )
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return RecoveryPack(
        pack_id=f"REC-{dup_result.invoice_id}",
        duplicate_invoice_id=dup_result.invoice_id,
        original_invoice_id=twin.result.invoice_id,
        supplier_name=dup_result.supplier_name,
        invoice_number=dup_result.invoice_number,
        currency=(duplicate.invoice or {}).get("currency") or "USD",
        amount=amount,
        exposure_status=exposure,
        evidence=evidence,
        drafts=drafts,
        verification=verification,
        validation=validation,
        status="ready" if validation.passed else "blocked",
        created_at=created,
        case_status="opened",
        case_history=[
            RecoveryCaseEvent(
                status="opened",
                actor="agent",
                at=created,
                note="recovery pack drafted — the case is a human's from here",
            )
        ],
    )
