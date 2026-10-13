"""Self-verification node logic: the agent critiques its own drafts.

Two paths, one contract (``VerifyResult``):

- **LLM critique** (provider mode): the drafting model is asked to
  check its own drafts against the verified facts — citations it was
  not given, figures that are not in the computed match, payment or
  bank-detail language. Self-critique catches composition errors the
  deterministic checklist cannot see.
- **Deterministic checklist** (default mode, and the fallback when an
  LLM critique is unusable): citation IDs in the drafts must be among
  the retrieved policies; every monetary figure stated must be one of
  the computed amounts (invoice total, lines total, tax — drafts may
  restate computed figures, never invent them); payment language is
  flagged; a foreign invoice id is flagged.

Verification never blocks a run by itself — it is evidence for the
approver and an input to the autonomy recommendation. The hard
stop is the guardrail layer.
"""

from __future__ import annotations

import re

from .guardrails import _PAYMENT_PATTERNS, _affirmative
from .model_backends import DraftContext
from .schemas import Drafts, VerifyResult

_CITATION = re.compile(r"\[(AP-[A-Z]+-\d+)\]")
_AMOUNT = re.compile(r"[$£€]?\s?\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|[$£€]\s?\d+(?:\.\d{1,2})?|\d+\.\d{2}")
_ID_TOKEN = re.compile(r"\bINV-\d{3,}\b")


def _amounts_in(text: str) -> list[float]:
    amounts: list[float] = []
    for match in _AMOUNT.finditer(text):
        if text[match.end():match.end() + 1] == "%":
            continue  # a percentage, not a monetary figure
        raw = re.sub(r"[$£€,\s]", "", match.group(0))
        try:
            value = float(raw)
        except ValueError:
            continue
        if value < 1.0:
            continue  # sub-unit decimals in drafts are confidences, not money
        amounts.append(value)
    return amounts


def checklist_grounding(
    *,
    drafts: Drafts,
    retrieved_ids: list[str],
    invoice_id: str,
    allowed_amounts: list[float] | None = None,
    related_ids: list[str] | None = None,
) -> list[str]:
    """Deterministic grounding issues for a draft (empty = grounded)."""
    combined = f"{drafts.subject}\n{drafts.supplier_letter}\n{drafts.ap_packet}"
    issues: list[str] = []
    for citation in sorted(set(_CITATION.findall(combined))):
        if citation not in retrieved_ids:
            issues.append(
                f"draft cites policy {citation}, which was not retrieved for this invoice"
            )
    allowed = allowed_amounts or []
    for amount in sorted(set(_amounts_in(combined))):
        if not any(abs(amount - known) <= 0.01 for known in allowed):
            issues.append(
                f"draft states a figure ({amount}) that is not among the computed "
                "amounts for this invoice — figures come from the match, not from prose"
            )
    hit = next(
        (h for h in (_affirmative(p, combined) for p in _PAYMENT_PATTERNS) if h), None
    )
    if hit:
        issues.append(f"draft contains payment language: '{hit}'")
    known_ids = {invoice_id, *(related_ids or [])}
    for token in sorted(set(_ID_TOKEN.findall(combined))):
        if token not in known_ids:
            issues.append(f"draft references a different invoice id ({token})")
    return issues


def verify_drafts(
    *,
    invoice,
    triage: dict,
    facts: dict,
    drafts: Drafts,
    retrieved_ids: list[str],
    backend,
    policies: list[dict] | None = None,
) -> VerifyResult:
    """Verify a draft pair: LLM critique in provider mode, checklist otherwise.

    ``policies`` are the retrieved policies the drafts were grounded
    in (id, title, text). The LLM critique is shown them so it can
    judge citations against the set that was actually retrieved —
    without them its prompt reads "none retrieved" and every
    citation looks invented."""
    issues = checklist_grounding(
        drafts=drafts,
        retrieved_ids=retrieved_ids,
        invoice_id=invoice.invoice_id,
        allowed_amounts=facts.get("allowed_amounts"),
        related_ids=[facts["duplicate_of"]] if facts.get("duplicate_of") else [],
    )
    verify_fn = getattr(backend, "verify_draft", None)
    if verify_fn is None:
        return VerifyResult(grounded=not issues, issues=issues, source="checklist")
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
        subject=drafts.subject,
        supplier_letter=drafts.supplier_letter,
        ap_packet=drafts.ap_packet,
        policy_details=list(policies or []),
    )
    try:
        critique = verify_fn(context)
    except Exception as exc:  # verification never fails the run
        return VerifyResult(
            grounded=not issues,
            issues=issues,
            source="checklist",
            note=f"LLM self-verification failed ({exc}) — deterministic checklist used",
        )
    if critique is None:
        return VerifyResult(
            grounded=not issues,
            issues=issues,
            source="checklist",
            note="LLM self-verification reply was unusable — deterministic checklist used",
        )
    merged = list(dict.fromkeys([*issues, *critique["issues"]]))
    return VerifyResult(
        grounded=critique["grounded"] and not issues,
        issues=merged,
        source="llm",
        note=critique.get("summary", ""),
    )
