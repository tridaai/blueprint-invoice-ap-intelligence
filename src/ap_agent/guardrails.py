"""Guardrails as code: deterministic checks on every draft.

These run on the supplier letter and the AP packet in every mode —
template, LLM, and repaired drafts alike — and a failed blocking
check makes the draft unapprovable (the service layer refuses
approval and the API returns 422).

The rules encode the blueprint's core promise: **the agent never
states or implies that a payment has been made or will be made, never
quotes or acts on bank-detail changes from documents, and never
promises an outcome.** A supplier's own words can violate them — a
supplier relaying "payment will be made by Friday, please confirm" is
reported speech, not an instruction, so the injection screen leaves
it alone; when a drafter pastes it into a letter, these checks are
what stop it. See the guardrail-trap sample.
"""

from __future__ import annotations

import re

from .schemas import Drafts, GuardrailCheck, InvoiceInput, Validation

# Affirmative payment language. Each match is checked against a
# negation window (see _affirmative): "no payment is made by this
# letter" is the required disclaimer, not a violation.
_PAYMENT_PATTERNS = [
    r"\bpayment (has been|was|is) (made|sent|issued|processed|released)\b",
    r"\bpayment\b[^.]{0,30}\bwill be made\b",
    r"\bpayment will be\b",
    r"\b(has been|was|is|will be) paid\b",
    r"\bwe (have|'ve) paid\b",
    r"\bpaid (on|by) (the )?\d",
    r"\bremittance (has been|was) sent\b",
    r"\bfunds (have been|were|will be) (transferred|sent|released)\b",
]

# Bank-detail content in a draft: quoting account identifiers or
# directing payment to a (new) account. The fraud control (AP-BANK-01)
# is that drafts never carry this content at all — a bank change is
# verified out-of-band, never echoed back.
_BANK_PATTERNS = [
    r"\baccount number\b[^.]{0,25}\d",
    r"\bsort code\b",
    r"\biban\b\s?[A-Z0-9]{6,}",
    r"\bnew bank account\b",
    r"\bupdated bank (account|details)\b",
    r"\bremit\b[^.]{0,50}\baccount\b",
    r"\baccount\b[^.]{0,20}\b\d{6,}\b",
]

_NEGATION = re.compile(r"\b(not|never|no|n't|nor)\b", re.IGNORECASE)


def _affirmative(pattern: str, text: str) -> str | None:
    """First match of ``pattern`` not preceded by a negation within
    30 characters — returns the matched phrase, or None."""
    for match in re.finditer(pattern, text, re.IGNORECASE):
        window = text[max(0, match.start() - 30):match.start()]
        if _NEGATION.search(window):
            continue
        return match.group(0)
    return None


_PROMISE_PATTERNS = [
    # Outcome promises — the agent may never promise a decision.
    r"\bwe (promise|guarantee)\b",
    r"\bguaranteed (approval|acceptance|payment|credit)\b",
    r"\bwill be (approved|accepted|credited)\b",
    r"\bapproval is (guaranteed|assured|certain)\b",
    r"\byou will (be|get) (approved|credited|a credit)\b",
    r"\bwe will (approve|pay|credit)\b",
    # Time commitments: a promise about WHEN something will happen,
    # without naming who does it ("this will be resolved within 5
    # days") — the same failure class as an explicit promise.
    r"\bwill (arrive|be sent|be issued|be resolved|be processed|be ready|reach you)\b[^.]{0,40}\b(within|in|by)\b[^.]{0,20}\b(hours?|days?|weeks?|tomorrow|eod|end of day|end of week)\b",
    r"\bwithin (the next )?\d+ (hours?|days?|weeks?)\b",
    r"\bby (tomorrow|end of day|eod|end of week|monday|tuesday|wednesday|thursday|friday)\b[^.]{0,30}\b(will|you|payment|decision|credit)\b",
]

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PASSPORT = re.compile(r"\b[A-Z]\d{8}\b")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _find_card(text: str) -> str | None:
    for match in re.finditer(r"\b(?:\d[ -]?){13,19}\b", text):
        digits = re.sub(r"[ -]", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return match.group(0)
    return None


def validate_drafts(
    invoice: InvoiceInput,
    drafts: Drafts,
    citations: list[str],
) -> Validation:
    """Run all guardrail checks over the drafts. ``passed`` is the
    conjunction — every check here is blocking."""
    letter = drafts.supplier_letter or ""
    packet = drafts.ap_packet or ""
    combined = f"{drafts.subject}\n{letter}\n{packet}"
    checks: list[GuardrailCheck] = []

    present = invoice.invoice_id in combined
    checks.append(
        GuardrailCheck(
            name="references_invoice_id",
            passed=present,
            detail=(
                f"invoice id {invoice.invoice_id} referenced"
                if present
                else f"drafts never reference invoice {invoice.invoice_id}"
            ),
        )
    )

    payment_hit = next(
        (hit for hit in (_affirmative(p, combined) for p in _PAYMENT_PATTERNS) if hit),
        None,
    )
    bank_hit = next(
        (m.group(0) for p in _BANK_PATTERNS if (m := re.search(p, combined, re.IGNORECASE))),
        None,
    )
    language_hit = payment_hit or bank_hit
    checks.append(
        GuardrailCheck(
            name="no_payment_language",
            passed=language_hit is None,
            detail=(
                "no payment-made / payment-promised language and no bank-detail content"
                if language_hit is None
                else (
                    f"payment language found: '{payment_hit}' — the agent never states a payment was made or will be made"
                    if payment_hit
                    else f"bank-detail content found: '{bank_hit}' — bank changes are verified out-of-band, never echoed in drafts [AP-BANK-01]"
                )
            ),
        )
    )

    promise_hit = next(
        (m.group(0) for p in _PROMISE_PATTERNS if (m := re.search(p, combined, re.IGNORECASE))),
        None,
    )
    checks.append(
        GuardrailCheck(
            name="no_prohibited_promises",
            passed=promise_hit is None,
            detail=(
                "no outcome or timing promises"
                if promise_hit is None
                else f"prohibited promise found: '{promise_hit}'"
            ),
        )
    )

    pii_hit = None
    if _SSN.search(combined):
        pii_hit = "what looks like a US Social Security number"
    elif _find_card(combined):
        pii_hit = "what looks like a payment card number (Luhn-valid)"
    elif _PASSPORT.search(combined):
        pii_hit = "what looks like a passport number"
    checks.append(
        GuardrailCheck(
            name="no_pii_in_draft",
            passed=pii_hit is None,
            detail=(
                "no SSN / card / passport patterns in the drafts"
                if pii_hit is None
                else f"PII in draft: {pii_hit} — personal identifiers never belong in drafts"
            ),
        )
    )

    checks.append(
        GuardrailCheck(
            name="policy_citations_present",
            passed=bool(citations),
            detail=(
                f"policy citations present: {citations}"
                if citations
                else "no policy citations — every triage must cite its policies"
            ),
        )
    )

    next_step = "next step" in combined.lower()
    checks.append(
        GuardrailCheck(
            name="states_next_step",
            passed=next_step,
            detail=(
                "the drafts state the next step"
                if next_step
                else "the drafts never state the next step for the supplier"
            ),
        )
    )

    return Validation(passed=all(c.passed for c in checks), checks=checks)
