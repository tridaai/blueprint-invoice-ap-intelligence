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

# Currency amounts as a letter states them: with a symbol ($1,180.00),
# with a currency code (USD 1180 / 1180 USD), or labelled in prose
# ("subtotal of 999.00"). The supplier letter is prose a model writes;
# figures in it must trace to the computed fact block (see the
# figures_must_trace check below) — the judge pack caught letters
# restating computed arithmetic in prose and drifting (a subtotal
# stated as 1000.00 against a computed 1070.00).
_AMOUNT_SYMBOL = re.compile(r"[$€£]\s?(\d[\d,]*(?:\.\d{1,2})?)")
_AMOUNT_CODE_AFTER = re.compile(
    r"\b(?:USD|EUR|GBP|INR|CAD|AUD|CHF|JPY)\s?(\d[\d,]*(?:\.\d{1,2})?)", re.I
)
_AMOUNT_CODE_BEFORE = re.compile(
    r"(\d[\d,]*(?:\.\d{1,2})?)\s?(?:USD|EUR|GBP|INR|CAD|AUD|CHF|JPY)\b", re.I
)
_AMOUNT_LABELLED = re.compile(
    r"\b(?:subtotal|total|amount|tax|balance|gross|net|sum)\b"
    r"[^.\d]{0,24}?(\d[\d,]*\.\d{1,2})\b",
    re.I,
)


def _letter_amounts(letter: str) -> list[float]:
    """Every currency amount the supplier letter states, as floats."""
    found: list[float] = []
    for pattern in (
        _AMOUNT_SYMBOL,
        _AMOUNT_CODE_AFTER,
        _AMOUNT_CODE_BEFORE,
        _AMOUNT_LABELLED,
    ):
        for match in pattern.finditer(letter):
            try:
                value = float(match.group(1).replace(",", ""))
            except ValueError:
                continue
            if value > 0:
                found.append(round(value, 2))
    return found


def _invoice_amounts(invoice: InvoiceInput) -> list[float]:
    """The fallback figure registry, derived from the invoice itself:
    its stated totals, tax, line products and unit prices. Callers
    with the computed match facts pass the fuller registry (graph.py's
    facts view) — this fallback keeps direct callers honest."""
    amounts: list[float] = []
    for value in (invoice.total_amount, invoice.tax_amount):
        if value is not None:
            amounts.append(float(value))
    for line in invoice.line_items:
        amounts.append(round(line.quantity * line.unit_price, 2))
        amounts.append(float(line.unit_price))
    po = invoice.purchase_order
    if po is not None:
        if po.total is not None:
            amounts.append(float(po.total))
        for line in po.lines:
            amounts.append(round(line.quantity * line.unit_price, 2))
            amounts.append(float(line.unit_price))
    return amounts


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


# What a repair redraft must DO about each failed check, in the
# drafter's own terms. The repair prompt names the failed checks
# (``graph._failed_checks_text``) and appends the directive for
# each — a bare "the drafts never state the next step" left a live
# Nemotron redraft describing the next step in fresh prose and
# failing the same check a second time; the directive names the
# required element itself, so the fix is mechanical, not a guess.
CHECK_REPAIR_DIRECTIVES = {
    "references_invoice_id": (
        "name the invoice id in the subject, the letter, or the "
        "packet — the drafts must say which case they belong to"
    ),
    "no_payment_language": (
        "remove every statement that a payment was made or will be "
        "made, and any bank-detail content — bank changes are "
        "verified out-of-band, never echoed in a draft"
    ),
    "no_prohibited_promises": (
        "remove every promised outcome or timing — describe the "
        "process, never commit to a result or a date"
    ),
    "no_pii_in_draft": (
        "remove the personal identifier — drafts never carry SSN, "
        "card, or passport numbers"
    ),
    "policy_citations_present": (
        "cite the governing policies in the packet, in [AP-...] "
        "form, from the retrieved set in the facts"
    ),
    "citations_within_retrieved": (
        "cite only policies from the retrieved set in the facts — "
        "remove every citation that was not retrieved for this case"
    ),
    "figures_must_trace": (
        "take the amounts out of the letter body — figures live in "
        "the packet's tables; the letter refers to them, it does "
        "not restate arithmetic in prose"
    ),
    "states_next_step": (
        'add a line beginning "Next step:" to the supplier letter, '
        "saying what happens now and what (if anything) AP still "
        "needs — describing the next step in other words does not "
        "satisfy the check"
    ),
}


def repair_directive(check_name: str) -> str:
    """The fix a repair redraft is told to make for one failed
    check ("" when the check has no directive — the check's own
    detail then carries the instruction)."""
    return CHECK_REPAIR_DIRECTIVES.get(check_name, "")


def validate_drafts(
    invoice: InvoiceInput,
    drafts: Drafts,
    citations: list[str],
    allowed_amounts: list[float] | None = None,
) -> Validation:
    """Run all guardrail checks over the drafts. ``passed`` is the
    conjunction — every check here is blocking. ``allowed_amounts``
    is the verified figure registry the letter's amounts must trace
    to (the computed match facts); when omitted it is derived from
    the invoice itself."""
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

    # Citations must come from the retrieved set. A provider draft
    # that cites a plausible-but-unretrieved policy (the judge pack
    # caught [AP-DOC-02] in a letter) is inventing authority — the
    # verify checklist flags it as ungrounded, and this check makes
    # it blocking, over letter and packet alike.
    cited = sorted(set(re.findall(r"\[(AP-[A-Z]+-\d+)\]", combined)))
    unretrieved = [c for c in cited if c not in citations]
    checks.append(
        GuardrailCheck(
            name="citations_within_retrieved",
            passed=not unretrieved,
            detail=(
                "every cited policy was retrieved for this invoice"
                if not unretrieved
                else (
                    f"drafts cite policies that were not retrieved: "
                    f"{unretrieved} — citations come from the retrieved "
                    "set, never from memory"
                )
            ),
        )
    )

    # Figures in the supplier letter must trace to the verified fact
    # block. Figures live in structured fields and the packet's
    # tables; a letter that restates arithmetic in prose drifts (the
    # judge pack: a subtotal of 1000.00 against a computed 1070.00).
    # The packet is exempt — it prints the computed figures directly.
    registry = {
        round(float(v), 2)
        for v in (
            allowed_amounts
            if allowed_amounts is not None
            else _invoice_amounts(invoice)
        )
    }
    registry.add(0.0)
    stated = _letter_amounts(letter)
    untraceable = sorted(
        {a for a in stated if all(abs(a - r) > 0.011 for r in registry)}
    )
    checks.append(
        GuardrailCheck(
            name="figures_must_trace",
            passed=not untraceable,
            detail=(
                f"every figure in the supplier letter traces to the "
                f"computed fact block ({len(stated)} figure(s) checked)"
                if not untraceable
                else (
                    "supplier letter states figure(s) that do not trace "
                    "to the computed facts: "
                    + ", ".join(f"{a:.2f}" for a in untraceable[:3])
                    + " — figures live in the packet's tables; the letter "
                    "refers to them, it does not restate amounts that were "
                    "never computed"
                )
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
