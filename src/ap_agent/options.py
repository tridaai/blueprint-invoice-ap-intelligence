"""Options node logic: next-step proposals, scored deterministically.

The five option kinds are the only paths an invoice case can take
from triage (``schemas.RecoveryOption.kind``):

- ``approve_for_payment_prep`` — clean match: the packet goes to a
  finance approver with a recommendation to prepare payment. (The
  agent never approves and never pays; this is a recommendation.)
- ``route_to_purchasing`` — purchasing owns the resolution: no PO,
  a price dispute, an order question.
- ``request_credit_note`` — ask the supplier for a credit note for
  the variance or the unreceived quantity.
- ``hold_for_review`` — hold for a finance approver: duplicates,
  fraud indicators, anything unclear.
- ``reject_to_supplier_draft`` — draft returning the document to the
  supplier (not an invoice, unreadable).

In provider mode the LLM may propose the wording of the options; in
every mode their *kinds* are validated here and their scores are
computed by deterministic code (the model never does this arithmetic):

    score = 60 * fit + 40 * match_fit - severity_penalty - amount_penalty

where ``fit`` ∈ [0, 1] is how well the kind matches the triage
outcome, ``match_fit`` how well it matches the computed match status,
``severity_penalty`` applies to proceeding when severity is high, and
``amount_penalty`` applies to proceeding above the approval threshold.
A bank-change fraud flag overrides the fit table for the two extreme
kinds: proceeding scores zero, holding scores full — the fraud hold
is not negotiable by arithmetic.
"""

from __future__ import annotations

from .model_backends import DraftContext
from .schemas import RecoveryOption
from .triage import HIGH_VALUE_THRESHOLD

OPTION_KINDS = {
    "approve_for_payment_prep",
    "route_to_purchasing",
    "request_credit_note",
    "hold_for_review",
    "reject_to_supplier_draft",
}

_TITLES = {
    "approve_for_payment_prep": "Prepare for payment (approver decides)",
    "route_to_purchasing": "Route to purchasing",
    "request_credit_note": "Request a credit note from the supplier",
    "hold_for_review": "Hold for finance approver review",
    "reject_to_supplier_draft": "Draft returning the document to the supplier",
}

_DESCRIPTIONS = {
    "approve_for_payment_prep": (
        "The three-way match is clean; the packet recommends the approver "
        "prepare this invoice for payment. The approver decides — this "
        "step approves and pays nothing."
    ),
    "route_to_purchasing": (
        "Purchasing owns the order side of this exception (missing order, "
        "price dispute, delivery question) and resolves it with the supplier."
    ),
    "request_credit_note": (
        "Ask the supplier for a credit note covering the variance or the "
        "quantity not received, so the payable amount matches what was "
        "ordered and received."
    ),
    "hold_for_review": (
        "Hold the case for a finance approver: a suspected duplicate, a "
        "fraud indicator, or an exception that needs human judgement first."
    ),
    "reject_to_supplier_draft": (
        "The document cannot be processed as an invoice (wrong document "
        "type or unreadable); a return note is drafted for approver review."
    ),
}

# How well each kind fits each triage outcome (0..1).
_FIT: dict[str, dict[str, float]] = {
    "clean_match": {
        "approve_for_payment_prep": 1.0,
        "route_to_purchasing": 0.15,
        "request_credit_note": 0.05,
        "hold_for_review": 0.35,
        "reject_to_supplier_draft": 0.0,
    },
    "price_variance": {
        "approve_for_payment_prep": 0.1,
        "route_to_purchasing": 0.8,
        "request_credit_note": 1.0,
        "hold_for_review": 0.55,
        "reject_to_supplier_draft": 0.05,
    },
    "quantity_variance": {
        "approve_for_payment_prep": 0.05,
        "route_to_purchasing": 0.7,
        "request_credit_note": 0.9,
        "hold_for_review": 0.6,
        "reject_to_supplier_draft": 0.05,
    },
    "missing_po": {
        "approve_for_payment_prep": 0.0,
        "route_to_purchasing": 1.0,
        "request_credit_note": 0.1,
        "hold_for_review": 0.5,
        "reject_to_supplier_draft": 0.2,
    },
    "missing_receipt": {
        "approve_for_payment_prep": 0.0,
        "route_to_purchasing": 0.6,
        "request_credit_note": 0.1,
        "hold_for_review": 0.9,
        "reject_to_supplier_draft": 0.1,
    },
    "duplicate_suspected": {
        "approve_for_payment_prep": 0.0,
        "route_to_purchasing": 0.3,
        "request_credit_note": 0.1,
        "hold_for_review": 1.0,
        "reject_to_supplier_draft": 0.3,
    },
    "unreadable": {
        "approve_for_payment_prep": 0.0,
        "route_to_purchasing": 0.2,
        "request_credit_note": 0.0,
        "hold_for_review": 0.5,
        "reject_to_supplier_draft": 1.0,
    },
    "not_an_invoice": {
        "approve_for_payment_prep": 0.0,
        "route_to_purchasing": 0.1,
        "request_credit_note": 0.0,
        "hold_for_review": 0.4,
        "reject_to_supplier_draft": 1.0,
    },
}

# How well each kind fits the computed match status (0..1).
_MATCH_FIT: dict[str, dict[str, float]] = {
    "full_match": {
        "approve_for_payment_prep": 1.0,
        "route_to_purchasing": 0.2,
        "request_credit_note": 0.1,
        "hold_for_review": 0.4,
        "reject_to_supplier_draft": 0.0,
    },
    "partial_match": {
        "approve_for_payment_prep": 0.2,
        "route_to_purchasing": 0.8,
        "request_credit_note": 0.8,
        "hold_for_review": 0.7,
        "reject_to_supplier_draft": 0.1,
    },
    "no_match": {
        "approve_for_payment_prep": 0.0,
        "route_to_purchasing": 0.6,
        "request_credit_note": 0.3,
        "hold_for_review": 0.8,
        "reject_to_supplier_draft": 0.3,
    },
    "no_po": {
        "approve_for_payment_prep": 0.0,
        "route_to_purchasing": 1.0,
        "request_credit_note": 0.1,
        "hold_for_review": 0.5,
        "reject_to_supplier_draft": 0.2,
    },
    "no_receipt": {
        "approve_for_payment_prep": 0.0,
        "route_to_purchasing": 0.6,
        "request_credit_note": 0.1,
        "hold_for_review": 0.9,
        "reject_to_supplier_draft": 0.1,
    },
}

_SEVERITY_PENALTY = {"low": 0.0, "moderate": 6.0, "high": 16.0}


def template_proposals(triage: dict, facts: dict) -> list[dict]:
    """The deterministic proposal set for a triage (all five kinds are
    always proposed — scoring, not presence, expresses the fit)."""
    return [
        {"kind": kind, "title": _TITLES[kind], "description": _DESCRIPTIONS[kind]}
        for kind in (
            "approve_for_payment_prep",
            "route_to_purchasing",
            "request_credit_note",
            "hold_for_review",
            "reject_to_supplier_draft",
        )
    ]


def score_option(kind: str, triage: dict, facts: dict) -> float:
    """The deterministic score for one option kind (see module docstring)."""
    outcome = triage.get("outcome", "missing_po")
    match_status = facts.get("match_status", "no_po")
    fit = _FIT.get(outcome, _FIT["missing_po"]).get(kind, 0.0)
    match_fit = _MATCH_FIT.get(match_status, _MATCH_FIT["no_po"]).get(kind, 0.0)
    if facts.get("fraud_blocked"):
        # The fraud hold is not negotiable by arithmetic.
        if kind == "approve_for_payment_prep":
            fit, match_fit = 0.0, 0.0
        elif kind == "hold_for_review":
            fit, match_fit = 1.0, 1.0
    severity_penalty = 0.0
    amount_penalty = 0.0
    if kind == "approve_for_payment_prep":
        severity_penalty = _SEVERITY_PENALTY.get(triage.get("severity", "moderate"), 6.0)
        total = facts.get("total_amount") or 0.0
        if total > HIGH_VALUE_THRESHOLD:
            amount_penalty = 10.0
    return round(60.0 * fit + 40.0 * match_fit - severity_penalty - amount_penalty, 2)


def score_options(proposals: list[dict], triage: dict, facts: dict) -> list[RecoveryOption]:
    """Validate kinds, score every proposal, mark the argmax recommended."""
    scored: list[RecoveryOption] = []
    for proposal in proposals:
        kind = str(proposal.get("kind", "")).strip().lower()
        if kind not in OPTION_KINDS:
            continue
        scored.append(
            RecoveryOption(
                kind=kind,
                title=str(proposal.get("title") or _TITLES[kind])[:120],
                description=str(proposal.get("description") or _DESCRIPTIONS[kind])[:300],
                score=score_option(kind, triage, facts),
            )
        )
    if not scored:
        scored = score_options(template_proposals(triage, facts), triage, facts)
    best = max(scored, key=lambda o: o.score)
    for option in scored:
        option.recommended = option is best
    scored.sort(key=lambda o: o.score, reverse=True)
    return scored


def build_options(
    *,
    invoice,
    triage: dict,
    facts: dict,
    diagnosis_summary: str,
    backend,
) -> tuple[list[RecoveryOption], str]:
    """Propose + score the next-step options for a triage.

    In provider mode the LLM proposes the option wording (kinds
    validated, scores computed here); otherwise the template proposal
    set runs. Returns (options, note) — the note records a degraded
    LLM path.
    """
    propose = getattr(backend, "propose_options", None)
    if propose is None:
        return score_options(template_proposals(triage, facts), triage, facts), ""
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
        diagnosis_summary=diagnosis_summary,
        policy_details=[],
    )
    note = ""
    try:
        proposals = propose(context)
    except Exception as exc:  # options never fail the run
        proposals = None
        note = f"LLM option proposals failed ({exc}) — template proposals used"
    if not proposals:
        proposals = template_proposals(triage, facts)
        if not note:
            note = "LLM option proposals were unusable — template proposals used"
    return score_options(proposals, triage, facts), note
