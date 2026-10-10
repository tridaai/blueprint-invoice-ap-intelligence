"""Rules-vs-LLM triage cross-check — the resolution policy.

Both paths triage the same invoice independently:

- the deterministic rule classifier (``triage.py``), and
- in provider mode, the LLM, which is never shown the rule result.

This module resolves the pair into the single triage the rest of the
pipeline uses. The policy is explicit and deliberately narrow (the
same policy as the reference blueprint family):

1. **Agreement** (same outcome): the rule result stands — identical
   conclusion, and the rule path is the auditable one. A severity
   difference is noted but does not change the outcome.
2. **Disagreement, rules strong**: the rule result stands
   (``rules_authoritative``). The disagreement is recorded in state
   and surfaced in the trace and console for the approver.
3. **Disagreement, rules weak, LLM confident**: when the rules' own
   confidence is below ``RULE_WEAK_CONFIDENCE`` AND the LLM
   confidence is at least ``LLM_ADOPT_CONFIDENCE``, the LLM triage
   is adopted (``llm_adopted``) — the case rules are known to miss:
   an exception described in phrasing no rule covers. The adopted
   result is flagged in its signals and rationale so the approver
   can see exactly why the triage is not the rule output.

Everything else — no LLM configured (default mode), a provider
error, an unusable reply — resolves to the rule result alone
(``rules_only``). Adoption never touches computed facts: the
three-way match comes from ``matching.py`` regardless of which
path wins.
"""

from __future__ import annotations

from .schemas import CrossCheck, Triage

RULE_WEAK_CONFIDENCE = 0.85
LLM_ADOPT_CONFIDENCE = 0.85


def resolve_triage(
    rule: Triage, llm: dict | None, backend_name: str = ""
) -> tuple[Triage, CrossCheck]:
    """Apply the resolution policy to a (rule, LLM) triage pair."""
    base = {"rule_outcome": rule.outcome}
    if llm is None:
        return rule, CrossCheck(
            **base,
            resolution="rules_only",
            note="No LLM triage available; the rule result stands alone.",
        )

    llm_fields = {
        "llm_outcome": llm["outcome"],
        "llm_severity": llm["severity"],
        "llm_confidence": llm["confidence"],
    }
    if llm["outcome"] == rule.outcome:
        note = "Both paths triaged this invoice the same way; the rule result stands."
        if llm["severity"] != rule.severity:
            note += (
                f" Severity differs (rules: {rule.severity}, LLM: "
                f"{llm['severity']}) — the rule severity is kept."
            )
        return rule, CrossCheck(**base, **llm_fields, resolution="agree", note=note)

    if rule.confidence < RULE_WEAK_CONFIDENCE and llm["confidence"] >= LLM_ADOPT_CONFIDENCE:
        final = Triage(
            outcome=llm["outcome"],
            severity=llm["severity"],
            confidence=llm["confidence"],
            rationale=(
                f"{llm['rationale']} (Adopted from the LLM cross-check: the "
                f"deterministic rules produced '{rule.outcome}' at confidence "
                f"{rule.confidence}, so this adoption is flagged for the "
                "approver.)"
            ),
            signals=list(rule.signals) + [
                f"llm-adopted: rules gave '{rule.outcome}' at confidence "
                f"{rule.confidence}; the LLM cross-check triaged "
                f"'{llm['outcome']}' at confidence {llm['confidence']} — "
                "flagged for the approver"
            ],
        )
        return final, CrossCheck(
            **base, **llm_fields,
            resolution="llm_adopted",
            note=(
                "Disagreement: the rules were low-confidence and the LLM was "
                "highly confident, so the LLM triage was adopted — flagged "
                "for the approver."
            ),
        )

    return rule, CrossCheck(
        **base, **llm_fields,
        resolution="rules_authoritative",
        note=(
            "Disagreement recorded for the approver; the deterministic "
            "rule result stays authoritative."
        ),
    )
