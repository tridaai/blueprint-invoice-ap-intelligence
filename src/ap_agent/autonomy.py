"""Autonomy recommendation: should cases like this one ever skip the gate?

After every run the pipeline prints a recommendation: whether this
run *would have been* eligible for auto-triage (no human gate) under
a conservative policy — and the reasons either way. It is a
recommendation only: nothing in this blueprint acts on it. The human
gate is unconditional; this exists so an adopting team can see, case
by case, what a future autonomy policy would have to be comfortable
with before they ever consider one.

Eligibility is deliberately narrow — ALL of:

- triage outcome ``clean_match``;
- severity ``low``;
- the invoice total is at or below the approval threshold
  (AP-APPR-01: 10,000);
- the three-way match is a full match;
- guardrails passed on the first draft (no repair was needed);
- the rules/LLM cross-check did not disagree (when it ran);
- the independent reviewer did not block;
- NO fraud screen flags (a bank-change pattern anywhere near the
  case disqualifies it outright, whatever the match says).
"""

from __future__ import annotations

from .schemas import (
    AutonomyRecommendation,
    CrossCheck,
    MatchFacts,
    ReviewResult,
    Triage,
    Validation,
)
from .triage import HIGH_VALUE_THRESHOLD


def recommend_autonomy(
    *,
    triage: Triage,
    facts: MatchFacts,
    validation: Validation,
    repair_attempted: bool,
    cross_check: CrossCheck | None,
    review: ReviewResult | None,
    reviewer_blocked: bool,
    fraud_flags: list[str] | None = None,
    total_amount: float | None = None,
) -> AutonomyRecommendation:
    reasons: list[str] = []
    blockers: list[str] = []

    if triage.outcome == "clean_match":
        reasons.append("triage outcome is clean_match")
    else:
        blockers.append(f"triage outcome is {triage.outcome}, not clean_match")

    if triage.severity == "low":
        reasons.append("severity is low")
    else:
        blockers.append(f"severity is {triage.severity}")

    if facts.match_status == "full_match":
        reasons.append("three-way match is a full match")
    else:
        blockers.append(f"match status is {facts.match_status}")

    if total_amount is not None and total_amount <= HIGH_VALUE_THRESHOLD:
        reasons.append(
            f"invoice total {total_amount} is at or below the approval "
            f"threshold {HIGH_VALUE_THRESHOLD} [AP-APPR-01]"
        )
    elif total_amount is not None:
        blockers.append(
            f"invoice total {total_amount} is above the approval "
            f"threshold {HIGH_VALUE_THRESHOLD} [AP-APPR-01]"
        )

    if fraud_flags:
        blockers.append(
            "fraud screen flagged the documents (bank-detail change "
            "pattern) — auto-triage is never eligible on a flagged case"
        )
    else:
        reasons.append("fraud screen is clear")

    if validation.passed:
        reasons.append("guardrails passed")
    else:
        blockers.append("guardrails failed")

    if repair_attempted:
        blockers.append("the draft needed a repair pass")
    else:
        reasons.append("no repair pass was needed")

    if cross_check is not None and cross_check.resolution not in ("agree", "rules_only"):
        blockers.append(
            f"the rules/LLM cross-check did not agree ({cross_check.resolution})"
        )
    elif cross_check is not None:
        reasons.append("the rules/LLM cross-check raised no disagreement")

    if reviewer_blocked:
        blockers.append("the independent reviewer blocked the draft")
    elif review is not None and review.enabled:
        reasons.append("the independent reviewer did not block")

    return AutonomyRecommendation(
        eligible_for_auto_triage=not blockers,
        reasons=reasons + [f"not eligible: {b}" for b in blockers],
    )
