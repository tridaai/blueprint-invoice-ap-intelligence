"""Drift-demoted lag segments: the governed path from the
watch to a standing decision.

The drift watch (:func:`paymentruns.lag_grade_drift_watch`)
reads the accuracy ledger in two windows and names a lag
statistic **degrading** when its recent recall falls away from
its earlier record. The forecast's selection demotes such a
statistic automatically, run by run — sound, but the demotion
is nobody's decision: it lifts the moment the recent window
turns over, and it returns the moment the window slides back,
with no human ever ruling on whether the segment's statistic
should stand in the candidate set at all.

This module is the drafting half of the governed path (the
service owns the workflow: draft → second-operator decision →
the decision applies by derivation):

- A statistic the watch names degrading drafts a
  **retire_segment** proposal: approval takes the segment out
  of the forecast's candidate set altogether — the selection
  skips it and names the exclusion in its basis — until a
  re-grade re-admits it. A retirement is not a recomputation:
  nothing the statistic measured is rewritten; it stops being
  a statistic that may project.
- An excluded segment whose recent record has recovered — the
  watch assesses it again and no longer names it degrading —
  drafts a **regrade_segment** proposal: approval re-admits it
  to the candidate set, re-graded on the record as it now
  stands, and the selection's ordinary grading decides whether
  it wins.

Which segments stand excluded is derived from the proposal
ledger itself (:func:`excluded_segments`): the latest APPROVED
decision per segment rules — a retirement excludes, a re-grade
re-admits, pending and rejected proposals move nothing. The
ledger is therefore the state and its own audit trail, and
there is no second store for the two to disagree in.

Request ids are sequential and readable (``LSP-0001`` …),
minted from the ids already stored, like the profile-change
ids.
"""

from __future__ import annotations

import re

_ID_SUFFIX = re.compile(r"(\d+)$")

#: The proposal kinds this module drafts.
PROPOSAL_KINDS = ("retire_segment", "regrade_segment")


def next_lag_proposal_id(existing_ids: list[str]) -> str:
    """The next sequential proposal id: ``LSP-0001`` …"""
    highest = 0
    for proposal_id in existing_ids:
        match = _ID_SUFFIX.search(proposal_id or "")
        if match:
            highest = max(highest, int(match.group(1)))
    return f"LSP-{highest + 1:04d}"


def segment_label(segment: str) -> str:
    """A statistic key in prose: ``currency:EUR`` reads as the
    per-currency (EUR) statistic, ``profile:X`` as the
    per-profile (X) one, ``global`` as the global pool — the
    selection's own vocabulary."""
    if segment == "global":
        return "global"
    if segment.startswith("currency:"):
        return f"per-currency ({segment.split(':', 1)[1]})"
    if segment.startswith("profile:"):
        return f"per-profile ({segment.split(':', 1)[1]})"
    return segment


def excluded_segments(proposals: list) -> set[str]:
    """The segments standing excluded from the forecast's
    candidate set, derived from the proposal ledger: the latest
    approved decision per segment rules — an approved retirement
    excludes, an approved re-grade re-admits. Pending and
    rejected proposals are questions and refusals, not state.
    Proposals are read in id order (the ledger's order)."""
    excluded: set[str] = set()
    for proposal in sorted(proposals, key=lambda p: p.proposal_id):
        if proposal.status != "approved":
            continue
        if proposal.kind == "retire_segment":
            excluded.add(proposal.segment)
        elif proposal.kind == "regrade_segment":
            excluded.discard(proposal.segment)
    return excluded


def _recall(value) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "—"


def proposals_from_watch(
    watch: dict, grades: dict, excluded: set[str]
) -> list[dict]:
    """The proposal spec each watch reading implies, in segment
    order.

    A spec is a plain dict — ``segment``, ``kind``, ``basis``
    (the watch's account, composed from its own recalls and
    member counts) and ``evidence`` (the readings the draft
    rests on, for the decider to compare). At most one spec per
    segment: a degrading segment that stands admitted drafts
    its retirement; an excluded segment the watch no longer
    names degrading drafts its re-grade; every other reading
    drafts nothing — a standing statistic needs no ruling, and
    an excluded one still decaying needs no second question.
    Pure over the inputs; the service owns ids, actors, and
    storage.
    """
    specs: list[dict] = []
    for segment in sorted(watch):
        entry = watch[segment] or {}
        grade = grades.get(segment) or {}
        evidence = {
            "earlier_recall": entry.get("earlier_recall"),
            "recent_recall": entry.get("recent_recall"),
            "overall_recall": entry.get("overall_recall"),
            "earlier_graded_members": entry.get("earlier_graded_members"),
            "recent_graded_members": entry.get("recent_graded_members"),
            "grade_recall": grade.get("recall"),
            "grade_precision": grade.get("precision"),
            "graded_members": grade.get("graded_members"),
        }
        label = segment_label(segment)
        if entry.get("degrading") and segment not in excluded:
            basis = (
                f"The {label} lag statistic is degrading: recent "
                f"recall {_recall(entry.get('recent_recall'))} over "
                f"its latest {entry.get('recent_graded_members')} "
                f"graded members, down from "
                f"{_recall(entry.get('earlier_recall'))} on the "
                f"earlier record ({entry.get('earlier_graded_members')} "
                "graded members). The drift watch already demotes "
                "it inside the forecast's selection, run by run — "
                "but a demotion is automatic and lifts the moment "
                "the recent window turns over. The proposal "
                "retires the statistic from the candidate set "
                "under a governed decision, until a re-grade — "
                "drafted when its recent record recovers — "
                "re-admits it. Approval rewrites nothing the "
                "statistic measured; it removes the statistic "
                "from the set that may project."
            )
            specs.append(
                {
                    "segment": segment,
                    "kind": "retire_segment",
                    "basis": basis,
                    "evidence": evidence,
                }
            )
            continue
        if (
            segment in excluded
            and entry.get("assessed")
            and not entry.get("degrading")
        ):
            basis = (
                f"The {label} lag statistic stands excluded from "
                "the forecast's candidate set under an approved "
                "retirement, and its recent record has recovered: "
                f"recent recall {_recall(entry.get('recent_recall'))} "
                f"over its latest {entry.get('recent_graded_members')} "
                f"graded members, against "
                f"{_recall(entry.get('earlier_recall'))} on the "
                "earlier record — the watch assesses it again and "
                "no longer names it degrading. The proposal "
                "re-admits the statistic to the candidate set, "
                "re-graded on the record as it now stands; the "
                "selection's ordinary grading then decides "
                "whether it wins, run by run."
            )
            specs.append(
                {
                    "segment": segment,
                    "kind": "regrade_segment",
                    "basis": basis,
                    "evidence": evidence,
                }
            )
    return specs
