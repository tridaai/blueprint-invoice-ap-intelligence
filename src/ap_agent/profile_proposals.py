"""Effectiveness-driven profile proposals: the governed path
from a grade to a change.

``handoff_profiles.profile_effectiveness`` grades the routing
profiles and names configuration theatre plainly — but a read
changes nothing, and an operator reading /metrics still has to
carry the finding into a configuration change by hand. This
module is the drafting half of the governed path (the service
owns the workflow: draft → second-operator decision → apply):

- A profile that **governs no exports** is configuration on
  paper only; the change its grade implies is retirement —
  delete the record, so the configuration surface says only
  what is true. The proposal's ``proposed`` is None: approval
  removes, it does not set values.
- A profile whose exports carry the at-import marker **at the
  same rate as ungoverned exports**, with the marker flag still
  off, is routing no safer than nothing — and the one lever the
  profile itself holds over that is ``markers_rejected``: turn
  it on, so marked members are flagged at proposal time while
  onboarding is still cheap. The proposal's ``proposed`` is the
  profile's graded values with that one flag changed.

Two grade families deliberately draft NOTHING. A flagged
supplier who keeps exporting marked is an onboarding failure —
its remedy is the governed master-change path, not profile
configuration. And a breached lead promise is a resolution-path
regression: the dates written broke the promise the profile
made, and no edit to the profile un-breaches them. Both stay
where they belong — in the grade's notes, in front of a human.

Request ids are sequential and readable (``PCP-0001`` …), minted
from the ids already stored, like the master-change ids.
"""

from __future__ import annotations

import re

from .schemas import HandoffProfile

_ID_SUFFIX = re.compile(r"(\d+)$")

#: The proposal kinds this module drafts.
PROPOSAL_KINDS = ("retire", "tighten_markers")


def next_proposal_id(existing_ids: list[str]) -> str:
    """The next sequential proposal id: ``PCP-0001`` …"""
    highest = 0
    for proposal_id in existing_ids:
        match = _ID_SUFFIX.search(proposal_id or "")
        if match:
            highest = max(highest, int(match.group(1)))
    return f"PCP-{highest + 1:04d}"


def profile_values(profile: HandoffProfile) -> dict:
    """A profile's routing values as a flat dict — the snapshot
    a proposal carries for its ``current`` (and, adjusted, its
    ``proposed``), so the decider compares values, not prose."""
    return {
        "scope": profile.scope,
        "supplier_name": profile.supplier_name,
        "currency": profile.currency,
        "format": profile.format,
        "lead_days": profile.lead_days,
        "markers_rejected": profile.markers_rejected,
        "batch_booking": profile.batch_booking,
    }


def proposals_from_grades(
    profiles: list[HandoffProfile], grades: list[dict]
) -> list[dict]:
    """The proposal spec each grade implies, in profile order.

    A spec is a plain dict — ``profile_id``, ``kind``, ``basis``
    (the grade's account, composed from its own notes and
    numbers), ``current`` (the profile's values as graded) and
    ``proposed`` (the values approval would set, or None for a
    retirement). At most one spec per profile: a profile that
    governs nothing has no marker rate to answer for, and a
    governing profile is not retired for a marker rate. Pure
    over the inputs; the service owns ids, actors, and storage.
    """
    by_id = {profile.profile_id: profile for profile in profiles}
    specs: list[dict] = []
    for grade in grades:
        profile = by_id.get(grade.get("profile_id", ""))
        if profile is None:
            continue
        notes = grade.get("notes") or []
        current = profile_values(profile)
        if grade.get("exports_governed", 0) == 0:
            basis = (
                f"{profile.profile_id} governs no exports yet — "
                "configuration on paper only, graded over the "
                "stored payment runs. A profile nothing routes "
                "through is theatre: it reads as routing knowledge "
                "in force while no file has ever followed it. The "
                "proposal retires it; the scope can be profiled "
                "again the day a bank or ERP fact is actually known."
            )
            specs.append(
                {
                    "profile_id": profile.profile_id,
                    "kind": "retire",
                    "basis": basis,
                    "current": current,
                    "proposed": None,
                }
            )
            continue
        marker_note = next(
            (n for n in notes if "same rate as ungoverned" in n), ""
        )
        if marker_note and not profile.markers_rejected:
            proposed = dict(current, markers_rejected=True)
            basis = (
                f"{marker_note} The profile's one lever over the "
                "marker is its rejection stance, and it is off: "
                "with markers_rejected on, a member whose bank "
                "refuses markers is flagged at proposal time, while "
                "onboarding the supplier is still cheap. The "
                "proposal turns the flag on; every other value "
                "stands as graded."
            )
            specs.append(
                {
                    "profile_id": profile.profile_id,
                    "kind": "tighten_markers",
                    "basis": basis,
                    "current": current,
                    "proposed": proposed,
                }
            )
    return specs
