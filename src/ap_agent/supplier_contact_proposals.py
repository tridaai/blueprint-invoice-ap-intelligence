"""Suppliers re-analysis cannot unblock: the governed path
from the pattern to an onboarding change.

The re-analysis analytics (``clarify.reanalysis_analytics``)
name the suppliers whose clarification replies keep failing
to move their cases: the record is re-read, the new evidence
is weighed, and the outcome lands exactly where it was —
because the document that would settle the case was never
collected in the first place. Chasing it mid-dispute, one
clarification round at a time, is the process failing
politely, at full price, forever.

This module is the drafting half of the governed path (the
service owns the workflow: draft → second-operator decision →
the decision applies by derivation):

- A supplier whose re-analyses repeatedly land **unchanged**
  (``MIN_UNCHANGED_FOR_COLLECT`` or more) and who is not
  already listed drafts a **collect_documents_upfront**
  proposal: approval puts the supplier on the documents-up-front
  list — their onboarding document checklist gains the missing
  documents, collected before the first invoice instead of
  chased after the tenth. The queue's information waits for a
  listed supplier read ``documents_upfront``: the pattern,
  named where it happens.
- A listed supplier whose record since listing shows every
  re-analysis moving its case — re-analyses up, none landing
  unchanged — drafts a **release_documents_upfront** proposal:
  the checklist did its work, and approval takes the supplier
  off the list. A supplier who simply went quiet drafts
  nothing: no new cases proves nothing about the checklist.

Who stands listed is derived from the proposal ledger itself
(:func:`documents_upfront_suppliers`): the latest APPROVED
decision per supplier rules — an approved collect lists, an
approved release unlists. Pending and rejected proposals move
nothing. The ledger is therefore the state and its own audit
trail, and there is no second store for the two to disagree
in.

Request ids are sequential and readable (``SCP-0001`` …),
minted from the ids already stored, like the lag-segment ids.
"""

from __future__ import annotations

import re

_ID_SUFFIX = re.compile(r"(\d+)$")

#: The proposal kinds this module drafts.
PROPOSAL_KINDS = (
    "collect_documents_upfront",
    "release_documents_upfront",
)

#: How many unchanged re-analyses make a pattern worth a
#: governed question. One unchanged round is a supplier having
#: a bad week; two is the checklist's shape showing.
MIN_UNCHANGED_FOR_COLLECT = 2


def next_supplier_proposal_id(existing_ids: list[str]) -> str:
    """The next sequential proposal id: ``SCP-0001`` …"""
    highest = 0
    for proposal_id in existing_ids:
        match = _ID_SUFFIX.search(proposal_id or "")
        if match:
            highest = max(highest, int(match.group(1)))
    return f"SCP-{highest + 1:04d}"


def _approved_in_order(proposals: list) -> list:
    return sorted(
        (p for p in proposals if p.status == "approved"),
        key=lambda p: p.proposal_id,
    )


def documents_upfront_suppliers(proposals: list) -> set[str]:
    """The suppliers standing on the documents-up-front list,
    derived from the proposal ledger: the latest approved
    decision per supplier rules — an approved collect lists,
    an approved release unlists. Pending and rejected proposals
    are questions and refusals, not state."""
    listed: set[str] = set()
    for proposal in _approved_in_order(proposals):
        if proposal.kind == "collect_documents_upfront":
            listed.add(proposal.supplier_name)
        elif proposal.kind == "release_documents_upfront":
            listed.discard(proposal.supplier_name)
    return listed


def listing_evidence(proposals: list) -> dict[str, dict]:
    """Per listed supplier, the evidence of the approved
    collect proposal that currently lists them — the readings
    the listing rests on, which a release is measured against
    (re-analyses and unchanged counts are cumulative over the
    supplier's record, so "since listing" is the difference
    from these readings). A later approved collect re-lists on
    its own evidence; an approved release clears the entry."""
    evidence: dict[str, dict] = {}
    for proposal in _approved_in_order(proposals):
        if proposal.kind == "collect_documents_upfront":
            evidence[proposal.supplier_name] = dict(proposal.evidence)
        elif proposal.kind == "release_documents_upfront":
            evidence.pop(proposal.supplier_name, None)
    return evidence


def proposals_from_analytics(analytics: dict, proposals: list) -> list[dict]:
    """The proposal spec each analytics reading implies, in
    supplier order.

    A spec is a plain dict — ``supplier_name``, ``kind``,
    ``basis`` (the analytics' account, composed from its own
    counts) and ``evidence`` (the readings the draft rests on,
    for the decider to compare). At most one spec per supplier:
    an unlisted supplier repeatedly not unblocked drafts its
    collect; a listed supplier whose every re-analysis since
    listing moved its case drafts its release; every other
    reading drafts nothing — a supplier being unblocked needs
    no ruling, and a listed one still landing unchanged needs
    no second question. Pure over the inputs; the service owns
    ids, actors, and storage.
    """
    rows = {
        row.get("supplier_name"): row
        for row in (analytics or {}).get("suppliers_not_unblocked") or []
        if row.get("supplier_name")
    }
    listed = documents_upfront_suppliers(proposals)
    listings = listing_evidence(proposals)
    specs: list[dict] = []
    for supplier in sorted(set(rows) | listed):
        row = rows.get(supplier) or {}
        reanalyses = int(row.get("reanalyses") or 0)
        unchanged = int(row.get("unchanged") or 0)
        if supplier not in listed:
            if unchanged < MIN_UNCHANGED_FOR_COLLECT:
                continue  # one bad round is a week, not a pattern
            basis = (
                f"{supplier} is re-analysed and not unblocked, "
                f"again: {reanalyses} re-analyses across its "
                f"record, {unchanged} of them landing with the "
                "outcome exactly where it was — the replies "
                "arrive, the evidence is weighed, and the case "
                "does not move, because the document that would "
                "settle it was never collected. The proposal "
                "puts the supplier on the documents-up-front "
                "list: their onboarding checklist gains those "
                "documents, collected before the first invoice "
                "instead of chased through clarification rounds "
                "after the tenth. Approval changes no invoice "
                "and no standing case; it changes what the next "
                "onboarding asks for."
            )
            specs.append(
                {
                    "supplier_name": supplier,
                    "kind": "collect_documents_upfront",
                    "basis": basis,
                    "evidence": {
                        "reanalyses": reanalyses,
                        "unchanged": unchanged,
                        "min_unchanged": MIN_UNCHANGED_FOR_COLLECT,
                    },
                }
            )
            continue
        listing = listings.get(supplier) or {}
        listed_reanalyses = int(listing.get("reanalyses") or 0)
        listed_unchanged = int(listing.get("unchanged") or 0)
        if reanalyses <= listed_reanalyses or unchanged != listed_unchanged:
            continue  # quiet proves nothing; still stuck needs no re-ask
        basis = (
            f"{supplier} stands on the documents-up-front list "
            f"since a governed approval (then: {listed_reanalyses} "
            f"re-analyses, {listed_unchanged} landing unchanged). "
            f"Since listing, {reanalyses - listed_reanalyses} "
            "further re-analyses have run and every one moved "
            "its case — none landed unchanged. The checklist "
            "did its work: the documents now arrive with the "
            "onboarding, and the clarification rounds settle. "
            "The proposal takes the supplier off the list; a "
            "fresh pattern of unchanged re-analyses would draft "
            "a fresh collect, on fresh evidence."
        )
        specs.append(
            {
                "supplier_name": supplier,
                "kind": "release_documents_upfront",
                "basis": basis,
                "evidence": {
                    "reanalyses": reanalyses,
                    "unchanged": unchanged,
                    "reanalyses_at_listing": listed_reanalyses,
                    "unchanged_at_listing": listed_unchanged,
                },
            }
        )
    return specs
