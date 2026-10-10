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
- The listings are also *graded*
  (:func:`documents_upfront_effectiveness`): information
  waits and clarification loops per case, before vs after the
  listing moment. A listing the grade reads **no_effect** —
  measured, and neither rate moved — drafts its own release
  on the grade's evidence: the loop the collect opened is
  closed by measurement, not by the pattern merely pausing.
  An approval of such a release is guarded in the grade's own
  terms (the service re-grades; a listing that has started
  working since the draft is not released on stale evidence).

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


def listing_moments(proposals: list) -> dict[str, str]:
    """Per listed supplier, the stamp their current listing
    took effect — the approved collect's own application
    stamp (``applied_at``, falling back to the decision and
    draft stamps). A later approved collect re-lists from its
    own moment; an approved release clears the entry. The
    before/after line the effectiveness grading splits on."""
    moments: dict[str, str] = {}
    for proposal in _approved_in_order(proposals):
        if proposal.kind == "collect_documents_upfront":
            moments[proposal.supplier_name] = (
                proposal.applied_at
                or proposal.decided_at
                or proposal.proposed_at
                or ""
            )
        elif proposal.kind == "release_documents_upfront":
            moments.pop(proposal.supplier_name, None)
    return moments


def documents_upfront_effectiveness(records: list, proposals: list) -> dict:
    """Grade the documents-up-front listings: did the
    checklist change move the pattern it was bought for?

    Round 19 made the listing *visible* and reversible on the
    re-analysis pattern alone. A listing is an intervention,
    and an intervention nobody grades is configuration
    theatre — so each listed supplier is read before vs after
    their listing moment (the approved collect's stamp, from
    :func:`listing_moments`) on the two measures the checklist
    exists to move:

    - **information waits raised** — cases that still blocked
      on an information request (the request's own stamp; a
      request stored before stamps reads by its case's date);
    - **clarification loops** — re-analyses run (the chain
      entries' stamps; unstamped entries read by their case).

    Cases are split by their invoice's submission date, and
    both measures are read *per case* in each period — a
    supplier with ten quiet cases after listing and two waits
    is not the same reading as one with two cases and two
    waits. The verdict:

    - ``working`` — both rates fell (or one fell and the
      other held at zero movement): the documents now arrive
      with the onboarding and the chasing eased.
    - ``no_effect`` — either rate is flat or worse: measured,
      and the listing changed nothing it was bought to change.
    - ``unmeasured`` — no cases since listing, or no stamped
      pre-listing pattern to compare against: the grade says
      so rather than dressing silence as success.

    Pure over the records and the proposal ledger."""
    from .matching import _parse_date

    listed = documents_upfront_suppliers(proposals)
    moments = listing_moments(proposals)
    rows: dict[str, dict] = {}
    for supplier in sorted(listed):
        listed_at = moments.get(supplier, "")
        listing_date = _parse_date(listed_at)
        row = {
            "supplier_name": supplier,
            "listed_at": listed_at,
            "cases_before": 0,
            "cases_after": 0,
            "waits_before": 0,
            "waits_after": 0,
            "loops_before": 0,
            "loops_after": 0,
            "unchanged_before": 0,
            "unchanged_after": 0,
        }
        rows[supplier] = row
        if listing_date is None:
            continue
        for record in records:
            result = record.result
            if (result.supplier_name or "") != supplier:
                continue
            invoice = record.invoice or {}
            case_date = _parse_date(
                str(invoice.get("submitted_at", ""))
            ) or _parse_date(str(invoice.get("invoice_date", "")))
            if case_date is None:
                continue
            after = case_date >= listing_date
            row["cases_after" if after else "cases_before"] += 1
            request = result.information_request
            if request is not None:
                # A raised wait counts whether or not it has
                # ended — the request stays on the record after
                # a reply, and the grade measures the chasing
                # the checklist was meant to prevent.
                wait_date = _parse_date(request.requested_at) or case_date
                row[
                    "waits_after" if wait_date >= listing_date
                    else "waits_before"
                ] += 1
            for entry in result.reanalysis_history:
                entry_date = _parse_date(entry.at) or case_date
                side = "after" if entry_date >= listing_date else "before"
                row[f"loops_{side}"] += 1
                if entry.previous_outcome == entry.new_outcome:
                    row[f"unchanged_{side}"] += 1

    def _rate(events: int, cases: int) -> float | None:
        return round(events / cases, 4) if cases else None

    graded_rows: list[dict] = []
    counts = {"working": 0, "no_effect": 0, "unmeasured": 0}
    for supplier in sorted(rows):
        row = rows[supplier]
        row["waits_per_case_before"] = _rate(
            row["waits_before"], row["cases_before"]
        )
        row["waits_per_case_after"] = _rate(
            row["waits_after"], row["cases_after"]
        )
        row["loops_per_case_before"] = _rate(
            row["loops_before"], row["cases_before"]
        )
        row["loops_per_case_after"] = _rate(
            row["loops_after"], row["cases_after"]
        )
        if row["cases_after"] == 0 or (
            row["waits_before"] + row["loops_before"]
        ) == 0:
            verdict = "unmeasured"
        else:
            waits_down = (
                row["waits_per_case_after"] < row["waits_per_case_before"]
            )
            loops_down = (
                row["loops_per_case_after"] < row["loops_per_case_before"]
            )
            waits_held = (
                row["waits_per_case_after"] == row["waits_per_case_before"]
            )
            loops_held = (
                row["loops_per_case_after"] == row["loops_per_case_before"]
            )
            verdict = (
                "working"
                if (waits_down or loops_down)
                and (waits_down or waits_held)
                and (loops_down or loops_held)
                else "no_effect"
            )
        row["verdict"] = verdict
        counts[verdict] += 1
        graded_rows.append(row)
    return {
        "suppliers": graded_rows,
        "listed": len(graded_rows),
        **counts,
    }


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


def proposals_from_analytics(
    analytics: dict, proposals: list, effectiveness: dict | None = None
) -> list[dict]:
    """The proposal spec each analytics reading implies, in
    supplier order.

    A spec is a plain dict — ``supplier_name``, ``kind``,
    ``basis`` (the analytics' account, composed from its own
    counts) and ``evidence`` (the readings the draft rests on,
    for the decider to compare). At most one spec per supplier:
    an unlisted supplier repeatedly not unblocked drafts its
    collect; a listed supplier drafts its release on either of
    two readings — the effectiveness grade's (``effectiveness``,
    supplier → grade row from
    :func:`documents_upfront_effectiveness`: the listing was
    measured and moved nothing, so the governed question is
    whether it stands at all) or the analytics' own (every
    re-analysis since listing moved its case — the checklist
    did its work). The grade is consulted first: a listing
    that changed nothing is the sharper question. Every other
    reading drafts nothing — a supplier being unblocked needs
    no ruling, and a listed one still landing unchanged needs
    no second question. Pure over the inputs; the service owns
    ids, actors, and storage.
    """
    grades = effectiveness or {}
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
        grade = grades.get(supplier) or {}
        if grade.get("verdict") == "no_effect":
            listed_at = grade.get("listed_at") or "a governed approval"
            basis = (
                f"{supplier} stands on the documents-up-front "
                f"list since {listed_at}, and the listing has "
                "now been graded "
                "against the pattern it was bought to move — and "
                "it moved nothing. Before listing: "
                f"{grade.get('waits_per_case_before')} information "
                f"waits and {grade.get('loops_per_case_before')} "
                "clarification loops per case; since listing: "
                f"{grade.get('waits_per_case_after')} waits and "
                f"{grade.get('loops_per_case_after')} loops per "
                f"case across {grade.get('cases_after')} case(s). "
                "A checklist change that changes nothing is "
                "configuration theatre: it costs every onboarding "
                "and buys no settling documents. The proposal "
                "takes the supplier off the list on that evidence; "
                "rejecting it keeps the listing standing, with "
                "the grade on the record."
            )
            specs.append(
                {
                    "supplier_name": supplier,
                    "kind": "release_documents_upfront",
                    "basis": basis,
                    "evidence": {
                        "drafted_from": "effectiveness_grade",
                        "verdict": "no_effect",
                        "listed_at": grade.get("listed_at", ""),
                        "cases_after": grade.get("cases_after", 0),
                        "waits_per_case_before": grade.get(
                            "waits_per_case_before"
                        ),
                        "waits_per_case_after": grade.get(
                            "waits_per_case_after"
                        ),
                        "loops_per_case_before": grade.get(
                            "loops_per_case_before"
                        ),
                        "loops_per_case_after": grade.get(
                            "loops_per_case_after"
                        ),
                        "reanalyses": reanalyses,
                        "unchanged": unchanged,
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
