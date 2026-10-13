"""Independent review node logic: a second pair of eyes on the drafts.

The reviewer is deliberately NOT the drafter: in provider mode it runs
the reviewer persona (optionally a different model, ``REVIEWER_MODEL``)
over checks the drafter never sees — AP packet completeness above
all — and in every mode a disjoint deterministic checklist guards
packet structure:

- the packet records a diagnosis summary;
- the packet records the three-way match outcome;
- the packet cites at least one retrieved policy;
- the packet names a recommended option;
- the supplier letter states the next step;
- the packet cites exactly the retrieved policies (no strays).

A ``block`` verdict flags the run (``reviewer_blocked``): the draft
still reaches the approver, but the flag is prominent and the case
is never eligible for an auto-triage recommendation. Set
``REVIEWER=off`` to disable the node.
"""

from __future__ import annotations

import re

from .config import env_str, load_dotenv
from .model_backends import DraftContext
from .schemas import Diagnosis, Drafts, RecoveryOption, ReviewResult

_CITATION = re.compile(r"\[(AP-[A-Z]+-\d+)\]")


def reviewer_enabled() -> bool:
    load_dotenv()
    return env_str("REVIEWER", "on").strip().lower() not in {"off", "0", "false", "no"}


def packet_contents(
    *,
    drafts: Drafts,
    diagnosis: Diagnosis | None,
    options: list[RecoveryOption],
    retrieved_ids: list[str],
) -> dict:
    """The packet facts the reviewer judges — computed in code, from
    the structured diagnosis/options, not re-parsed from prose."""
    packet = drafts.ap_packet or ""
    recommended = next((o for o in options if o.recommended), None)
    return {
        "packet_summary_present": bool(diagnosis and diagnosis.summary),
        "packet_match_present": "match" in packet.lower(),
        "packet_citations": sorted(set(_CITATION.findall(packet))),
        "packet_options": [o.kind for o in options],
        "packet_recommended": recommended.title if recommended else None,
        "diagnosis_citations": list(retrieved_ids),
    }


def checklist_review(
    *, drafts: Drafts, contents: dict, retrieved_ids: list[str]
) -> tuple[str, list[str]]:
    """Deterministic packet-structure review. Returns (verdict, findings)."""
    findings: list[str] = []
    blockers: list[str] = []
    if not contents["packet_summary_present"]:
        blockers.append("the AP packet records no diagnosis summary")
    if not contents["packet_match_present"]:
        blockers.append("the AP packet does not record the match outcome")
    if not contents["packet_citations"]:
        blockers.append("the AP packet cites no policy")
    if not contents["packet_recommended"]:
        blockers.append("the packet names no recommended option")
    if "next step" not in (drafts.supplier_letter or "").lower():
        findings.append("the supplier letter does not state the next step")
    for citation in contents["packet_citations"]:
        if citation not in retrieved_ids:
            findings.append(f"packet cites policy {citation}, which was not retrieved")
    for citation in contents["diagnosis_citations"]:
        if citation not in contents["packet_citations"]:
            findings.append(f"diagnosis cites {citation} but the packet does not carry it")
    if blockers:
        return "block", blockers + findings
    if findings:
        return "concerns", findings
    return "pass", ["packet structure complete: match outcome, citations, and a recommendation are all present"]


def review_drafts(
    *,
    invoice,
    triage: dict,
    facts: dict,
    diagnosis: Diagnosis | None,
    options: list[RecoveryOption],
    drafts: Drafts,
    retrieved_ids: list[str],
    backend,
    policies: list[dict] | None = None,
) -> ReviewResult:
    """Review a draft pair: LLM reviewer in provider mode, checklist otherwise.

    ``policies`` are the retrieved policies (id, title, text) — the
    reviewer judges the packet's citations against them."""
    if not reviewer_enabled():
        return ReviewResult(verdict="skipped", source="checklist", enabled=False)
    contents = packet_contents(drafts=drafts, diagnosis=diagnosis, options=options, retrieved_ids=retrieved_ids)
    verdict, findings = checklist_review(
        drafts=drafts, contents=contents, retrieved_ids=retrieved_ids
    )
    review_fn = getattr(backend, "review_draft", None)
    if review_fn is None:
        return ReviewResult(verdict=verdict, findings=findings, source="checklist")
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
        diagnosis_summary=diagnosis.summary if diagnosis else "",
        diagnosis_citations=contents["diagnosis_citations"],
        packet_summary_present=contents["packet_summary_present"],
        packet_citations=contents["packet_citations"],
        packet_options=contents["packet_options"],
        packet_recommended=contents["packet_recommended"],
        subject=drafts.subject,
        supplier_letter=drafts.supplier_letter,
        ap_packet=drafts.ap_packet,
        policy_details=list(policies or []),
    )
    note = ""
    try:
        critique = review_fn(context)
    except Exception as exc:  # review never fails the run
        critique = None
        note = f"LLM review failed ({exc}) — deterministic checklist review used"
    if critique is None:
        return ReviewResult(
            verdict=verdict,
            findings=findings,
            source="checklist",
            note=note or "LLM review reply was unusable — deterministic checklist review used",
        )
    merged = list(dict.fromkeys([*findings, *critique["findings"]]))
    severity = {"pass": 0, "concerns": 1, "block": 2}
    final = max(verdict, critique["verdict"], key=lambda v: severity[v])
    if final == "block" and verdict != "block":
        merged.append("the independent reviewer blocked this draft (see findings)")
    if critique.get("model"):
        merged.append(f"reviewer model: {critique['model']}")
    return ReviewResult(verdict=final, findings=merged, source="llm")
