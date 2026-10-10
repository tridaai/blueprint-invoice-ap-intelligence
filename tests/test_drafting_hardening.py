"""Drafting hardening — the regressions from the 2026-10-10 live
Nemotron run.

The pipeline ran live on NVIDIA Nemotron-3-Super and the safety
spine caught two drafting flaws in the model's output:

(a) the AP packet carried NONE of the policies the diagnosis
    cited — the independent reviewer blocked the draft; and
(b) the supplier letter described the next step in prose but
    never carried the required "Next step:" element, failed the
    ``states_next_step`` guardrail, and the bounded repair —
    told only that the drafts "never state the next step" —
    could not fix what it could not see.

The fixes are compositional and instructional, never weaker
checks: the packet gains the diagnosis's policies
deterministically at draft time (``graph._carry_policy_citations``,
from the retrieved set the diagnosis rests on), the draft
prompts name the required next-step element explicitly, and the
repair feedback names the failed checks AND the fix each one
requires. This stub backend reproduces both flaws on its first
draft and answers the repair with the template drafts — the
pipeline must land at a passing validation with the packet
carrying every retrieved policy.
"""

from __future__ import annotations

import re

import pytest

from ap_agent.graph import _carry_policy_citations
from ap_agent.guardrails import CHECK_REPAIR_DIRECTIVES, repair_directive
from ap_agent.model_backends import MockModelBackend
from ap_agent.prompts import DRAFT_SYSTEM_PROMPT, DRAFT_USER_TEMPLATE
from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

_CITATION = re.compile(r"\[(AP-[A-Z]+-\d+)\]")

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


class _LiveFlawBackend(MockModelBackend):
    """First draft = the live flaws; repair draft = the fix.

    The packet keeps its prose but loses every [AP-...] bracket
    (the citations become plain text the reviewer cannot see),
    and the letter's "Next step:" paragraph is rewritten as
    flowing prose that describes the step without the element.
    A call carrying ``repair_feedback`` records it and returns
    the unaltered template drafts.
    """

    name = "live-flaw-stub"

    def __init__(self) -> None:
        super().__init__()
        self.feedbacks: list[str] = []
        self.first_packet: str = ""

    def draft_documents(self, context):
        subject, letter, packet = super().draft_documents(context)
        feedback = context.get("repair_feedback")
        if feedback:
            self.feedbacks.append(feedback)
            return subject, letter, packet
        flawed_packet = _CITATION.sub(r"\1", packet)
        paragraphs = [
            (
                "The packet now goes to a finance approver for their "
                "review; nothing further is needed from you unless we "
                "write again."
            )
            if paragraph.startswith("Next step:")
            else paragraph
            for paragraph in letter.split("\n\n")
        ]
        self.first_packet = flawed_packet
        return subject, "\n\n".join(paragraphs), flawed_packet


@pytest.fixture()
def flaw_service() -> tuple[InvoiceService, _LiveFlawBackend]:
    backend = _LiveFlawBackend()
    service = InvoiceService(
        store=InMemoryStore(), backend=backend, checkpointer=False
    )
    return service, backend


# ---------------------------------------------------------------------------
# The end-to-end regression: both flaws in, passing drafts out
# ---------------------------------------------------------------------------


def test_live_flaws_are_composed_and_repaired_into_passing_drafts(flaw_service):
    service, backend = flaw_service
    result = service.analyze(SAMPLES["INV-1001"])

    retrieved = [p.policy_id for p in result.policies]
    assert retrieved, "the sample must retrieve policies for this test"

    # The stub's raw packet really was citation-free (flaw (a) in).
    assert _CITATION.findall(backend.first_packet) == []

    # Composition carried every retrieved policy into the packet
    # the approver reads — deterministically, at draft time.
    packet_citations = _CITATION.findall(result.drafts.ap_packet)
    assert set(retrieved) <= set(packet_citations)

    # The reviewer saw the composed packet: no citation blocker,
    # no block at all (the letter's missing element is a finding
    # the repair then closes — review runs before validation).
    assert result.review is not None
    assert result.reviewer_blocked is False
    assert not any(
        "cites no policy" in finding or "does not carry it" in finding
        for finding in result.review.findings
    )

    # Flaw (b) failed first, was named to the repair, and the
    # repair landed: exactly one bounded attempt, now passing.
    assert result.original_validation is not None
    failed_first = {
        check.name for check in result.original_validation.checks if not check.passed
    }
    assert failed_first == {"states_next_step"}
    assert result.repair_attempted is True
    assert result.repair_attempts == 1
    assert result.repaired is True
    assert result.validation is not None and result.validation.passed

    # The repair was told WHICH check failed and the exact fix.
    assert len(backend.feedbacks) == 1
    feedback = backend.feedbacks[0]
    assert "states_next_step" in feedback
    assert 'add a line beginning "Next step:"' in feedback


# ---------------------------------------------------------------------------
# Composition: deterministic, idempotent, retrieved-only
# ---------------------------------------------------------------------------


def test_carry_policy_citations_appends_only_the_missing():
    packet = "AP PACKET\nDiagnosis cites [AP-MATCH-01] inline."
    carried = _carry_policy_citations(
        packet, ["AP-MATCH-01", "AP-CURR-01", "AP-APPR-01"]
    )
    assert carried.startswith(packet)
    appended = carried[len(packet):]
    assert "[AP-CURR-01]" in appended and "[AP-APPR-01]" in appended
    assert "[AP-MATCH-01]" not in appended  # already carried


def test_carry_policy_citations_is_a_noop_when_complete():
    packet = "Policy citations: [AP-MATCH-01] [AP-CURR-01]"
    assert _carry_policy_citations(packet, ["AP-MATCH-01", "AP-CURR-01"]) == packet
    assert _carry_policy_citations(packet, []) == packet


def test_carry_policy_citations_handles_an_empty_packet():
    assert _carry_policy_citations("", ["AP-MATCH-01"]) == (
        "Policy citations: [AP-MATCH-01]"
    )


def test_template_drafts_pass_without_repair_and_stay_byte_stable():
    """The offline path is untouched by the hardening: the
    template packet already carries the citations, so composition
    is a no-op and no repair is attempted."""
    service = InvoiceService(
        store=InMemoryStore(), backend=MockModelBackend(), checkpointer=False
    )
    result = service.analyze(SAMPLES["INV-1001"])
    assert result.validation is not None and result.validation.passed
    assert result.repair_attempted is False
    assert result.drafts.ap_packet.count("Policy citations:") == 1


# ---------------------------------------------------------------------------
# Prompts + directives: the required element is named, not implied
# ---------------------------------------------------------------------------


def test_draft_prompts_name_the_next_step_element_explicitly():
    assert 'a line that begins "Next step:"' in DRAFT_SYSTEM_PROMPT
    assert 'line beginning "Next step:"' in DRAFT_USER_TEMPLATE


def test_every_guardrail_check_has_a_repair_directive():
    from ap_agent.guardrails import validate_drafts
    from ap_agent.samples import sample_invoice_models as models
    from ap_agent.schemas import Drafts

    invoice = SAMPLES["INV-1001"]
    drafts = Drafts(
        subject="Invoice INV-1001",
        supplier_letter="Dear supplier,\n\nNext step: review.",
        ap_packet="AP PACKET [AP-MATCH-01]",
    )
    validation = validate_drafts(invoice, drafts, ["AP-MATCH-01"])
    names = {check.name for check in validation.checks}
    assert names <= set(CHECK_REPAIR_DIRECTIVES)
    assert 'beginning "Next step:"' in repair_directive("states_next_step")
    assert repair_directive("no_such_check") == ""
    assert models  # (import guard — samples module loads)
