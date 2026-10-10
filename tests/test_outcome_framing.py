"""Outcome framing: the letter states the outcome as computed.

The 2026-10-10 live judge slice caught a draft claiming the
invoice was "received and matched" while the record showed a
partial match with a quantity variance: every figure traced,
no promise was made, and the framing still overstated the
record. Two sources fed the claim — the template (and the
provider fallback) stamped EVERY subject "received and
matched" regardless of outcome, and nothing policed an outcome
claim in a draft the way ``figures_must_trace`` polices a
figure.

The fix mirrors the figures discipline:

* the subject's claim is COMPOSED from the computed outcome
  (``outcome_subject_suffix``) in the template drafter and the
  provider fallback alike — an unknown outcome defaults to
  "under review", never to "matched"; and
* a new guardrail, ``outcome_claim_must_trace``, fails a draft
  whose outcome claim exceeds the computed outcome: outcome
  claims ("received and matched", "fully matched", "clean
  match") require a clean_match outcome, and the match claim
  ("full match") requires a full_match status — so a duplicate
  (full match, held outcome) may say its match is full but may
  not say it was "received and matched".
"""

from __future__ import annotations

import pytest

from ap_agent.guardrails import validate_drafts
from ap_agent.model_backends import MockModelBackend, outcome_subject_suffix
from ap_agent.prompts import DRAFT_SYSTEM_PROMPT
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import Drafts
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
CITATIONS = ["AP-MATCH-01"]

BASE_LETTER = (
    "Dear Northbeam team,\n\nThank you for invoice NB-4471 (our reference "
    "INV-1003).\n\nNext step: the packet goes to a finance approver for "
    "review.\n\nThis letter records the match outcome. It is not a payment "
    "confirmation, and no payment is made or scheduled by this letter."
)
BASE_PACKET = "AP PACKET — invoice INV-1003\nPolicy citations: [AP-MATCH-01]"


def _validate(subject: str, letter: str, outcome, match_status):
    return validate_drafts(
        SAMPLES["INV-1003"],
        Drafts(subject=subject, supplier_letter=letter, ap_packet=BASE_PACKET),
        CITATIONS,
        outcome=outcome,
        match_status=match_status,
    )


def _check(validation, name="outcome_claim_must_trace"):
    return next(c for c in validation.checks if c.name == name)


# ---------------------------------------------------------------------------
# The guardrail: claims traced against the computed outcome
# ---------------------------------------------------------------------------


def test_received_and_matched_on_a_variance_outcome_fails():
    validation = _validate(
        "Invoice INV-1003 — received and matched",
        BASE_LETTER,
        outcome="quantity_variance",
        match_status="partial_match",
    )
    check = _check(validation)
    assert not check.passed
    assert "received and matched" in check.detail
    assert "quantity_variance" in check.detail
    assert not validation.passed


def test_the_same_claim_on_a_clean_match_passes():
    validation = _validate(
        "Invoice INV-1003 — received and matched",
        BASE_LETTER + "\nThe invoice matches the purchase order.",
        outcome="clean_match",
        match_status="full_match",
    )
    assert _check(validation).passed
    assert validation.passed


def test_a_full_match_claim_on_a_partial_match_fails():
    validation = _validate(
        "Invoice INV-1003 — received",
        BASE_LETTER + "\nThe three-way check is a full match.",
        outcome="quantity_variance",
        match_status="partial_match",
    )
    check = _check(validation)
    assert not check.passed
    assert "full match" in check.detail


def test_a_duplicate_may_state_its_full_match_but_not_a_clean_outcome():
    # The duplicate's match IS full — the case is held for
    # duplication, not mismatch. The match claim traces; the
    # outcome claim ("received and matched") does not.
    honest = _validate(
        "Invoice INV-1003 — received",
        BASE_LETTER + "\nThe three-way match itself is a full match.",
        outcome="duplicate_suspected",
        match_status="full_match",
    )
    assert _check(honest).passed

    overstated = _validate(
        "Invoice INV-1003 — received and matched",
        BASE_LETTER,
        outcome="duplicate_suspected",
        match_status="full_match",
    )
    assert not _check(overstated).passed


def test_a_negated_claim_is_not_a_claim():
    validation = _validate(
        "Invoice INV-1003 — received",
        BASE_LETTER + "\nThe quantities are not fully matched yet.",
        outcome="quantity_variance",
        match_status="partial_match",
    )
    assert _check(validation).passed


def test_no_outcome_supplied_reports_unevaluated_not_silent():
    validation = validate_drafts(
        SAMPLES["INV-1003"],
        Drafts(
            subject="Invoice INV-1003 — received and matched",
            supplier_letter=BASE_LETTER,
            ap_packet=BASE_PACKET,
        ),
        CITATIONS,
    )
    check = _check(validation)
    assert check.passed
    assert "not evaluated" in check.detail


# ---------------------------------------------------------------------------
# Composition: the subject is derived from the computed outcome
# ---------------------------------------------------------------------------


def test_subject_suffix_is_outcome_derived():
    assert outcome_subject_suffix("clean_match") == "received and matched"
    for outcome in (
        "price_variance",
        "quantity_variance",
        "missing_po",
        "missing_receipt",
        "duplicate_suspected",
        "unreadable",
        "not_an_invoice",
    ):
        assert "matched" not in outcome_subject_suffix(outcome)
    # A future outcome never inherits the matched claim by default.
    assert outcome_subject_suffix("some_new_outcome") == "received and under review"


def test_template_subject_for_the_variance_case_states_the_variance():
    service = InvoiceService(store=InMemoryStore(), checkpointer=False)
    result = service.analyze(SAMPLES["INV-1003"])
    assert result.triage.outcome == "quantity_variance"
    assert "received and matched" not in result.drafts.subject
    assert "quantity variance" in result.drafts.subject
    # And the pipeline's own validation — now outcome-aware —
    # passes the composed drafts with no repair.
    assert result.validation is not None and result.validation.passed
    assert _check(result.validation).passed
    assert result.repair_attempted is False


def test_template_subject_for_a_clean_case_keeps_the_matched_claim():
    service = InvoiceService(store=InMemoryStore(), checkpointer=False)
    result = service.analyze(SAMPLES["INV-1001"])
    assert result.triage.outcome == "clean_match"
    assert result.drafts.subject.endswith("received and matched")
    assert result.validation is not None and result.validation.passed


def test_draft_prompt_names_the_outcome_framing_rule():
    assert "Outcome framing" in DRAFT_SYSTEM_PROMPT
    assert "outcome_claim_must_trace" in DRAFT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# The live catch, reproduced with a stub and repaired
# ---------------------------------------------------------------------------


class _OverstatingBackend(MockModelBackend):
    """First draft = the live catch: the subject claims the case
    was 'received and matched' whatever the outcome, and the
    letter opens with the same claim. A repair call returns the
    (now outcome-derived) template drafts."""

    name = "overstating-stub"

    def __init__(self) -> None:
        super().__init__()
        self.feedbacks: list[str] = []

    def draft_documents(self, context):
        subject, letter, packet = super().draft_documents(context)
        if context.get("repair_feedback"):
            self.feedbacks.append(context["repair_feedback"])
            return subject, letter, packet
        flawed_subject = subject.rsplit(":", 1)[0] + ": received and matched"
        flawed_letter = letter.replace(
            "Thank you for invoice",
            "Your invoice is received and matched. Thank you for invoice",
            1,
        )
        return flawed_subject, flawed_letter, packet


@pytest.fixture()
def overstating_service() -> tuple[InvoiceService, _OverstatingBackend]:
    backend = _OverstatingBackend()
    service = InvoiceService(
        store=InMemoryStore(), backend=backend, checkpointer=False
    )
    return service, backend


def test_the_live_catch_is_caught_and_repaired(overstating_service):
    service, backend = overstating_service
    result = service.analyze(SAMPLES["INV-1003"])
    assert result.triage.outcome == "quantity_variance"

    # The first draft failed exactly the new check.
    assert result.original_validation is not None
    failed_first = {
        check.name for check in result.original_validation.checks if not check.passed
    }
    assert failed_first == {"outcome_claim_must_trace"}

    # The repair was told the check and its fix, and landed.
    assert result.repair_attempted is True
    assert result.repair_attempts == 1
    assert result.repaired is True
    assert result.validation is not None and result.validation.passed
    assert len(backend.feedbacks) == 1
    assert "outcome_claim_must_trace" in backend.feedbacks[0]
    assert "state the outcome exactly as computed" in backend.feedbacks[0]

    # The surviving drafts state the case as computed.
    assert "received and matched" not in result.drafts.subject
    assert "received and matched" not in result.drafts.supplier_letter
