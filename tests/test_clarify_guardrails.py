"""The information request is a supplier-facing letter — it is
policed like one.

clarify.py composes the missing-information request in a separate
node from the drafts, and historically nothing validated it: a
provider could word it with a payment promise, an invented figure,
or a "matched" claim about a case that is blocked precisely
because it did not match, and the text went to the approver
unexamined. These tests pin the policing: the same guardrail
checks scoped to this surface (guardrails.
validate_information_request), one repair recompose told the
failed checks and their fixes by name, and a template fallback
that records what the model tried to send.
"""

from __future__ import annotations

import sys

import pytest

from ap_agent.clarify import build_information_request, template_message
from ap_agent.graph import run_invoice
from ap_agent.guardrails import validate_information_request
from ap_agent.schemas import InvoiceInput, MatchFacts


def _invoice() -> InvoiceInput:
    return InvoiceInput(
        invoice_id="INV-2001",
        supplier_name="Northbeam Office Supplies",
        invoice_number="NB-4471",
        total_amount=1180.0,
        tax_amount=180.0,
    )


def _facts() -> MatchFacts:
    return MatchFacts(has_po=False, has_receipt=False, match_status="no_po")


def _build(backend) -> object:
    return build_information_request(
        invoice=_invoice(),
        triage_outcome="missing_po",
        facts=_facts(),
        extractions=None,
        backend=backend,
    )


class _NoCompose:
    """A backend without the compose capability (the mock path)."""


class _StubCompose:
    """Answers compose calls from a queue and records the contexts
    it was handed (so a test can read the repair feedback)."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.contexts: list[dict] = []

    def compose_information_request(self, context) -> str | None:
        self.contexts.append(dict(context))
        return self._replies.pop(0) if self._replies else None


_BAD = (
    "Good news: invoice INV-2001 has been fully matched, and "
    "payment will be made within 3 days. The total of $9,999.00 "
    "is confirmed."
)

_GOOD = (
    "Re: invoice INV-2001 — Northbeam Office Supplies invoice NB-4471\n\n"
    "This invoice is on hold in accounts payable. Before it can be "
    "processed, we still need:\n"
    "- the purchase order for this invoice (or its PO number)\n\n"
    "Next step: send the item above — once we have it, the invoice "
    "goes back through the three-way match."
)


def test_template_message_passes_its_own_guardrails():
    request = _build(_NoCompose())
    assert request is not None and request.needed
    assert request.draft_source == "template"
    assert request.rejected_draft_checks == []
    assert request.validation is not None and request.validation.passed
    assert "Next step:" in request.message


def test_noncompliant_provider_letter_is_caught_and_falls_back():
    backend = _StubCompose([_BAD, _BAD])
    request = _build(backend)
    # The stored message is the template, never the model's text.
    assert request.message == template_message(
        _invoice(), request.missing_items
    )
    assert request.draft_source == "template_fallback"
    assert request.validation is not None and request.validation.passed
    failed = set(request.rejected_draft_checks)
    assert "outcome_claim_must_trace" in failed
    assert "no_payment_language" in failed
    assert "no_prohibited_promises" in failed
    assert "figures_must_trace" in failed
    assert "states_next_step" in failed
    # The repair attempt was told the failed checks by name, with
    # the surface's own fix for the next-step check.
    feedback = backend.contexts[1].get("repair_feedback", "")
    assert "outcome_claim_must_trace" in feedback
    assert 'add a line beginning "Next step:" to the message' in feedback


def test_repair_recompose_can_save_a_provider_draft():
    backend = _StubCompose([_BAD, _GOOD])
    request = _build(backend)
    assert request.draft_source == "provider_repaired"
    assert request.message == _GOOD
    assert request.validation is not None and request.validation.passed
    # The first draft's failures stay on the record.
    assert "outcome_claim_must_trace" in request.rejected_draft_checks


def test_compliant_provider_draft_is_stored_as_is():
    backend = _StubCompose([_GOOD])
    request = _build(backend)
    assert request.draft_source == "provider"
    assert request.message == _GOOD
    assert request.rejected_draft_checks == []
    assert len(backend.contexts) == 1


def test_compose_failure_keeps_the_template():
    class _Exploding:
        def compose_information_request(self, context):
            raise RuntimeError("provider down")

    request = _build(_Exploding())
    assert request.draft_source == "template"
    assert request.validation is not None and request.validation.passed


# -- the validation itself ---------------------------------------------


def _validate(message: str, **kwargs):
    return validate_information_request(
        _invoice(),
        message,
        outcome=kwargs.get("outcome", "missing_po"),
        match_status=kwargs.get("match_status", "no_po"),
        allowed_amounts=kwargs.get("allowed_amounts"),
    )


def _check(validation, name: str):
    return next(c for c in validation.checks if c.name == name)


def test_outcome_claim_fails_for_a_blocked_case():
    v = _validate(
        "Re: INV-2001 — the invoice has been received and matched.\n"
        "Next step: none."
    )
    assert not _check(v, "outcome_claim_must_trace").passed
    assert not v.passed


def test_negated_outcome_claim_is_not_a_claim():
    v = _validate(
        "Re: INV-2001 — the invoice is not fully matched yet.\n"
        "Next step: send the purchase order."
    )
    assert _check(v, "outcome_claim_must_trace").passed


def test_figures_must_trace_in_the_request():
    bad = _validate(
        "Re: INV-2001 — the amount of $555.00 is wrong.\n"
        "Next step: send the purchase order."
    )
    assert not _check(bad, "figures_must_trace").passed
    good = _validate(
        "Re: INV-2001 — the stated total of $1,180.00 does not "
        "settle it.\nNext step: send the purchase order.",
        allowed_amounts=[1180.0],
    )
    assert _check(good, "figures_must_trace").passed


def test_identity_next_step_and_payment_language_checks():
    v = _validate("Please send the purchase order soon.")
    assert not _check(v, "references_invoice_id").passed
    assert not _check(v, "states_next_step").passed
    v = _validate(
        "Re: INV-2001 — payment was sent yesterday.\n"
        "Next step: nothing."
    )
    assert not _check(v, "no_payment_language").passed


# -- end to end: the pipeline's gate node carries the policing --------


def test_pipeline_run_polices_the_information_request():
    from ap_agent.model_backends import MockModelBackend

    class _BadComposeBackend(MockModelBackend):
        def compose_information_request(self, context):
            return _BAD.replace("INV-2001", "INV-2002")

    from ap_agent.schemas import DocumentInput

    invoice = InvoiceInput(
        invoice_id="INV-2002",
        supplier_name="Northbeam Office Supplies",
        invoice_number="NB-4472",
        total_amount=250.0,
        documents=[
            DocumentInput(
                document_id="DOC-2002",
                doc_type="invoice",
                text_layer=(
                    "INVOICE NB-4472\nNorthbeam Office Supplies\n"
                    "Total due: 250.00"
                ),
            )
        ],
    )
    result = run_invoice(invoice, backend=_BadComposeBackend())
    info = result.information_request
    assert info is not None and info.needed
    assert info.draft_source == "template_fallback"
    assert info.validation is not None and info.validation.passed
    assert "fully matched" not in info.message
    assert "outcome_claim_must_trace" in info.rejected_draft_checks


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
