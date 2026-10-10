"""Guardrails as code: the checks that make drafts unapprovable.

Each check is exercised on text that should pass and text that should
fail — including the reported-speech cases the screen deliberately
lets through (the guardrail layer is the one that owns them).
"""

from __future__ import annotations

from ap_agent.guardrails import validate_drafts
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import Drafts

INV = sample_invoice_models()[0]  # INV-1001
CITATIONS = ["AP-MATCH-01"]

GOOD_LETTER = (
    "Dear Northbeam team,\n\nThank you for invoice NB-4471 (our reference "
    "INV-1001). The invoice matches the purchase order.\n\nNext step: the "
    "packet goes to a finance approver for review.\n\nThis letter records "
    "the match outcome. It is not a payment confirmation, and no payment "
    "is made or scheduled by this letter."
)
GOOD_PACKET = (
    "AP PACKET — invoice INV-1001\nThree-way match: full match.\n"
    "Policy citations: [AP-MATCH-01]"
)


def _validate(letter: str = GOOD_LETTER, packet: str = GOOD_PACKET, citations=CITATIONS, subject: str = "Invoice INV-1001 received"):
    return validate_drafts(
        INV, Drafts(subject=subject, supplier_letter=letter, ap_packet=packet), citations
    )


def _check(validation, name):
    return next(c for c in validation.checks if c.name == name)


def test_good_drafts_pass_everything():
    validation = _validate()
    assert validation.passed, [c.detail for c in validation.checks if not c.passed]


def test_citation_outside_the_retrieved_set_fails():
    # The judge pack caught a provider letter citing a plausible
    # policy that was never retrieved — invented authority, blocking.
    packet = GOOD_PACKET + "\nSee also [AP-DOC-02]."
    validation = _validate(packet=packet)
    assert not _check(validation, "citations_within_retrieved").passed
    assert not validation.passed


def test_citation_inside_the_retrieved_set_passes():
    assert _check(_validate(), "citations_within_retrieved").passed


def test_payment_made_language_fails():
    letter = GOOD_LETTER + "\nPayment has been made for this invoice."
    validation = _validate(letter=letter)
    assert not _check(validation, "no_payment_language").passed
    assert not validation.passed


def test_payment_will_be_made_on_a_date_fails():
    letter = GOOD_LETTER.replace(
        "Next step: the packet goes to a finance approver for review.",
        "Payment will be made by Friday.",
    )
    assert not _check(_validate(letter=letter), "no_payment_language").passed


def test_negated_disclaimer_is_not_a_violation():
    # "no payment is made" is the required disclaimer, not a promise.
    assert _check(_validate(), "no_payment_language").passed


def test_bank_details_in_draft_fail():
    packet = GOOD_PACKET + "\nPlease remit to the new bank account, account number 8839204716."
    validation = _validate(packet=packet)
    assert not _check(validation, "no_payment_language").passed


def test_sort_code_in_draft_fails():
    packet = GOOD_PACKET + "\nSort code 20-44-18, account 12345678."
    assert not _check(_validate(packet=packet), "no_payment_language").passed


def test_outcome_promise_fails():
    letter = GOOD_LETTER + "\nWe guarantee approval of this invoice."
    assert not _check(_validate(letter=letter), "no_prohibited_promises").passed


def test_time_commitment_fails():
    letter = GOOD_LETTER + "\nThis will be resolved within 5 days."
    assert not _check(_validate(letter=letter), "no_prohibited_promises").passed


def test_ssn_in_draft_fails_pii():
    packet = GOOD_PACKET + "\nContact SSN 123-45-6789."
    assert not _check(_validate(packet=packet), "no_pii_in_draft").passed


def test_luhn_valid_card_fails_pii():
    packet = GOOD_PACKET + "\nCard 4242 4242 4242 4242 on file."
    assert not _check(_validate(packet=packet), "no_pii_in_draft").passed


def test_invoice_reference_number_is_not_pii():
    # Normal AP numbers (invoice ids, PO numbers, amounts) must not
    # trip the PII check.
    packet = GOOD_PACKET + "\nPO-9001 · total 1180.00 · NB-4471."
    assert _check(_validate(packet=packet), "no_pii_in_draft").passed


def test_missing_citations_fail():
    validation = _validate(citations=[])
    assert not _check(validation, "policy_citations_present").passed


def test_missing_invoice_id_fails():
    letter = "Dear supplier,\n\nNext step: review."
    packet = "AP PACKET\nPolicy citations: [AP-MATCH-01]"
    validation = _validate(letter=letter, packet=packet, subject="Invoice received")
    assert not _check(validation, "references_invoice_id").passed


def test_missing_next_step_fails():
    letter = "Dear Northbeam team,\n\nInvoice INV-1001 received and matched."
    validation = _validate(letter=letter)
    assert not _check(validation, "states_next_step").passed
