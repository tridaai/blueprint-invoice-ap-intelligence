"""Information-request flow: when a case cannot proceed as-is.

Some triages are under-determined by documents, not by judgement: no
purchase order, no goods receipt, or a scan too poor to read. The only
honest next step is asking for the missing piece — from purchasing
(missing PO / receipt) or from the supplier (a legible copy). This
module composes that request — the missing-items list is computed by
code from the match facts, never invented by the model; in provider
mode the model only words the message.

The message is a supplier-facing letter, so it is policed like one:
every composed message — template or provider — passes
``guardrails.validate_information_request`` (the drafts' checks,
scoped to this surface), and the validation rides on the record. A
provider draft that fails gets one repair recompose told the failed
checks and their fixes by name; a repair that still fails falls back
to the template, with the failed checks recorded on the request —
the approver sees what the model tried to send, not just what it
sent.

The request is composed, never sent — like every output here it stops
at the approver gate, and leaves only if a human approves it and
routes it through their own system.
"""

from __future__ import annotations

from .guardrails import repair_directive, validate_information_request
from .model_backends import DraftContext
from .schemas import (
    InformationRequest,
    InvoiceInput,
    MatchFacts,
    Validation,
)

# Wording for each computed missing item.
_ITEM_PHRASES = {
    "purchase_order": "the purchase order for this invoice (or its PO number)",
    "goods_receipt": "confirmation of goods receipt for this delivery (the GRN)",
    "legible_invoice": "a legible copy of the invoice — the scan received could not be read reliably",
}


def humanize_item(item: str) -> str:
    return _ITEM_PHRASES.get(item, item.replace("_", " "))


def missing_items(facts: MatchFacts, extractions: list | None = None) -> list[str]:
    """The missing-items list, computed from the match facts.

    Driven by what the match could NOT see: no PO attached, no goods
    receipt attached. Legibility is added by the caller via the
    ``unreadable`` triage (see ``needs_information``)."""
    items: list[str] = []
    if not facts.has_po:
        items.append(humanize_item("purchase_order"))
    if facts.has_po and not facts.has_receipt:
        items.append(humanize_item("goods_receipt"))
    return items


def needs_information(triage_outcome: str, items: list[str]) -> bool:
    """The trigger: a document-blocked triage with something concrete
    to ask for (missing PO, missing receipt, unreadable scan)."""
    if triage_outcome in ("missing_po", "missing_receipt"):
        return bool(items)
    return triage_outcome == "unreadable"


def template_message(invoice: InvoiceInput, items: list[str]) -> str:
    lines = [
        f"Re: invoice {invoice.invoice_id} — {invoice.supplier_name} "
        f"invoice {invoice.invoice_number or '—'}",
        "",
        "This invoice is on hold in accounts payable. Before it can be "
        "processed, we still need:",
        *[f"- {item}" for item in items],
        "",
        "Next step: send the item(s) above — once we have them, the "
        "invoice goes back through the three-way match. This message "
        "requests information only — it confirms no payment and "
        "commits to no timing.",
    ]
    return "\n".join(lines)


# The fix a repair recompose is told to make, worded for THIS
# surface where the drafts' directive names the wrong artefact
# (the drafts speak of the letter and the packet's tables; the
# request is one message and has no packet). The remaining checks
# reuse guardrails.repair_directive unchanged.
_CLARIFY_DIRECTIVES = {
    "references_invoice_id": (
        "name the invoice id in the message — the request must "
        "say which case it belongs to"
    ),
    "figures_must_trace": (
        "take the amounts out of the message — it asks for "
        "documents; it does not restate figures"
    ),
    "outcome_claim_must_trace": (
        "describe the case exactly as computed — the invoice is "
        "received and on hold; take out any wording that says or "
        "implies it was matched or approved"
    ),
    "states_next_step": (
        'add a line beginning "Next step:" to the message, '
        "saying what happens once the items arrive — describing "
        "the next step in other words does not satisfy the check"
    ),
}


def _failed_checks_text(validation: Validation) -> str:
    """The repair recompose's account of a failed message: which
    checks failed, what each found, the fix each requires — the
    same shape as the drafts' repair feedback (graph.py)."""
    lines = []
    for check in validation.checks:
        if check.passed:
            continue
        directive = _CLARIFY_DIRECTIVES.get(
            check.name
        ) or repair_directive(check.name)
        line = f"- {check.name}: {check.detail}"
        if directive:
            line += f" — fix: {directive}"
        lines.append(line)
    return "\n".join(lines)


def _failed_names(validation: Validation) -> list[str]:
    return [c.name for c in validation.checks if not c.passed]


def build_information_request(
    *,
    invoice: InvoiceInput,
    triage_outcome: str,
    facts: MatchFacts,
    extractions: list | None,
    backend,
    allowed_amounts: list[float] | None = None,
) -> InformationRequest | None:
    """Compose the request when the trigger fires; None otherwise.

    In provider mode the LLM words the message over the computed
    items; the template is the fallback and the default-mode path.
    Either way the stored message has passed
    ``validate_information_request``: a failing provider draft is
    repaired once with the failed checks named, and a repair that
    still fails falls back to the template — the failure stays on
    the record (``rejected_draft_checks``), never in the outbox.
    """
    items = missing_items(facts, extractions)
    if triage_outcome == "unreadable":
        legible = humanize_item("legible_invoice")
        if legible not in items:
            items = [*items, legible]
    if not needs_information(triage_outcome, items):
        return None

    def validate(message: str) -> Validation:
        return validate_information_request(
            invoice,
            message,
            outcome=triage_outcome,
            match_status=facts.match_status,
            allowed_amounts=allowed_amounts,
        )

    message = template_message(invoice, items)
    draft_source = "template"
    rejected: list[str] = []
    compose = getattr(backend, "compose_information_request", None)
    if compose is not None:

        def attempt(feedback: str = "") -> str | None:
            context = DraftContext(
                invoice_id=invoice.invoice_id,
                supplier_name=invoice.supplier_name,
                invoice_number=invoice.invoice_number,
                missing_items=items,
            )
            if feedback:
                context["repair_feedback"] = feedback
            try:
                return compose(context)
            except Exception:  # the request never fails the run
                return None

        candidate = attempt()
        if candidate:
            first = validate(candidate)
            if first.passed:
                message, draft_source = candidate, "provider"
            else:
                rejected = _failed_names(first)
                repaired = attempt(_failed_checks_text(first))
                if repaired:
                    second = validate(repaired)
                    if second.passed:
                        message = repaired
                        draft_source = "provider_repaired"
                    else:
                        rejected = _failed_names(second)
                        draft_source = "template_fallback"
                else:
                    draft_source = "template_fallback"
    return InformationRequest(
        needed=True,
        missing_items=items,
        message=message,
        validation=validate(message),
        draft_source=draft_source,
        rejected_draft_checks=rejected,
    )
