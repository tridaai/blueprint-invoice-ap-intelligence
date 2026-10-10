"""Information-request flow: when a case cannot proceed as-is.

Some triages are under-determined by documents, not by judgement: no
purchase order, no goods receipt, or a scan too poor to read. The only
honest next step is asking for the missing piece — from purchasing
(missing PO / receipt) or from the supplier (a legible copy). This
module composes that request — the missing-items list is computed by
code from the match facts, never invented by the model; in provider
mode the model only words the message.

The request is composed, never sent — like every output here it stops
at the approver gate, and leaves only if a human approves it and
routes it through their own system.
"""

from __future__ import annotations

from .model_backends import DraftContext
from .schemas import InformationRequest, InvoiceInput, MatchFacts

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
        "Once we have these, the invoice goes back through the three-way "
        "match. This message requests information only — it confirms no "
        "payment and commits to no timing.",
    ]
    return "\n".join(lines)


def build_information_request(
    *,
    invoice: InvoiceInput,
    triage_outcome: str,
    facts: MatchFacts,
    extractions: list | None,
    backend,
) -> InformationRequest | None:
    """Compose the request when the trigger fires; None otherwise.

    In provider mode the LLM words the message over the computed
    items; the template is the fallback and the default-mode path.
    """
    items = missing_items(facts, extractions)
    if triage_outcome == "unreadable":
        legible = humanize_item("legible_invoice")
        if legible not in items:
            items = [*items, legible]
    if not needs_information(triage_outcome, items):
        return None
    message = template_message(invoice, items)
    compose = getattr(backend, "compose_information_request", None)
    if compose is not None:
        context = DraftContext(
            invoice_id=invoice.invoice_id,
            supplier_name=invoice.supplier_name,
            invoice_number=invoice.invoice_number,
            missing_items=items,
        )
        try:
            llm_message = compose(context)
        except Exception:  # the request never fails the run
            llm_message = None
        if llm_message:
            message = llm_message
    return InformationRequest(needed=True, missing_items=items, message=message)
