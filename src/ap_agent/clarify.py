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

The loop closes here too: when the supplier (or purchasing) answers,
the reply is recorded against the open request and analysed in code
(:func:`analyse_information_reply`) — which of the missing items it
supplied, which are still missing, and whether the gap is closed —
so the resolution lives on the record, not in someone's inbox.

And when the reply carries the documents themselves and closes the
gap, the case does not sit at its blocked triage: the service
re-runs it with the supplied facts merged over the stored invoice
(:func:`merge_supplied_facts` — the overlay fills gaps and replaces
the illegible, never rewrites what the case already carries), and
the new outcome supersedes the blocked one with the supersession
recorded in the case's re-analysis chain. The stored invoice is
never rewritten: original + recorded overlay is the re-run's input,
reconstructible exactly. And the chains are read back
(:func:`reanalysis_analytics`): how often a completing reply
actually moved its case, which transitions the replies bought,
and which suppliers' replies most often change nothing.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .guardrails import repair_directive, validate_information_request
from .model_backends import DraftContext
from .schemas import (
    DocumentInput,
    GoodsReceipt,
    InformationRequest,
    InvoiceInput,
    MatchFacts,
    PurchaseOrder,
    Validation,
)

# Wording for each computed missing item.
_ITEM_PHRASES = {
    "purchase_order": "the purchase order for this invoice (or its PO number)",
    "goods_receipt": "confirmation of goods receipt for this delivery (the GRN)",
    "legible_invoice": "a legible copy of the invoice — the scan received could not be read reliably",
}

#: The canonical missing-item keys — the vocabulary a reply is
#: recorded and analysed in (a reply names keys; the record names
#: phrases, via :func:`humanize_item`).
INFORMATION_ITEM_KEYS = tuple(_ITEM_PHRASES)


def humanize_item(item: str) -> str:
    return _ITEM_PHRASES.get(item, item.replace("_", " "))


def missing_item_keys(facts: MatchFacts) -> list[str]:
    """The missing items' canonical keys, computed from the match
    facts — the same evidence as :func:`missing_items`, in the
    reply loop's vocabulary."""
    keys: list[str] = []
    if not facts.has_po:
        keys.append("purchase_order")
    if facts.has_po and not facts.has_receipt:
        keys.append("goods_receipt")
    return keys


def missing_items(facts: MatchFacts, extractions: list | None = None) -> list[str]:
    """The missing-items list, computed from the match facts.

    Driven by what the match could NOT see: no PO attached, no goods
    receipt attached. Legibility is added by the caller via the
    ``unreadable`` triage (see ``needs_information``)."""
    return [humanize_item(key) for key in missing_item_keys(facts)]


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
    keys = missing_item_keys(facts)
    if triage_outcome == "unreadable" and "legible_invoice" not in keys:
        keys = [*keys, "legible_invoice"]
    items = [humanize_item(key) for key in keys]
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
        missing_item_keys=keys,
        message=message,
        requested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        validation=validate(message),
        draft_source=draft_source,
        rejected_draft_checks=rejected,
    )


def apply_supplied_overlay(
    invoice: InvoiceInput,
    *,
    purchase_order: PurchaseOrder | None = None,
    goods_receipt: GoodsReceipt | None = None,
    documents: list[DocumentInput] | None = None,
) -> InvoiceInput:
    """Apply an already-validated overlay to a copy of the
    invoice: the PO / GRN set, each supplied document replacing
    the stored document with its ``document_id``. No refusals
    here — validation is :func:`merge_supplied_facts`' job; this
    is the fold the service uses to layer a request's accumulated
    overlay (and earlier chain entries' overlays) over the stored
    invoice before a re-run."""
    merged = invoice.model_copy(deep=True)
    if purchase_order is not None:
        merged.purchase_order = purchase_order.model_copy(deep=True)
    if goods_receipt is not None:
        merged.goods_receipt = goods_receipt.model_copy(deep=True)
    if documents:
        replacements = {doc.document_id: doc for doc in documents}
        merged.documents = [
            replacements.get(doc.document_id, doc).model_copy(deep=True)
            for doc in merged.documents
        ]
    return merged


def merge_supplied_facts(
    invoice: InvoiceInput,
    *,
    supplied_keys: list[str],
    purchase_order: PurchaseOrder | None = None,
    goods_receipt: GoodsReceipt | None = None,
    documents: list[DocumentInput] | None = None,
) -> tuple[InvoiceInput, list[str]]:
    """Merge a reply's supplied documents over the stored invoice:
    the overlay a re-analysis runs on.

    The overlay fills gaps and replaces the illegible — it never
    rewrites what the case already carries. A purchase order or
    goods receipt rides only where the stored invoice has none
    (a reply that tries to *replace* one is refused by name:
    that is a correction to the case, not an answer to the
    request, and it does not travel this path). A supplied
    document is the legible copy of a document already on the
    invoice — it replaces the stored document with the same
    ``document_id`` (an id the invoice does not carry is refused
    by name, for the same reason). And every payload must be
    accounted for: its canonical key must ride in
    ``supplied_keys`` — a document the reply did not record
    supplying is not merged on its own say-so.

    Returns ``(merged_invoice, overlay_keys)``: a deep copy of
    the invoice with the overlay applied, and the canonical keys
    the overlay carried (in request vocabulary order). The
    stored invoice is untouched — the caller keeps it as the
    original and records the overlay beside the re-run.
    """
    carried: list[tuple[str, object]] = []
    if purchase_order is not None:
        carried.append(("purchase_order", purchase_order))
    if goods_receipt is not None:
        carried.append(("goods_receipt", goods_receipt))
    if documents:
        carried.append(("legible_invoice", documents))
    for key, _payload in carried:
        if key not in supplied_keys:
            raise ValueError(
                f"the reply carries {humanize_item(key)} but does not "
                f"record supplying it — add '{key}' to supplied_items; "
                "a document is merged only for an item the reply "
                "accounts for"
            )
    if purchase_order is not None and invoice.purchase_order is not None:
        raise ValueError(
            "the reply carries a purchase order, but the case "
            "already has one on file — a reply fills the gap the "
            "request named; it does not replace the document the "
            "case was analysed with"
        )
    if goods_receipt is not None and invoice.goods_receipt is not None:
        raise ValueError(
            "the reply carries a goods receipt, but the case "
            "already has one on file — a reply fills the gap the "
            "request named; it does not replace the document the "
            "case was analysed with"
        )
    if documents:
        held = {doc.document_id for doc in invoice.documents}
        for doc in documents:
            if doc.document_id not in held:
                raise ValueError(
                    f"the reply carries a legible copy for document "
                    f"{doc.document_id or '(no id)'}, which the case "
                    "does not hold — a legible copy replaces the "
                    "document that could not be read; it is not a new "
                    "document for the file"
                )
    merged = apply_supplied_overlay(
        invoice,
        purchase_order=purchase_order,
        goods_receipt=goods_receipt,
        documents=documents,
    )
    applied = [key for key, _payload in carried]
    order = {key: i for i, key in enumerate(INFORMATION_ITEM_KEYS)}
    applied.sort(key=lambda key: order.get(key, len(order)))
    return merged, applied


def analyse_information_reply(
    request: InformationRequest,
    supplied_keys: list[str],
    *,
    received_at: str,
    note: str = "",
    recorded_by: str = "",
) -> InformationRequest:
    """Analyse one reply against the open request, in code.

    The reply is data, not a verdict: the items it supplied are
    intersected with the items the request computed as missing,
    and the resolution is arithmetic — everything supplied closes
    the gap (``answered``), some of it moves the case to
    ``partially_answered`` with the still-missing items named,
    and a reply that supplies nothing the request asked for
    leaves the status at ``awaiting_reply`` with the attempt on
    the record. Replies accumulate: a second reply adds its items
    to the first's, and ``reply_received_at`` moves to the latest
    arrival — the wait the queue ages runs from there. Supplied
    keys the request never asked for are ignored here (the
    service refuses unknown keys outright; a known key that was
    not missing is surplus, recorded nowhere). Pure: returns the
    updated request; the caller persists it.
    """
    missing_keys = list(request.missing_item_keys)
    if not missing_keys:
        # Requests stored before the keys were kept: recover them
        # from the phrases — the phrase IS the humanised key.
        by_phrase = {humanize_item(key): key for key in INFORMATION_ITEM_KEYS}
        missing_keys = [
            by_phrase[phrase]
            for phrase in request.missing_items
            if phrase in by_phrase
        ]
    already = {
        phrase
        for phrase in request.reply_supplied_items
    }
    supplied_phrases = [
        humanize_item(key)
        for key in missing_keys
        if key in supplied_keys and humanize_item(key) not in already
    ]
    all_supplied = [*request.reply_supplied_items, *supplied_phrases]
    still_missing = [
        humanize_item(key)
        for key in missing_keys
        if humanize_item(key) not in all_supplied
    ]
    if not still_missing and missing_keys:
        status = "answered"
    elif supplied_phrases or all_supplied:
        status = "partially_answered"
    else:
        status = "awaiting_reply"
    return request.model_copy(
        update={
            "reply_status": status,
            "reply_received_at": received_at,
            "reply_supplied_items": all_supplied,
            "reply_still_missing": still_missing,
            "reply_note": note if note else request.reply_note,
            "reply_recorded_by": recorded_by or request.reply_recorded_by,
        }
    )


def reanalysis_analytics(records: list) -> dict:
    """Read the re-analysis chains back: what the replies bought.

    Every supersession a completing reply caused is recorded on
    its case (``AgentResult.reanalysis_history``) — this is the
    aggregate over that ledger. How many cases were re-run at
    all, how many re-runs there were, and — the figure that
    separates a reply that informed from a reply that merely
    arrived — how often the re-run's outcome CHANGED against how
    often it landed exactly where the blocked case already stood.
    A same-outcome re-run is not a failure of the loop: it is
    the supplier's claim tested and found wanting, and it is
    counted as its own signal, per supplier (below).

    The transitions name the movements (``previous->new``), the
    overlay counts say which supplied item does the unblocking
    (a case whose chain carries the PO twice counts it twice —
    each loop's overlay is that loop's evidence), and the two
    histograms keep the loop's shape honest: how many loops a
    case needed, and how many overlay components it accumulated
    across them. ``suppliers_not_unblocked`` names the suppliers
    whose replies most often fail to move the case — the reply
    landed, the re-run ran, the outcome stood — worst first,
    capped at five; a supplier whose replies always move the
    case never appears. Pure over the records.
    """
    cases = 0
    entries_total = 0
    changed = 0
    unchanged = 0
    transitions: dict[str, int] = {}
    overlay_items: dict[str, int] = {}
    loops_per_case: dict[str, int] = {}
    overlays_per_case: dict[str, int] = {}
    supplier_rows: dict[str, dict] = {}

    def _bucket(histogram: dict[str, int], count: int) -> None:
        key = str(count) if count < 3 else "3+"
        histogram[key] = histogram.get(key, 0) + 1

    for record in records:
        result = record.result
        chain = result.reanalysis_history
        if not chain:
            continue
        cases += 1
        _bucket(loops_per_case, len(chain))
        _bucket(overlays_per_case, sum(len(entry.overlay) for entry in chain))
        supplier = result.supplier_name or ""
        for entry in chain:
            entries_total += 1
            transition = f"{entry.previous_outcome}->{entry.new_outcome}"
            transitions[transition] = transitions.get(transition, 0) + 1
            for key in entry.overlay:
                overlay_items[key] = overlay_items.get(key, 0) + 1
            row = supplier_rows.setdefault(
                supplier, {"reanalyses": 0, "unchanged": 0}
            )
            row["reanalyses"] += 1
            if entry.previous_outcome == entry.new_outcome:
                unchanged += 1
                row["unchanged"] += 1
            else:
                changed += 1
    not_unblocked = [
        {
            "supplier_name": supplier,
            "reanalyses": row["reanalyses"],
            "unchanged": row["unchanged"],
        }
        for supplier, row in supplier_rows.items()
        if row["unchanged"] > 0
    ]
    not_unblocked.sort(
        key=lambda row: (-row["unchanged"], -row["reanalyses"], row["supplier_name"])
    )
    return {
        "cases_reanalysed": cases,
        "reanalyses": entries_total,
        "outcomes_changed": changed,
        "outcomes_unchanged": unchanged,
        "change_rate": (
            round(changed / entries_total, 4) if entries_total else None
        ),
        "transitions": dict(sorted(transitions.items())),
        "overlay_items": dict(sorted(overlay_items.items())),
        "loops_per_case": dict(sorted(loops_per_case.items())),
        "overlays_per_case": dict(sorted(overlays_per_case.items())),
        "suppliers_not_unblocked": not_unblocked[:5],
    }


def supplier_reanalysis_counts(records: list, supplier_name: str) -> dict:
    """One supplier's re-analysis counts, uncapped: the same
    counting :func:`reanalysis_analytics` aggregates, for a
    single supplier. The analytics' worst list is capped at
    five — a supplier can be crowded off it by worse ones
    without their own record changing — so a decision guard
    that must re-verify a named supplier's evidence reads the
    counts here, never off the capped list. Pure."""
    reanalyses = 0
    unchanged = 0
    for record in records:
        result = record.result
        if (result.supplier_name or "") != supplier_name:
            continue
        for entry in result.reanalysis_history:
            reanalyses += 1
            if entry.previous_outcome == entry.new_outcome:
                unchanged += 1
    return {"reanalyses": reanalyses, "unchanged": unchanged}
