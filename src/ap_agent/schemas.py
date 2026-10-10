"""Typed schemas for the invoice / accounts-payable document agent.

Every boundary in the pipeline is a Pydantic model: the invoice that
arrives (with its purchase order and goods receipt), the OCR and
vision results, the three-way match computed in code, the triage
decision, the drafts, and the final result that waits at the finance
approver's gate. The model never invents the numbers on these objects
— computed fields are computed in code (see ``matching.py``); the
model only does language work over them.

Intake is messy in the real world, so normalisation happens here, at
the boundary: document-type aliases are folded to a small vocabulary
and numeric field values are coerced instead of rejected.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Intake normalisation
# ---------------------------------------------------------------------------

DOC_TYPE_ALIASES = {
    "invoice": "invoice",
    "supplier_invoice": "invoice",
    "tax_invoice": "invoice",
    "bill": "invoice",
    "purchase_order": "purchase_order",
    "po": "purchase_order",
    "purchase order": "purchase_order",
    "goods_receipt": "goods_receipt",
    "grn": "goods_receipt",
    "goods_received_note": "goods_receipt",
    "goods received note": "goods_receipt",
    "delivery_note": "goods_receipt",
    "receiving_report": "goods_receipt",
    "supplier_email": "supplier_email",
    "email": "supplier_email",
    "supplier email": "supplier_email",
    "cover_email": "supplier_email",
    "credit_note": "credit_note",
    "credit note": "credit_note",
    "credit_memo": "credit_note",
    "remittance": "remittance_note",
    "remittance_advice": "remittance_note",
    "remittance advice": "remittance_note",
    "statement": "remittance_note",
}

KNOWN_DOC_TYPES = {
    "invoice", "purchase_order", "goods_receipt",
    "supplier_email", "credit_note", "remittance_note",
}


def normalize_doc_type(raw: str) -> str:
    key = " ".join(str(raw or "").strip().lower().replace("_", " ").split())
    key_us = key.replace(" ", "_")
    return DOC_TYPE_ALIASES.get(key) or DOC_TYPE_ALIASES.get(key_us) or key_us


def _coerce_field_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _to_float(v: Any) -> float:
    if isinstance(v, str):
        v = v.replace(",", "").replace("$", "").replace("€", "").replace("£", "").strip()
        if v == "":
            return 0.0
    return float(v or 0)


class DocumentInput(BaseModel):
    """One document as it arrived: a scan (with a text layer), an email,
    or a system record. ``ocr_text`` is the text the document carries —
    what the offline OCR engine returns, and what a real OCR NIM would
    replace with its own read. ``raw_text`` is kept as a fallback text
    layer for documents that arrive already textual (emails)."""

    doc_type: str
    document_id: str = ""
    object_key: str = ""  # S3 object key of the document blob, when stored
    raw_text: str = ""
    ocr_text: str = ""
    ocr_confidence: Optional[float] = None  # provided scan-quality metadata, 0..1
    scan_quality: str = ""  # provided metadata hint: good | poor | unreadable
    fields: dict[str, str] = Field(default_factory=dict)

    doc_type_provided: str = ""
    doc_type_flagged: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            raw = str(data.get("doc_type", ""))
            norm = normalize_doc_type(raw)
            data["doc_type_provided"] = raw
            data["doc_type"] = norm
            data["doc_type_flagged"] = norm not in KNOWN_DOC_TYPES
            fields = data.get("fields") or {}
            if isinstance(fields, dict):
                data["fields"] = {str(k): _coerce_field_value(v) for k, v in fields.items()}
        return data

    @property
    def text_layer(self) -> str:
        """The document's text as the offline pipeline sees it."""
        return self.ocr_text or self.raw_text


class LineItem(BaseModel):
    description: str = ""
    sku: str = ""
    quantity: float = 0.0
    unit_price: float = 0.0
    line_total: Optional[float] = None

    @field_validator("quantity", "unit_price", "line_total", mode="before")
    @classmethod
    def _num(cls, v: Any) -> Any:
        if v is None:
            return None
        return _to_float(v)


class PurchaseOrder(BaseModel):
    po_number: str = ""
    supplier_name: str = ""
    currency: str = ""
    lines: list[LineItem] = Field(default_factory=list)
    total: Optional[float] = None

    @field_validator("total", mode="before")
    @classmethod
    def _num(cls, v: Any) -> Any:
        if v is None:
            return None
        return _to_float(v)


class GoodsReceipt(BaseModel):
    receipt_id: str = ""
    po_number: str = ""
    received_date: str = ""
    lines: list[LineItem] = Field(default_factory=list)


class InvoiceInput(BaseModel):
    invoice_id: str
    supplier_name: str = ""
    supplier_email_text: str = ""
    notes: str = ""
    invoice_number: str = ""
    invoice_date: str = ""
    due_date: str = ""
    payment_terms: str = ""  # e.g. "2/10 net 30" — parsed in code (discounts.py)
    currency: str = "USD"
    tax_amount: Optional[float] = None
    total_amount: Optional[float] = None
    submitted_at: str = ""  # ISO datetime/date
    line_items: list[LineItem] = Field(default_factory=list)
    documents: list[DocumentInput] = Field(default_factory=list)
    purchase_order: Optional[PurchaseOrder] = None
    goods_receipt: Optional[GoodsReceipt] = None

    @field_validator("tax_amount", "total_amount", mode="before")
    @classmethod
    def _num_opt(cls, v: Any) -> Any:
        if v is None:
            return None
        return _to_float(v)


# ---------------------------------------------------------------------------
# OCR + vision (the two ports #1/#2 did not have)
# ---------------------------------------------------------------------------

class OcrResult(BaseModel):
    document_id: str = ""
    doc_type: str = ""
    engine: str = "offline"  # offline | nim
    mean_confidence: float = 1.0
    char_count: int = 0
    low_confidence: bool = False
    note: str = ""


class VisionResult(BaseModel):
    document_type_detected: str = ""
    is_invoice: bool = False
    legible: bool = True
    quality: str = "good"  # good | poor | unreadable
    confidence: float = 0.0
    source: str = "metadata"  # vlm | metadata
    note: str = ""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

EXTRACTED_FIELDS = [
    "invoice_number", "supplier_name", "invoice_date", "due_date",
    "currency", "subtotal", "tax_amount", "total", "po_number",
]


class FieldExtraction(BaseModel):
    field: str
    value: Optional[str] = None
    confidence: float = 0.0
    status: str = "source_provided"  # match|mismatch|missing_in_text|extracted_only|source_provided


class DocumentExtraction(BaseModel):
    document_id: str = ""
    doc_type: str = ""
    source: str = "provided"  # provided | llm
    note: str = ""
    fields: list[FieldExtraction] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Three-way match (computed in code — never by the model)
# ---------------------------------------------------------------------------

class LineMatch(BaseModel):
    line_no: int = 0
    description: str = ""
    status: str = "matched"  # matched|price_variance|quantity_variance|short_received|missing_on_po|missing_on_invoice|unmatched
    invoice_qty: float = 0.0
    po_qty: float = 0.0
    received_qty: float = 0.0
    invoice_price: float = 0.0
    po_price: float = 0.0
    price_variance_pct: float = 0.0
    detail: str = ""


class MatchFacts(BaseModel):
    match_status: str = "no_po"  # full_match | partial_match | no_match | no_po | no_receipt
    line_matches: list[LineMatch] = Field(default_factory=list)
    matched_lines: int = 0
    total_lines: int = 0
    match_score: float = 0.0  # share of invoice lines fully matched, 0..1
    invoice_lines_total: float = 0.0  # sum of computed line totals
    invoice_total_stated: Optional[float] = None
    totals_variance_pct: float = 0.0
    tax_stated: Optional[float] = None
    tax_expected: Optional[float] = None
    tax_variance: float = 0.0
    currency: str = "USD"
    currency_consistent: bool = True
    has_po: bool = False
    has_receipt: bool = False
    po_number: str = ""
    duplicate_suspected: bool = False
    duplicate_of: str = ""
    invoice_age_days: int = 0
    discrepancy_count: int = 0
    discrepancies: list[str] = Field(default_factory=list)
    document_pair_warning: str = ""


# ---------------------------------------------------------------------------
# Early-payment discount (computed in code — never by the model)
# ---------------------------------------------------------------------------

class DiscountEvaluation(BaseModel):
    """The "2/10 net 30" comparison, priced deterministically.

    ``gross_discount`` is what the supplier's terms offer for paying
    by ``discount_deadline``; ``capital_cost`` is what paying early
    costs (the invoice total tied up for ``days_early`` days at the
    configured cost of capital); ``net_saving`` is the difference.
    The evaluation exists whenever the terms parse and the dates are
    known — ``worthwhile`` is the verdict the option scoring reads.
    """

    terms_text: str = ""  # normalised, e.g. "2/10 net 30"
    discount_pct: float = 0.0
    discount_days: int = 0
    net_days: int = 0
    discount_deadline: str = ""  # ISO date
    days_early: int = 0  # due date minus the evaluation date
    gross_discount: float = 0.0
    capital_cost: float = 0.0
    net_saving: float = 0.0
    cost_of_capital_annual_pct: float = 0.0
    worthwhile: bool = False
    note: str = ""


# ---------------------------------------------------------------------------
# Triage + cross-check
# ---------------------------------------------------------------------------

TRIAGE_OUTCOMES = {
    "clean_match", "price_variance", "quantity_variance", "missing_po",
    "missing_receipt", "duplicate_suspected", "unreadable", "not_an_invoice",
}

SEVERITIES = {"low", "moderate", "high"}


class Triage(BaseModel):
    outcome: str = "missing_po"  # one of TRIAGE_OUTCOMES
    severity: str = "moderate"  # low | moderate | high
    confidence: float = 0.0
    rationale: str = ""
    signals: list[str] = Field(default_factory=list)


class CrossCheck(BaseModel):
    rule_outcome: str = ""
    llm_outcome: Optional[str] = None
    llm_severity: Optional[str] = None
    llm_confidence: Optional[float] = None
    resolution: str = "not_run"  # agree | rules_authoritative | llm_adopted | rules_only | not_run
    note: str = ""


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class RetrievedPolicy(BaseModel):
    policy_id: str
    title: str = ""
    text: str = ""
    score: float = 0.0
    sources: list[str] = Field(default_factory=list)  # retrieval provenance: keyword / semantic


# ---------------------------------------------------------------------------
# Diagnosis / options / drafts / checks
# ---------------------------------------------------------------------------

class Diagnosis(BaseModel):
    summary: str = ""
    risk_factors: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    source: str = "template"  # llm | template
    note: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class RecoveryOption(BaseModel):
    kind: str  # approve_for_payment_prep | route_to_purchasing | request_credit_note | hold_for_review | reject_to_supplier_draft
    title: str = ""
    description: str = ""
    score: float = 0.0
    recommended: bool = False


class Drafts(BaseModel):
    supplier_letter: str = ""
    ap_packet: str = ""
    subject: str = ""


class VerifyResult(BaseModel):
    grounded: bool = True
    issues: list[str] = Field(default_factory=list)
    source: str = "checklist"  # llm | checklist
    note: str = ""


class ReviewResult(BaseModel):
    verdict: str = "pass"  # pass | concerns | block
    findings: list[str] = Field(default_factory=list)
    source: str = "checklist"  # llm | checklist
    enabled: bool = True


class GuardrailCheck(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class Validation(BaseModel):
    passed: bool = True
    checks: list[GuardrailCheck] = Field(default_factory=list)


class InformationRequest(BaseModel):
    needed: bool = False
    missing_items: list[str] = Field(default_factory=list)
    # The missing items' canonical keys (purchase_order |
    # goods_receipt | legible_invoice) beside their humanised
    # phrases — the vocabulary a supplier's reply is analysed in.
    missing_item_keys: list[str] = Field(default_factory=list)
    message: str = ""
    # When the request was composed (the run's stamp). The reply
    # loop's ageing runs from here; requests stored before the
    # stamp existed read "" and the queue falls back to the
    # invoice's own submission date.
    requested_at: str = ""
    # The request is a supplier-facing letter, so it is policed
    # like one (guardrails.validate_information_request): the
    # stored message's validation rides on the record, with the
    # message's provenance — template | provider |
    # provider_repaired | template_fallback — and, when a provider
    # draft was not stored, the checks it failed.
    validation: Optional[Validation] = None
    draft_source: str = "template"
    rejected_draft_checks: list[str] = Field(default_factory=list)
    # The reply loop: a supplier's answer, recorded against the
    # open request and analysed in code (clarify.
    # analyse_information_reply). reply_status runs
    # awaiting_reply → partially_answered → answered; the
    # supplied and still-missing items are named in the same
    # humanised phrases as missing_items, so the record reads
    # as one account of the gap, not two vocabularies.
    reply_status: str = "awaiting_reply"
    reply_received_at: str = ""
    reply_supplied_items: list[str] = Field(default_factory=list)
    reply_still_missing: list[str] = Field(default_factory=list)
    reply_note: str = ""
    reply_recorded_by: str = ""
    # The supplied documents themselves, accumulated across this
    # request's replies (clarify.merge_supplied_facts): the PO,
    # the GRN, the legible copy. They ride on the request until a
    # reply closes the gap; then the case is re-analysed with
    # them merged over the stored invoice as the named overlay,
    # and they move into the re-analysis chain entry that the
    # supersession records. The stored invoice is never rewritten.
    overlay_purchase_order: Optional["PurchaseOrder"] = None
    overlay_goods_receipt: Optional["GoodsReceipt"] = None
    overlay_documents: list["DocumentInput"] = Field(default_factory=list)


class ReanalysisEntry(BaseModel):
    """One supersession in a case's analysis chain: the case was
    re-run because an information reply closed the gap it was
    blocked on, and the new run's outcome replaced the blocked
    one.

    The entry is the whole account of the supersession: the
    outcome it replaced and the outcome that replaced it, the
    items the completing reply supplied, the overlay the re-run
    ran on (the supplied documents, named — the full payloads
    ride below, so original invoice + overlay reconstructs the
    re-run's input exactly), and the request as the reply closed
    it. The stored invoice itself is never rewritten by a
    re-analysis: the overlay rides here, not there, so the
    original extraction stands untouched and every link of the
    chain stays auditable."""

    trigger: str = "information_reply"
    previous_outcome: str = ""
    new_outcome: str = ""
    supplied_items: list[str] = Field(default_factory=list)
    # The overlay components the re-run carried, by canonical
    # item key (purchase_order | goods_receipt | legible_invoice).
    overlay: list[str] = Field(default_factory=list)
    overlay_purchase_order: Optional["PurchaseOrder"] = None
    overlay_goods_receipt: Optional["GoodsReceipt"] = None
    overlay_documents: list["DocumentInput"] = Field(default_factory=list)
    closed_request: Optional["InformationRequest"] = None
    at: str = ""  # ISO timestamp of the re-analysis
    by: str = ""  # the operator who recorded the completing reply
    note: str = ""


class AutonomyRecommendation(BaseModel):
    eligible_for_auto_triage: bool = False
    reasons: list[str] = Field(default_factory=list)


class RunTelemetry(BaseModel):
    backend: str = "mock"
    model: str = ""
    model_calls: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    latency_seconds: float = 0.0
    budget: Optional[dict[str, Any]] = None
    # The completion budgets in force for this run, per call site
    # (the provider-call registry plus the OCR/vision ports), as
    # resolved when the run executed — env overrides included. A
    # starved or truncated reply is explained from the record, not
    # re-derived from code after the fact.
    completion_budgets: Optional[dict[str, int]] = None


class TraceStep(BaseModel):
    name: str
    title: str = ""
    details: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    invoice_id: str
    status: str = "awaiting_approver"  # awaiting_approver | approved | rejected
    supplier_name: str = ""
    invoice_number: str = ""
    ocr_results: list[OcrResult] = Field(default_factory=list)
    vision: Optional[VisionResult] = None
    extractions: list[DocumentExtraction] = Field(default_factory=list)
    facts: MatchFacts = Field(default_factory=MatchFacts)
    injection_flags: list[str] = Field(default_factory=list)
    fraud_flags: list[str] = Field(default_factory=list)
    bank_change_blocked: bool = False
    bank_master_check: Optional[BankMasterCheck] = None
    triage: Triage = Field(default_factory=Triage)
    cross_check: Optional[CrossCheck] = None
    policies: list[RetrievedPolicy] = Field(default_factory=list)
    vector_store: str = ""
    diagnosis: Diagnosis = Field(default_factory=Diagnosis)
    discount: Optional[DiscountEvaluation] = None
    options: list[RecoveryOption] = Field(default_factory=list)
    drafts: Optional[Drafts] = None
    verification: Optional[VerifyResult] = None
    review: Optional[ReviewResult] = None
    reviewer_blocked: bool = False
    validation: Optional[Validation] = None
    repair_attempted: bool = False
    repaired: bool = False
    repair_attempts: int = 0
    original_validation: Optional[Validation] = None
    information_request: Optional[InformationRequest] = None
    needs_information: bool = False
    # The supersession chain: each entry is one re-analysis that
    # replaced this case's earlier outcome (see ReanalysisEntry).
    # Empty for a case analysed once; the current result is the
    # chain's head, and its outcome is the last entry's
    # new_outcome.
    reanalysis_history: list["ReanalysisEntry"] = Field(default_factory=list)
    autonomy: AutonomyRecommendation = Field(default_factory=AutonomyRecommendation)
    telemetry: Optional[RunTelemetry] = None
    trace: list[TraceStep] = Field(default_factory=list)
    decision: Optional[str] = None  # approved | rejected
    decided_by: Optional[str] = None
    decided_at: Optional[str] = None  # ISO timestamp of the gate decision
    decision_reason: Optional[str] = None
    dispatch_status: Optional[str] = None  # sent | failed | None
    posting_receipt: Optional["PostingReceipt"] = None
    # Open-posting escalations fired for this invoice (see
    # PostingEscalation): the webhook ledger, deduped per open
    # episode. Empty until a sweep finds the posting crossed.
    posting_escalations: list["PostingEscalation"] = Field(default_factory=list)
    # Information-wait escalations fired for this invoice (see
    # InformationEscalation): the webhook ledger, deduped per wait
    # episode. Empty until a sweep finds the wait crossed.
    information_escalations: list["InformationEscalation"] = Field(default_factory=list)
    external_action_taken: bool = False
    error: Optional[str] = None


class PostingReceipt(BaseModel):
    """The ERP's posting outcome, recorded back onto the invoice.

    The output adapter (see ``examples/erp_posting.py``) shapes an
    approved result into a posting for the customer's ERP; the ERP
    — or the operator carrying its answer — reports the outcome
    back, and this receipt is that report: ``posted`` with the
    ERP's document number as the external reference, or ``failed``
    with its reason. Recording it is bookkeeping on the invoice
    record: the agent posted nothing itself, and the receipt
    changes no decision. The idempotency key ties the receipt to
    the exact posting it answers."""

    invoice_id: str
    status: str  # posted | failed
    external_reference: str = ""  # the ERP's document number, when posted
    idempotency_key: str = ""  # the key the posting carried
    detail: str = ""  # the ERP's reason, when failed
    recorded_by: str = ""
    recorded_at: str = ""  # ISO timestamp
    source: str = "erp_posting_receipt"


class PostingEscalation(BaseModel):
    """One escalation for an open posting: a signed webhook fired
    when the posting crossed an escalation threshold still
    unposted. The escalation is a *ladder*, not a single shout:
    level 1 (``posting_escalation``) fires past
    ``POSTING_ESCALATION_DAYS``; level 2
    (``posting_escalation_urgent``) fires past twice that, for an
    item that has already been escalated once and is *still*
    open — neglect compounds loudly instead of expiring.

    The ledger lives on the invoice record, beside the receipt
    fields it escalates against. Dedupe is per *episode and
    rung* — (receipt_state, episode_since, level): an episode
    runs from the approval while no receipt exists, or from the
    latest failed receipt (a failure is new information, worth a
    ladder of its own once the episode crosses). A posted
    receipt closes the item and ends the story; it is the reset.
    An attempt is recorded whether the receiver answered
    (``sent``) or not (``failed``): the ledger is the record
    that the crossing was escalated, once per rung."""

    receipt_state: str = ""  # none | failed — the state escalated against
    episode_since: str = ""  # ISO date the episode began — the crossing's identity
    escalation_days: int = 0  # the rung's threshold in force when it fired
    level: int = 1  # ladder rung: 1 = posting_escalation, 2 = posting_escalation_urgent
    age_days: Optional[int] = None  # the posting's age at the crossing
    dispatched_at: str = ""  # ISO timestamp of the dispatch
    dispatch_status: str = ""  # sent | failed


class InformationEscalation(BaseModel):
    """One escalation for an information wait: a signed webhook
    fired when a case blocked on an information request crossed
    twice the aging threshold with the gap still open — the
    posting ladder's shape, one rung, for the wait that depends
    on someone outside the building.

    The ledger lives on the invoice record, beside the request
    it escalates. Dedupe is per *episode*: an episode runs from
    the request (or from a partial reply, which restarts the
    wait — the remaining items have been owed only since that
    answer landed), and each episode escalates at most once. A
    complete reply closes the wait and ends the story; it is the
    reset. An attempt is recorded whether the receiver answered
    (``sent``) or not (``failed``): the ledger is the record that
    the crossing was escalated, once."""

    episode_since: str = ""  # ISO date the wait episode began — the crossing's identity
    escalation_days: int = 0  # the threshold in force when it fired
    waiting_days: Optional[int] = None  # the wait's age at the crossing
    dispatched_at: str = ""  # ISO timestamp of the dispatch
    dispatch_status: str = ""  # sent | failed


class PostingReceiptRequest(BaseModel):
    """Record a posting receipt (the body of
    ``POST /invoices/{id}/posting-receipt``)."""

    status: str  # posted | failed
    external_reference: str = ""
    idempotency_key: str = ""
    detail: str = ""
    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class HandoffGroupReceiptInput(BaseModel):
    """One group's answer in a bank-feedback recording: the
    group is named by its requested execution date (its PmtInf
    id is the run id + that date, computed at recording)."""

    execution_date: str  # the group's requested execution date
    status: str  # accepted | rejected
    detail: str = ""  # the bank's reason — required when rejected
    bank_reference: str = ""


class HandoffGroupReceiptsRequest(BaseModel):
    """Record the bank's per-group answers for a batch-booked
    export (the body of
    ``POST /payment-runs/{id}/handoff/group-receipts``): the
    answers as the bank file gave them, recorded by a named
    operator. A recording is validated whole — one bad answer
    refuses it entire."""

    receipts: list[HandoffGroupReceiptInput] = Field(default_factory=list)
    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class OpenPostingItem(BaseModel):
    """One approved invoice the ERP loop still owes an answer for:
    approved longer ago than the grace window, with no posted
    receipt recorded — either no receipt at all, or a failed one
    nobody has superseded with a successful posting."""

    invoice_id: str
    supplier_name: str = ""
    invoice_number: str = ""
    total_amount: Optional[float] = None
    currency: str = "USD"
    decided_by: str = ""
    decided_at: str = ""
    age_days: Optional[int] = None  # since the approval (see age_basis)
    age_basis: str = ""  # decision | submission | invoice_date
    receipt_state: str = "none"  # none | failed
    receipt_detail: str = ""  # the ERP's failure reason, when failed


class ApprovedNotPostedReport(BaseModel):
    """The ERP loop's open items: approved invoices with no posted
    receipt after the grace window (``POSTING_RECEIPT_GRACE_DAYS``).
    A read model over the store — it posts nothing and chases
    nothing; it makes the loop's open ends as visible as the
    approver queue's."""

    as_of: str = ""
    grace_days: int = 7
    count: int = 0
    totals_by_currency: dict[str, float] = Field(default_factory=dict)
    items: list[OpenPostingItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Marker-risk worklist (supplier onboarding, ranked by marked value)
# ---------------------------------------------------------------------------

class MarkerWorklistItem(BaseModel):
    """One supplier's marked (uncompleted) export exposure, in one
    currency: the value its hand-off exports carried the
    ``SUPPLIER-MASTER-AT-IMPORT`` marker for, because the governed
    master held no approved bank details for it. This is the
    supplier-onboarding worklist's unit of work — onboard this
    supplier through the governed master-change path and the
    marker (and the import risk it prices) goes away.

    The row also carries the onboarding's *flow state*, joined
    from the governed change pipeline (``worklist.py``): a
    supplier whose latest master change is still live — pending,
    or approved and awaiting its effective date — is
    ``in_flight``, not open work: somebody is already on it, and
    the row names the change, its proposer, and how long it has
    been waiting (the decision is owed by a second operator). A
    supplier whose latest change was *rejected* shows the
    rejection — who decided, when, and the recorded note — so
    re-proposing is a deliberate act against a visible refusal,
    not a hopeful retry. An expired pending change is open
    again: it decided nothing and covers nothing."""

    supplier_name: str = ""
    currency: str = "USD"
    marked_value: float = 0.0  # export-time net, summed over marked members
    invoice_count: int = 0  # exported members that carried the marker
    invoice_ids: list[str] = Field(default_factory=list)
    export_count: int = 0  # runs those members exported in
    last_export_date: str = ""  # ISO date of the latest marked export
    # The onboarding flow state, from the supplier's latest
    # governed master change (see the class docstring).
    onboarding_state: str = "open"  # open | in_flight | rejected
    in_flight_change_id: str = ""  # the live change's request id
    in_flight_proposed_by: str = ""  # who proposed it (decision is owed by another)
    in_flight_age_days: Optional[int] = None  # days since the live change was proposed
    # The pending change's expiry early warning: how much of
    # its decision window is left, and whether it is inside
    # the warning band — a pending change undecided past the
    # window expires and the supplier silently returns to
    # open, so the closing window is surfaced while a decision
    # can still rescue it. Only pending changes expire; an
    # approved change awaiting its effective date carries
    # neither field.
    in_flight_days_to_expiry: Optional[int] = None
    in_flight_expiring_soon: bool = False
    in_flight_reproposes: str = ""  # rejected change the live one re-proposes
    rejected_change_id: str = ""  # the rejected change's request id
    rejected_by: str = ""  # who rejected it
    rejected_at: str = ""  # ISO timestamp of the rejection
    rejection_note: str = ""  # the decider's recorded note


class MarkerWorklist(BaseModel):
    """Suppliers ranked by marked export value — the
    master-onboarding worklist, derived from the same export
    receipts the completion-coverage metric prices. Ranked within
    currency (values are never compared across currencies),
    biggest exposure first. Suppliers the master now covers with
    approved bank details are resolved: off the list, counted in
    ``resolved_supplier_count`` with their formerly-marked value,
    so the queue's shrinkage is visible. Listed suppliers carry
    their onboarding flow state: ``in_flight_count`` are already
    being worked through the governed change pipeline (do not
    chase them twice) and ``rejected_count`` had their latest
    change refused (re-propose deliberately, against the recorded
    rejection)."""

    items: list[MarkerWorklistItem] = Field(default_factory=list)
    supplier_count: int = 0
    totals_by_currency: dict[str, float] = Field(default_factory=dict)
    resolved_supplier_count: int = 0
    resolved_value_by_currency: dict[str, float] = Field(default_factory=dict)
    in_flight_count: int = 0  # distinct listed suppliers with a live change
    rejected_count: int = 0  # distinct listed suppliers whose latest change was rejected
    expiring_soon_count: int = 0  # distinct suppliers whose pending change is inside the expiry warning band


# ---------------------------------------------------------------------------
# Credit notes (a supplier document that arrives after the invoice)
# ---------------------------------------------------------------------------

class CreditNoteInput(BaseModel):
    """A credit note as it arrives: its own document, referencing the
    supplier invoice number it credits. Matching it against the
    processed invoices is a query over the store (creditnotes.py) —
    it changes no invoice's state."""

    credit_note_id: str
    supplier_name: str = ""
    credit_note_number: str = ""
    credit_date: str = ""
    currency: str = "USD"
    amount: Optional[float] = None
    referenced_invoice_number: str = ""  # the supplier invoice number credited
    reason: str = ""
    documents: list[DocumentInput] = Field(default_factory=list)

    @field_validator("amount", mode="before")
    @classmethod
    def _num_opt(cls, v: Any) -> Any:
        if v is None:
            return None
        return _to_float(v)


class CreditNoteMatch(BaseModel):
    """The credit-note match report: which processed invoice the note
    belongs to, how much that invoice was overbilled (computed from
    its match facts), and whether the credit settles that variance."""

    credit_note_id: str
    credit_note_number: str = ""
    supplier_name: str = ""
    referenced_invoice_number: str = ""
    matched_invoice_id: str = ""  # our invoice id; "" when unmatched
    status: str = "unmatched"  # settles_variance | partial_settlement | exceeds_variance | no_variance | currency_mismatch | unmatched
    credit_amount: float = 0.0
    overbilled_amount: float = 0.0
    remaining_variance: float = 0.0
    settles_variance: bool = False
    basis: list[str] = Field(default_factory=list)
    note: str = ""


# ---------------------------------------------------------------------------
# Supplier statements (reconciled against the processed invoices)
# ---------------------------------------------------------------------------

class StatementLine(BaseModel):
    invoice_number: str
    amount: Optional[float] = None
    date: str = ""

    @field_validator("amount", mode="before")
    @classmethod
    def _num_opt(cls, v: Any) -> Any:
        if v is None:
            return None
        return _to_float(v)


class SupplierStatement(BaseModel):
    """A supplier's statement of account: the invoices they believe
    are open. Reconciliation compares it against what AP processed —
    report only; it changes no invoice's state."""

    statement_id: str
    supplier_name: str = ""
    statement_date: str = ""
    currency: str = "USD"
    lines: list[StatementLine] = Field(default_factory=list)


class StatementLineResult(BaseModel):
    invoice_number: str
    status: str = "missing_from_ap"  # matched | amount_differs | missing_from_ap
    statement_amount: Optional[float] = None
    ap_amount: Optional[float] = None
    ap_invoice_id: str = ""
    difference: Optional[float] = None  # statement - AP, when both known
    # Duplicate screening (round 7): a statement line is also where a
    # duplicate payment hides — the supplier bills twice for an
    # invoice AP already processed twice, or lists the same number
    # twice. The flag rides alongside the match status: the line
    # still matched, but paying it again is the risk being named.
    duplicate_suspected: bool = False
    duplicate_invoice_id: str = ""  # the second processed record, when AP holds one
    duplicate_of: str = ""  # the original, when the matched record is itself the duplicate


class StatementReconciliation(BaseModel):
    """The reconciliation report: line-by-line outcome, the invoices
    AP processed that the statement omits, totals, and the drafted
    summary a human confirms."""

    statement_id: str
    supplier_name: str = ""
    statement_date: str = ""
    currency: str = "USD"
    lines: list[StatementLineResult] = Field(default_factory=list)
    missing_from_statement: list[dict] = Field(default_factory=list)
    matched_count: int = 0
    differs_count: int = 0
    missing_from_ap_count: int = 0
    missing_from_statement_count: int = 0
    duplicate_count: int = 0  # lines flagged by the duplicate screen
    statement_total: float = 0.0
    ap_total: float = 0.0
    summary: str = ""
    summary_source: str = "template"  # the deterministic composer; never an LLM figure
    confirmed_by: Optional[str] = None
    confirmed_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Supplier master data (the out-of-band record AP-BANK-01 checks against)
# ---------------------------------------------------------------------------

class SupplierMasterRecord(BaseModel):
    """One supplier's approved master record: the bank details AP
    verified out-of-band, plus terms and tax id. This is the record
    document-stated bank details are checked AGAINST — a document
    that states details matching it is evidence; one that states
    different details is the fraud signal. Master data is internal:
    it never appears in drafts (the guardrails forbid bank content
    there), and comparisons report masked labels only."""

    supplier_name: str
    bank_name: str = ""
    account_number: str = ""  # full identifier — internal master data
    iban: str = ""
    sort_code: str = ""
    payment_terms: str = ""  # e.g. "2/10 net 30"
    tax_id: str = ""


class BankMasterCheck(BaseModel):
    """The supplier-master comparison for one invoice, computed in
    code at screening time: the bank identifiers the documents state
    vs the master record. ``matched`` / ``mismatched`` carry masked
    labels only (kind + last four) — full identifiers never leave
    the master record."""

    status: str = "not_checked"  # checked_match | checked_mismatch | no_master_record | no_details_in_documents | not_checked
    supplier_name: str = ""
    source: str = ""  # which master implementation answered: memory | postgres
    matched: list[str] = Field(default_factory=list)
    mismatched: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Governed supplier-master changes (a change is proposed, approved by a
# second operator, and only then applied — effective-dated, audited)
# ---------------------------------------------------------------------------

class MasterChangeEvent(BaseModel):
    """One entry in a master change's audit history: the proposal,
    the decision, and the application to the master, each with its
    actor and timestamp."""

    status: str  # proposed | approved | rejected | applied | expired
    actor: str = ""
    at: str = ""  # ISO timestamp
    note: str = ""


class MasterChangeRequest(BaseModel):
    """A governed change to one supplier's master record.

    Bank-detail changes are the classic AP fraud vector, so the
    master never changes on a document's say-so: an operator (or a
    document, via an operator) *proposes* the change, a second
    operator *approves* it — proposer ≠ approver — recording the
    out-of-band verification (the call-back to a number already on
    file) that backs the decision, and only then is the change
    applied to the master, on or after its effective date. Until
    it is applied, the fraud screen keeps comparing documents
    against the previous record, so the new details still read as a
    mismatch. ``current`` snapshots the master record the proposal
    was made against, so the approver sees old vs new. A request
    nobody decides within the expiry window (``MASTER_CHANGE_
    EXPIRY_DAYS``, default 30) expires instead: undecidable,
    audited, and re-proposable — a live credential change never
    sits around waiting for a careless approval."""

    request_id: str
    supplier_name: str
    proposed: SupplierMasterRecord
    current: Optional[SupplierMasterRecord] = None  # None = new supplier
    source: str = "operator"  # operator | document
    source_invoice_id: str = ""  # set when source == "document"
    reproposes: str = ""  # the rejected change this one re-proposes
    reason: str = ""
    proposed_by: str = ""
    proposed_at: str = ""
    status: str = "pending"  # pending | approved | rejected | expired
    decided_by: Optional[str] = None
    decided_at: Optional[str] = None
    decision_note: str = ""
    verification_ref: str = ""  # out-of-band verification, recorded at approval
    effective_from: str = ""  # ISO date the change takes effect
    applied_at: Optional[str] = None  # when the change landed in the master
    history: list[MasterChangeEvent] = Field(default_factory=list)


class MasterChangeProposeRequest(BaseModel):
    """Propose a master change directly (operator-entered details)."""

    proposed: SupplierMasterRecord
    source: str = "operator"  # operator | document
    source_invoice_id: str = ""
    reason: str = ""
    effective_from: str = ""  # ISO date; "" = effective on approval
    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class MasterChangeFromInvoiceRequest(BaseModel):
    """Propose a master change from a processed invoice's documents:
    the bank details the document states become the proposal."""

    reason: str = ""
    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class MasterChangeDecisionRequest(BaseModel):
    """The second operator's decision on a master change."""

    approve: bool
    verification_ref: str = ""  # required to approve: the out-of-band check
    note: str = ""
    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class MasterChangeReproposeRequest(BaseModel):
    """Re-propose a rejected master change, answering the refusal."""

    response: str = ""  # how the new attempt answers the rejection
    effective_from: str = ""  # ISO date; "" = effective on approval
    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class TwinClusterDecisionRequest(BaseModel):
    """Hand an escalated twin cluster off as one case:
    ``approve_all`` approves every member still at the gate
    through the cluster, ``split`` dissolves the grouping (the
    members are not the same invoice after all). A reason is
    required either way — it is stamped on every member's
    record beside the act."""

    action: str  # approve_all | split
    reason: str = ""
    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class TwinClusterDecision(BaseModel):
    """The record of one cluster hand-off act: the cluster, the
    act, its actor and reason, and each member's outcome under
    it — the same account stamped per member on the records."""

    cluster_id: str
    action: str  # approve_all | split
    actor: str = ""
    reason: str = ""
    decided_at: str = ""
    members: list[dict] = Field(default_factory=list)


class InformationReplyRequest(BaseModel):
    """Record a supplier's reply to an open information request:
    which of the missing items it supplied (canonical keys —
    purchase_order | goods_receipt | legible_invoice), when it
    arrived, and a note. The analysis (what it resolved, what is
    still missing) is computed in code, never asserted here.

    The reply may also carry the supplied documents themselves
    (the PO, the GRN, the legible copy as a replacement
    document). When a reply carrying documents closes the gap,
    the case is re-analysed with those documents merged over
    the stored invoice as a named overlay — the stored invoice
    is never rewritten — and the new outcome supersedes the
    blocked one, with the supersession on the record."""

    supplied_items: list[str] = Field(default_factory=list)
    note: str = ""
    received_at: str = ""  # ISO date/datetime; "" = now
    purchase_order: Optional[PurchaseOrder] = None
    goods_receipt: Optional[GoodsReceipt] = None
    documents: list[DocumentInput] = Field(default_factory=list)
    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


# ---------------------------------------------------------------------------
# Payment runs (proposed from approved invoices — never executed)
# ---------------------------------------------------------------------------

class PaymentRunLine(BaseModel):
    """One approved invoice inside a proposed payment run. Amounts
    are the invoice's stored figures; the discount columns come from
    the run's stored discount evaluation — all computed, none
    narrated."""

    invoice_id: str
    supplier_name: str = ""
    invoice_number: str = ""
    due_date: str = ""
    gross_amount: float = 0.0
    discount_amount: float = 0.0  # capturable early-payment discount, else 0
    discount_deadline: str = ""
    net_amount: float = 0.0  # gross - discount


class PaymentRunDrift(BaseModel):
    """One run member whose discount figure drifted between the
    run's confirmation and its hand-off export: the discount the
    run was confirmed with is no longer the discount the bank run
    can take (its window closed in between). The export re-prices
    every member (see ``paymentruns.discount_drift``) and itemises
    each drift here — the exported file pays the drifted member at
    its export-time figure, never silently at the stale one."""

    invoice_id: str
    supplier_name: str = ""
    discount_deadline: str = ""  # the deadline the run was confirmed against
    discount_at_proposal: float = 0.0
    discount_at_export: float = 0.0  # what is still capturable at export
    discount_lost: float = 0.0  # proposal − export, the drift amount
    reason: str = ""


class PaymentRunWaitLapse(BaseModel):
    """One run member whose early-payment discount lapsed *while
    the run waited for its second confirmation*: capturable when
    the first confirmer signed, gone when the second did. This is
    the priced unit of the four-eyes wait (see
    ``PaymentRun.four_eyes_wait_cost``) — the discount the control
    itself cost, as opposed to drift, which prices the whole
    confirm-to-export span whatever caused it."""

    invoice_id: str
    supplier_name: str = ""
    discount_amount: float = 0.0  # the discount lost to the wait
    discount_deadline: str = ""  # the deadline that closed mid-wait


class PaymentRunForecast(BaseModel):
    """One discounted run member whose discount is *forecast* to
    lapse before the run is likely to be exported.

    A forecast, computed in code at confirmation — not a
    measurement, and never a promise: the member's discount
    deadline is set against the run's expected export date (its
    proposal date plus the measured mean proposal→export lag over
    the runs already exported — the same lag ``/metrics``
    publishes), and the member is flagged when the deadline falls
    before that date. It exists so the confirmer hears "export
    within N days or this discount is projected to lapse" while
    they can still act; the drift re-verification at export
    remains the authority on what actually lapsed."""

    invoice_id: str
    supplier_name: str = ""
    discount_amount: float = 0.0  # the discount at risk
    discount_deadline: str = ""  # the member's discount deadline
    headroom_days: Optional[int] = None  # deadline − confirmation date
    expected_export_date: str = ""  # proposal date + the observed lag
    message: str = ""


class HandoffCreditorEntry(BaseModel):
    """One run member's creditor-account completion in a hand-off
    export: where the file's creditor account came from.

    ``supplier_master`` — the governed master record holds approved
    bank details for the supplier, so the export emitted them (the
    IBAN where recorded, else the account number) instead of the
    completion marker. ``at_import`` — the master holds no approved
    details for the supplier, so the member exported with the
    documented ``SUPPLIER-MASTER-AT-IMPORT`` marker for the
    importing system to complete. ``account_label`` is masked
    (kind + last four): the full identifier lives in the master
    record and in the exported file — never on the run record."""

    invoice_id: str
    supplier_name: str = ""
    source: str = "at_import"  # supplier_master | at_import
    account_kind: str = ""  # iban | account | "" (nothing completed)
    account_label: str = ""  # masked label, e.g. "iban ••••4471"
    # The requested execution date the member was exported under
    # (pain.001 only): the file's single ReqExctnDt, or — under a
    # batch-booking profile — the member's own PmtInf group's
    # date, so completion coverage can split by PmtInf grouping.
    # "" for CSV exports (the layout carries no execution date)
    # and for exports from before the stamp existed.
    execution_date: str = ""


class HandoffGroupReceipt(BaseModel):
    """The bank's answer for one batch-booked ``PmtInf`` group.

    A batch-booked pain.001 file is several payment
    instructions in one message, and the bank answers them per
    group, not per file: one group books, another bounces (a
    creditor account the bank cannot credit, a value date it
    will not take). This is that answer, recorded against the
    group it belongs to — its ``PmtInf`` id (run id + the
    group's requested execution date) and the group's members,
    stamped from the export receipt at recording time, so each
    member carries the group's fate on the run.

    Recording is bookkeeping, like the posting receipt: the
    agent paid nothing and un-pays nothing — the receipt is the
    bank's word about the file the customer already handed it.
    A rejected group may be answered again (the group is
    corrected and re-presented; the new receipt supersedes the
    old, which stays on the run marked ``superseded``). An
    accepted group is final: the money is booked, and a second
    answer for it is a bank-side correction, not a receipt.
    """

    pmt_inf_id: str = ""  # f"{run_id}-{execution_date}" — the group's id in the file
    execution_date: str = ""  # the group's requested execution date
    status: str = ""  # accepted | rejected
    detail: str = ""  # the bank's reason — required when rejected
    bank_reference: str = ""  # the bank's reference for the group, when it gives one
    member_invoice_ids: list[str] = Field(default_factory=list)
    recorded_by: str = ""
    recorded_at: str = ""  # ISO timestamp
    superseded: bool = False  # a later receipt for the same group replaced this one


class HandoffGroupReexport(BaseModel):
    """One bounced batch group's solo re-export: the group
    re-presented as its own single-``PmtInf`` pain.001 file.

    The bank rejected the group (its standing receipt says so,
    with the bank's reason); re-sending the whole batch file
    for one bad group would re-present groups that already
    booked and invite double answers for money already moving.
    So the group goes out ALONE (``service.
    reexport_handoff_group``): one ``PmtInf`` carrying the
    group's ``PmtInf`` id, only its members, at their
    export-time figures with drift re-verified as of the
    re-export date and itemised here. The rejection reason the
    bank gave rides on this record — the corrected file exists
    because of it, and the record says so.

    The re-export resets the group's receipts: the standing
    rejected receipt is marked superseded (it answered the
    FIRST file), and the group stands unanswered against the
    new file until the bank answers it — the same recording
    endpoint takes that answer, against the group's original
    execution date, which is the group's identity in both
    files. Like the export itself, the re-export changes no
    payment state: it is a file in object storage, handed to
    the bank by a human."""

    pmt_inf_id: str = ""  # the group's id in both files: f"{run_id}-{execution_date}"
    execution_date: str = ""  # the group's original requested execution date
    reexport_execution_date: str = ""  # the ReqExctnDt written into the new file
    rejection_reason: str = ""  # the bank's reason, carried from the rejected receipt
    member_invoice_ids: list[str] = Field(default_factory=list)
    key: str = ""  # the object-store key the new file lives under
    sha256: str = ""  # the content hash of the exact bytes stored
    reexported_by: str = ""
    reexported_at: str = ""  # ISO timestamp
    # Discount drift re-verified as of the re-export date for
    # the group's members alone (same entries the export writes
    # for the whole run): the new file pays any drifted member
    # at its re-export-time figure, and this is the receipt of
    # what moved between the two files.
    drift: list[PaymentRunDrift] = Field(default_factory=list)
    drift_total: float = 0.0


class PaymentRun(BaseModel):
    """A proposed payment run: approved invoices grouped by currency
    and due window, totals computed in code. The run is a draft
    object — a finance lead confirms the proposal, and confirmation
    still pays nothing: it records that a human accepted the run as
    the pay list for the customer's AP system of record."""

    run_id: str
    currency: str = "USD"
    window_start: str = ""  # ISO date; the earliest due date in the run
    window_end: str = ""  # ISO date; the last day of the due window
    window_days: int = 7
    status: str = "proposed"  # proposed | confirmed
    lines: list[PaymentRunLine] = Field(default_factory=list)
    invoice_count: int = 0
    total_gross: float = 0.0
    total_discount: float = 0.0
    total_net: float = 0.0
    proposed_at: str = ""
    confirmed_by: Optional[str] = None
    confirmed_at: Optional[str] = None
    # Four-eyes on large runs: a run whose net total exceeded the
    # four-eyes threshold when proposed needs a second confirmation
    # — from a different person than the first — before the hand-off
    # file can be generated. The threshold is per-currency
    # (PAYMENT_RUN_FOUR_EYES_THRESHOLDS, with the global
    # PAYMENT_RUN_FOUR_EYES_THRESHOLD as the fallback);
    # ``four_eyes_threshold`` stamps the figure that applied to
    # this run, in the run's currency, at proposal time — 0 when
    # the run was proposed without thresholds configured.
    four_eyes_required: bool = False
    four_eyes_threshold: float = 0.0
    second_confirmed_by: Optional[str] = None
    second_confirmed_at: Optional[str] = None
    note: str = ""
    # The execution hand-off: the confirmed run rendered as a
    # bank-ready file (CSV layout v1, or ISO 20022 pain.001 XML —
    # ``handoff_format`` says which) and stored through the
    # object-store port. The file is the routing adapter to the
    # customer's banking / ERP system — generating it changes no
    # payment state. These fields are the export's receipt: where
    # the file lives, who generated it, and the sha256 of the
    # exact bytes stored.
    handoff_key: Optional[str] = None
    handoff_sha256: Optional[str] = None
    handoff_generated_at: Optional[str] = None
    handoff_generated_by: Optional[str] = None
    handoff_format: str = ""  # "" until exported; csv | pain001
    # Discount drift found at export time: members whose confirmed
    # discount was no longer capturable when the hand-off was
    # generated. The exported lines above already carry the
    # export-time figures (a drifted member is paid at full gross —
    # the amount actually owed once its window has closed); this is
    # the itemised receipt of what drifted and how much was lost.
    handoff_drift: list[PaymentRunDrift] = Field(default_factory=list)
    handoff_drift_total: float = 0.0  # sum of the members' discount_lost
    # Creditor-account completion receipt for the latest export:
    # one entry per member — completed from the supplier master
    # (the master holds approved bank details) or still carrying
    # the at-import marker (it does not). Masked labels only.
    handoff_creditor_entries: list[HandoffCreditorEntry] = Field(
        default_factory=list
    )
    # Whether the latest export batch-booked (the governing
    # profile's call): the pain.001 carried one PmtInf per
    # requested execution date, BtchBookg true. Stamped at export
    # — the profile in force NOW is not the profile that shaped
    # the file THEN, and the bank's per-group answers
    # (handoff_group_receipts) are only meaningful against the
    # file's own shape.
    handoff_batch_booking: bool = False
    # The bank's per-group answers for a batch-booked export:
    # one receipt per PmtInf group (accepted | rejected), the
    # group's members stamped from the export receipt. A rejected
    # group's later answer supersedes its earlier one.
    handoff_group_receipts: list[HandoffGroupReceipt] = Field(
        default_factory=list
    )
    # Solo re-exports of bounced batch groups: each rejected
    # group that was re-presented as its own single-PmtInf file,
    # with the bank's rejection reason carried on the record
    # (see HandoffGroupReexport). A fresh whole-file export does
    # NOT clear these — they are the history of the groups'
    # corrections; the receipts reset is what a fresh export
    # does, and it does it to every group at once.
    handoff_group_reexports: list[HandoffGroupReexport] = Field(
        default_factory=list
    )
    # Drift forecast, computed at confirmation: the discounted
    # members whose deadlines fall before the run's expected
    # export date (proposal date + the measured proposal→export
    # lag). ``drift_forecast_basis`` states the lag the projection
    # rests on — or why there is none (no exported runs yet).
    drift_forecast: list[PaymentRunForecast] = Field(default_factory=list)
    drift_forecast_basis: str = ""
    # The priced four-eyes wait, stamped when the second
    # confirmation lands (four-eyes runs only): the member
    # discounts that were capturable at the first confirmation
    # and lapsed before the second — the value the wait itself
    # destroyed, itemised in four_eyes_wait_lapses and totalled
    # here. A discount already gone before the first confirmation
    # is not the wait's fault and is never charged to it; one
    # still capturable at the second confirmation cost nothing.
    # Zero on runs that needed no second confirmer.
    four_eyes_wait_lapses: list[PaymentRunWaitLapse] = Field(default_factory=list)
    four_eyes_wait_cost: float = 0.0
    # The hand-off profile that governed the latest export (see
    # HandoffProfile below): the profile whose format was used
    # when the export did not name one, and whose lead days set
    # a pain.001 export's requested execution date. Empty when
    # no profile governs the run. ``handoff_execution_date`` is
    # the ReqExctnDt actually written into a pain.001 file (the
    # CSV carries no execution date; the stamp is "" there).
    handoff_profile_id: str = ""
    handoff_execution_date: str = ""
    # Proposal-time routing flags from the profiles: members
    # whose supplier's bank rejects at-import markers while the
    # master holds no approved bank details for them — flagged
    # when the run is proposed, not discovered at import.
    handoff_profile_flags: list[str] = Field(default_factory=list)


class PaymentRunRequest(BaseModel):
    window_days: int = 7
    as_of: str = ""  # ISO date the windows are anchored on; "" = today


class HandoffProfileEvent(BaseModel):
    """One entry in a hand-off profile's audit trail: who set
    the profile, when, and the values it carried. A profile is
    routing configuration a bank acts on, so every change is
    attributable — the trail is append-only and rides the
    profile record itself."""

    at: str = ""  # ISO timestamp
    actor: str = ""
    action: str = ""  # created | updated
    detail: str = ""  # the effective values, in words


class HandoffProfile(BaseModel):
    """A governed hand-off routing profile: how one scope's
    payment runs should leave the building.

    A run is one file in one format, but the right format is a
    property of the payee side, not of the run: one ERP ingests
    the CSV, another bank wants pain.001, a third needs a few
    days of lead before the requested execution date, and some
    banks reject files whose creditor account is an at-import
    marker instead of real details. The profile is where an
    operator records that knowledge — scoped to one supplier
    (``scope="supplier"``) or to a whole currency
    (``scope="currency"``) — and the export resolves it
    deterministically (``handoff_profiles.py``): an explicit
    format on the export call still wins; below that, the
    run's currency profile; below that, a supplier profile
    every profiled member of the run agrees on; else the CSV
    default. Setting a profile is a governed act: it names its
    actor and appends to the profile's audit trail, and it
    changes no run by itself — it is read at proposal time
    (marker flags) and at export time (format, lead days)."""

    profile_id: str  # HP-CUR-<CODE> | HP-SUP-<slug>
    scope: str = "currency"  # supplier | currency
    supplier_name: str = ""  # set when scope == "supplier"
    currency: str = ""  # set when scope == "currency"
    format: str = "csv"  # csv | pain001 — the scope's preferred hand-off format
    lead_days: int = 0  # requested-execution lead for pain.001 (see handoff.py)
    markers_rejected: bool = False  # the payee's bank rejects at-import markers
    # Batch booking: the payee's bank books per value date, so a
    # pain.001 export groups the run's lines into one PmtInf per
    # requested execution date (BtchBookg true) instead of one
    # PmtInf for the whole run (see handoff.batch_groups — the
    # profile's lead days floor every group's date).
    batch_booking: bool = False
    note: str = ""
    updated_by: str = ""
    updated_at: str = ""  # ISO timestamp of the latest set
    history: list[HandoffProfileEvent] = Field(default_factory=list)


class HandoffProfileRequest(BaseModel):
    """Setting a hand-off profile: the scope key (a supplier
    name or a currency code), the routing values, and the
    operator setting them — a profile without an actor is not
    governed, so the service refuses one."""

    actor: Optional[str] = None
    scope: str = "currency"  # supplier | currency
    supplier_name: str = ""
    currency: str = ""
    format: str = "csv"
    lead_days: int = 0
    markers_rejected: bool = False
    batch_booking: bool = False
    note: str = ""


class ProfileProposalEvent(BaseModel):
    """One entry in a profile-change proposal's audit trail:
    the drafting, and the decision that closed it."""

    status: str = ""  # proposed | approved | rejected
    actor: str = ""
    at: str = ""  # ISO timestamp
    note: str = ""


class ProfileChangeProposal(BaseModel):
    """A governed change to a hand-off routing profile, drafted
    from the profile's effectiveness grade.

    The grade (``handoff_profiles.profile_effectiveness``) names
    configuration theatre plainly — a profile that governs no
    exports, or whose exports carry the at-import marker at the
    same rate as ungoverned ones — but a read changes nothing.
    This is the governed path from the grade to the change,
    mirroring the master-change discipline: the proposal is
    *drafted* from the evidence (``basis`` carries the grade's
    account and the profile's values as graded), a second
    operator *decides* it (proposer ≠ decider), and only an
    approval applies it — a retirement deletes the profile
    record, a marker tightening sets the flag through the
    profile's own governed path. Drafting applies nothing, and
    the agent never applies on its own say-so: left undecided, a
    proposal is a question, not a change."""

    proposal_id: str  # PCP-0001 …
    profile_id: str
    kind: str  # retire | tighten_markers
    basis: str = ""  # the grade's account the proposal answers
    current: dict = Field(default_factory=dict)  # the profile's values as graded
    # The values approval would put in force; None for a
    # retirement (approval deletes the profile record instead).
    proposed: Optional[dict] = None
    proposed_by: str = ""
    proposed_at: str = ""
    status: str = "pending"  # pending | approved | rejected
    decided_by: Optional[str] = None
    decided_at: Optional[str] = None
    decision_note: str = ""
    applied_at: Optional[str] = None  # when approval landed the change
    history: list[ProfileProposalEvent] = Field(default_factory=list)


class LagSegmentProposalEvent(BaseModel):
    """One entry in a lag-segment proposal's audit trail: the
    drafting, and the decision that closed it."""

    status: str = ""  # proposed | approved | rejected
    actor: str = ""
    at: str = ""  # ISO timestamp
    note: str = ""


class LagSegmentProposal(BaseModel):
    """A governed decision about a lag statistic, drafted from
    the drift watch.

    The drift watch (``paymentruns.lag_grade_drift_watch``)
    demotes a decaying statistic inside the forecast's
    selection, run by run — sound, but automatic and silent: no
    human ever decided the segment's fate, and the demotion
    lifts the moment the recent window turns over, however
    briefly. This is the governed path from the watch to a
    standing decision, under the same discipline as the
    profile-change proposals (drafted from the evidence by one
    operator, decided by a second, applied only on approval):

    - ``retire_segment`` — the watch names the segment's
      statistic degrading; approval takes it out of the
      forecast's candidate set altogether (the selection skips
      it and says so in the basis) until a re-grade re-admits
      it. Nothing the statistic computed is rewritten; it is
      removed from the set of statistics that may project.
    - ``regrade_segment`` — an excluded segment whose recent
      record has recovered (the watch assesses it again and no
      longer names it degrading); approval re-admits it to the
      candidate set, re-graded on the record as it now stands.
      The ordinary grading then decides whether it wins.

    Which segments stand excluded is DERIVED from this ledger —
    the latest approved decision per segment — so the proposals
    are the audit trail and the state in one document, and
    drafting applies nothing: left undecided, a proposal is a
    question, not an exclusion.
    """

    proposal_id: str  # LSP-0001 …
    segment: str  # the statistic key: global | currency:EUR | profile:<id>
    kind: str  # retire_segment | regrade_segment
    basis: str = ""  # the watch's account the proposal answers
    # The watch + grade readings the draft rests on (recalls,
    # graded members) — the decider compares evidence, not prose.
    evidence: dict = Field(default_factory=dict)
    proposed_by: str = ""
    proposed_at: str = ""
    status: str = "pending"  # pending | approved | rejected
    decided_by: Optional[str] = None
    decided_at: Optional[str] = None
    decision_note: str = ""
    applied_at: Optional[str] = None  # when approval landed the decision
    history: list[LagSegmentProposalEvent] = Field(default_factory=list)


class LagSegmentProposalDraftRequest(BaseModel):
    """Draft lag-segment proposals from the drift watch: the
    operator running the drafting sweep."""

    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class LagSegmentProposalDecisionRequest(BaseModel):
    """The second operator's decision on a lag-segment proposal:
    proposer ≠ decider, as on profile changes."""

    approve: bool
    note: str = ""
    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class ProfileProposalDraftRequest(BaseModel):
    """Draft profile-change proposals from the effectiveness
    grades: the operator running the drafting sweep."""

    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class ProfileProposalDecisionRequest(BaseModel):
    """The second operator's decision on a profile-change
    proposal: proposer ≠ decider, as on master changes."""

    approve: bool
    note: str = ""
    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


# ---------------------------------------------------------------------------
# AP aging queue (a read model over the awaiting invoices)
# ---------------------------------------------------------------------------

class QueueItem(BaseModel):
    """One invoice waiting at the approver gate, aged by due date."""

    invoice_id: str
    supplier_name: str = ""
    invoice_number: str = ""
    status: str = "awaiting_approver"
    outcome: str = ""
    severity: str = ""
    total_amount: Optional[float] = None
    currency: str = "USD"
    due_date: str = ""
    days_overdue: Optional[int] = None  # None when no due date is known
    bucket: str = "no_due_date"  # overdue_30_plus|overdue_8_30|overdue_1_7|due_today|due_within_7|due_later|no_due_date
    age_days: Optional[int] = None  # since submission, when known
    discount_deadline: str = ""
    discount_days_left: Optional[int] = None


class RecoveryCaseAgingItem(BaseModel):
    """One open recovery case, aged: how long the case has existed
    and how long it has sat in its current status. A case breaches
    its SLA when the current status has lasted longer than the
    configured limit — money is leaking while it sits."""

    pack_id: str
    duplicate_invoice_id: str
    supplier_name: str = ""
    currency: str = "USD"
    amount: float = 0.0
    case_status: str = "opened"  # the step the case is stuck on
    age_days: Optional[int] = None  # since the case opened
    status_age_days: Optional[int] = None  # since the last transition
    sla_breached: bool = False
    bucket: str = "unknown"  # age_0_7 | age_8_30 | age_30_plus | unknown


class FourEyesAgingItem(BaseModel):
    """One large payment run waiting on its second confirmer past
    its aging threshold: the run is confirmed once, its hand-off
    is locked, and the second pair of eyes has not arrived. A
    queue-shaped stall — the run cannot export, and its discounts
    age while it waits — so it joins the approver's worklist as an
    action item, with the first confirmer named: they are the
    person to nudge, and the record of who is *not* allowed to be
    the second.

    The threshold is per run (``aging_threshold_days`` — the one
    that actually applied): the configured
    ``PAYMENT_RUN_FOUR_EYES_AGING_DAYS`` for an ordinary run, but
    tightened to 0 when the run's confirmation-time drift forecast
    flags members lapsing before the expected export — a run
    projected to burn discounts while it waits has no patience to
    spare, so it is chased from the first day of the wait, and the
    item carries the projected lapse value the wait puts at risk."""

    run_id: str
    currency: str = "USD"
    total_net: float = 0.0
    four_eyes_threshold: float = 0.0  # the gate the run was stamped against
    confirmed_by: str = ""  # the first confirmer — named, on purpose
    confirmed_at: str = ""  # ISO timestamp of the first confirmation
    age_days: Optional[int] = None  # since the first confirmation
    aging_threshold_days: int = 0  # the threshold that applied to this run
    forecast_tightened: bool = False  # the forecast tightened the threshold
    forecast_lapse_count: int = 0  # members the forecast flags
    forecast_value_at_risk: float = 0.0  # their discounts, summed


class TwinEscalationItem(BaseModel):
    """One duplicate-twin cluster escalated onto the approver's
    worklist, with its twin graph attached (members + the
    screened edges between them, as ``GET /audit/twins`` reads
    them). The graph alone was audit-only: a third copy could
    arrive, or one twin could be approved, while its sibling
    waited — and the approver working the queue never saw the
    cluster as a case. A cluster escalates when ANY trigger
    fires:

    * ``cluster_size`` — at least ``TWIN_CLUSTER_MIN_SIZE``
      records (default 3): the same invoice arriving three ways
      is a pattern to work, not a coincidence to file;
    * ``combined_value`` — the members' totals in a single
      currency reach ``TWIN_CLUSTER_VALUE_THRESHOLD`` (default
      10000): the double-payment exposure is material even in a
      pair (totals are summed per currency, never across);
    * ``approved_and_awaiting`` — one twin is already approved
      while another still waits at the gate: the exact shape of
      a double payment in flight, at any size or value.
    """

    cluster_id: str
    supplier_name: str = ""  # canonical (the screen's normalised form)
    invoice_number: str = ""  # canonical invoice number (the screen's key)
    size: int = 0
    totals_by_currency: dict[str, float] = Field(default_factory=dict)
    triggers: list[str] = Field(default_factory=list)
    arrival_shapes: list[str] = Field(default_factory=list)
    members: list[dict] = Field(default_factory=list)  # twin-graph nodes
    edges: list[dict] = Field(default_factory=list)  # screened duplicate_of edges


class InformationWaitItem(BaseModel):
    """One invoice blocked on an information request nobody has
    fully answered, aged onto the approver's worklist.

    The request leaves the building (a human sends it through
    their own system), and the wait that follows was invisible:
    the invoice sat at the gate with its drafts while purchasing
    or the supplier held the missing piece, and nothing aged.
    Past ``INFORMATION_WAIT_AGING_DAYS`` the wait joins the queue
    like the other stalls — the request's date, how long it has
    been waiting, the reply state (none yet, or a partial answer
    that restarted the wait when it landed), and the items still
    missing, named. A fully answered request closes the wait and
    drops off; a decided invoice is done waiting either way."""

    invoice_id: str
    supplier_name: str = ""
    invoice_number: str = ""
    outcome: str = ""
    requested_at: str = ""
    reply_status: str = "awaiting_reply"  # awaiting_reply | partially_answered
    reply_received_at: str = ""
    waiting_days: Optional[int] = None
    aging_threshold_days: int = 0
    still_missing: list[str] = Field(default_factory=list)
    # The wait episode's start (ISO date) — the anchor the age
    # runs from and the escalation ledger dedupes against.
    episode_since: str = ""
    # Whether this episode has already been escalated by the
    # information-wait sweep (its ledger entry is on the record).
    escalated: bool = False


class GroupReceiptWaitItem(BaseModel):
    """One batch-booked ``PmtInf`` group the bank has not
    answered, aged onto the approver's worklist.

    The bank answers per group, and an answer that never comes
    is its own kind of silence: the file went out, the other
    groups booked, and this group's members hang in a state
    nobody inside the building can see — not paid, not bounced,
    just unanswered. Past ``GROUP_RECEIPT_AGING_DAYS`` the
    group joins the queue like the other waits: its run, its
    members, the value riding on it, and how long the bank has
    had the file. The wait's clock runs from the file the bank
    is currently answering — the whole-file export, or the
    group's solo re-export when it was re-presented (a
    re-exported group reads ``reexported``: its silence is
    about the corrected file, not the original one). An
    answered group — accepted or rejected — is not waiting:
    a rejection is work for the re-export, not for the queue."""

    run_id: str
    execution_date: str = ""  # the group's requested execution date
    pmt_inf_id: str = ""  # the group's id in the file
    member_invoice_ids: list[str] = Field(default_factory=list)
    member_count: int = 0
    currency: str = "USD"
    group_value: float = 0.0  # the members' net total, in the run's currency
    file_sent_at: str = ""  # when the file being answered went out (ISO)
    waiting_days: Optional[int] = None
    aging_threshold_days: int = 0
    reexported: bool = False  # the file being answered is the solo re-export


class QueueView(BaseModel):
    """The aging queue: awaiting invoices, overdue first, with the
    triage mix — a pure read model over the store's records. Open
    recovery cases ride along with their own aging, because a case
    stuck at one step is the other half of the approver's workload."""

    as_of: str = ""
    count: int = 0
    overdue_count: int = 0
    triage_mix: dict[str, int] = Field(default_factory=dict)
    totals_by_currency: dict[str, float] = Field(default_factory=dict)
    buckets: dict[str, int] = Field(default_factory=dict)
    items: list[QueueItem] = Field(default_factory=list)
    recovery_cases: list[RecoveryCaseAgingItem] = Field(default_factory=list)
    recovery_buckets: dict[str, int] = Field(default_factory=dict)
    recovery_open_count: int = 0
    recovery_sla_breaches: int = 0
    recovery_sla_days: int = 0
    # Escalated open postings: approved invoices still unposted
    # past the escalation threshold (POSTING_ESCALATION_DAYS) join
    # the worklist as action items, each carrying its ERP failure
    # reason when a receipt failed.
    open_postings: list[OpenPostingItem] = Field(default_factory=list)
    open_posting_count: int = 0
    posting_escalation_days: int = 0
    # Four-eyes runs awaiting their second confirmer past the
    # aging threshold (PAYMENT_RUN_FOUR_EYES_AGING_DAYS): the run
    # is locked out of hand-off export until a second, different
    # confirmer arrives, so past the threshold the wait itself is
    # the action item — first confirmer named.
    four_eyes_pending: list[FourEyesAgingItem] = Field(default_factory=list)
    four_eyes_pending_count: int = 0
    four_eyes_aging_days: int = 0
    # Twin clusters escalated as work items (TWIN_CLUSTER_MIN_SIZE /
    # TWIN_CLUSTER_VALUE_THRESHOLD, or any approved+awaiting span):
    # the cluster and its graph join the worklist, so a twin
    # arriving while its sibling waits is worked as one case.
    twin_escalations: list[TwinEscalationItem] = Field(default_factory=list)
    twin_escalation_count: int = 0
    twin_cluster_min_size: int = 0
    twin_cluster_value_threshold: float = 0.0
    # Information waits: invoices blocked on an information
    # request past its aging threshold (INFORMATION_WAIT_AGING_DAYS)
    # with no complete reply recorded — the supplier or purchasing
    # holds the missing piece, and the wait is worklist work.
    information_waits: list[InformationWaitItem] = Field(default_factory=list)
    information_wait_count: int = 0
    information_wait_aging_days: int = 0
    # Batch groups the bank has not answered past their aging
    # threshold (GROUP_RECEIPT_AGING_DAYS): the file went out,
    # the other groups answered, this one's members hang
    # unbooked and unbounced — bank silence as a worklist item.
    group_receipt_waits: list[GroupReceiptWaitItem] = Field(
        default_factory=list
    )
    group_receipt_wait_count: int = 0
    group_receipt_aging_days: int = 0


# ---------------------------------------------------------------------------
# Escalation digest (the day's escalations, composed — never sent)
# ---------------------------------------------------------------------------

class DigestForecastLapse(BaseModel):
    """One discount projected to lapse, on a run confirmed on the
    digest's day: the forecast section of the digest. Carries the
    run it belongs to (``PaymentRunForecast`` itself does not).
    ``blocked_on_four_eyes`` names the stall inside the
    projection: the run cannot export until its second
    confirmation lands, so the expected export date the forecast
    assumes is hostage to the wait — the lapse is not just
    projected, it is queued behind a person."""

    run_id: str
    invoice_id: str
    supplier_name: str = ""
    currency: str = "USD"
    discount_amount: float = 0.0
    discount_deadline: str = ""
    expected_export_date: str = ""
    blocked_on_four_eyes: bool = False


class DigestDriftEntry(BaseModel):
    """One run exported on the digest's day with discount drift:
    the realised counterpart of the forecast section — what the
    drift cost, in the end, on the day it was measured."""

    run_id: str
    currency: str = "USD"
    members_drifted: int = 0
    value_lost: float = 0.0


class DigestOnboardingEntry(BaseModel):
    """One supplier whose onboarding change is about to expire
    undecided: the marker worklist's early warning, carried
    into the digest. The change was proposed, a second operator
    owes the decision, and ``days_to_expiry`` of the decision
    window remain — past the window the change expires, the
    supplier returns to open work, and the marked exports keep
    flowing to a bank the master still cannot complete for."""

    supplier_name: str = ""
    currency: str = "USD"
    change_id: str = ""
    proposed_by: str = ""
    age_days: Optional[int] = None  # days the change has waited
    days_to_expiry: Optional[int] = None  # days of window left
    marked_value: float = 0.0  # the export exposure riding on it


class DigestWaitBudgetEntry(BaseModel):
    """One currency whose four-eyes waits have destroyed more
    discount value this month than its monthly budget allows
    (``FOUR_EYES_WAIT_BUDGETS``): the wait cost month-to-date,
    the budget, and the excess. The digest is a daily surface
    carrying a monthly fact — a breached budget stays breached
    for the rest of the month, and every morning's digest keeps
    saying so until the month turns."""

    currency: str = "USD"
    month: str = ""  # YYYY-MM the cost is charged to
    wait_cost: float = 0.0  # discounts the waits destroyed, month to date
    budget: float = 0.0  # the monthly ceiling in force
    excess: float = 0.0  # wait_cost − budget


class DigestLagWatchEntry(BaseModel):
    """One lag statistic the drift watch has something to say
    about, announced in the digest.

    The watch grades each lag segment's recall over two windows
    of export history and demotes a decaying statistic inside
    the forecast's selection — silently, run by run. A forecast
    quietly degrading is an operations event, so the digest
    announces it: a segment whose recent recall fell away from
    its earlier record (``degrading``), and the segments that
    stand excluded from the candidate set by governed decision
    (``excluded`` — announced so the exclusion itself never
    becomes invisible). The recalls and graded-member counts
    ride along: the announcement carries its evidence, in the
    digest's discipline."""

    segment: str = ""  # e.g. "currency:EUR", "profile:...", "global"
    earlier_recall: Optional[float] = None
    recent_recall: Optional[float] = None
    earlier_graded_members: int = 0
    recent_graded_members: int = 0
    degrading: bool = False
    excluded: bool = False


class EscalationDigest(BaseModel):
    """The day's escalations in one operator summary: the queue's
    action items (recovery cases past their SLA, open postings past
    escalation, overdue invoices stuck at the gate, four-eyes runs
    waiting on a second confirmer) plus the day's payment-run news
    (discounts forecast to lapse on runs confirmed today, drift
    realised on runs exported today).

    Composed deterministically from the stored records, like the
    reconciliation summary — the agent drafts nothing free-form
    here and sends nothing: ``summary`` is the text a human reads
    (or forwards), and the sections beneath it are the same facts
    in structured form."""

    as_of: str = ""  # ISO date the digest covers
    summary: str = ""  # the composed operator text
    recovery_breaches: list[RecoveryCaseAgingItem] = Field(default_factory=list)
    recovery_breach_count: int = 0
    recovery_amount_by_currency: dict[str, float] = Field(default_factory=dict)
    posting_escalations: list[OpenPostingItem] = Field(default_factory=list)
    posting_escalation_count: int = 0
    stuck_items: list[QueueItem] = Field(default_factory=list)
    stuck_invoice_count: int = 0
    most_overdue_days: Optional[int] = None
    four_eyes_pending: list[FourEyesAgingItem] = Field(default_factory=list)
    four_eyes_pending_count: int = 0
    forecast_lapses: list[DigestForecastLapse] = Field(default_factory=list)
    forecast_value_by_currency: dict[str, float] = Field(default_factory=dict)
    drift_today: list[DigestDriftEntry] = Field(default_factory=list)
    drift_value_by_currency: dict[str, float] = Field(default_factory=dict)
    # Four-eyes wait budgets breached this month (see
    # DigestWaitBudgetEntry): a monthly governance fact on a
    # daily surface, present only while a breach stands.
    wait_budget_breaches: list[DigestWaitBudgetEntry] = Field(default_factory=list)
    # Onboarding changes inside their expiry warning band
    # (see DigestOnboardingEntry): decisions owed while the
    # window can still be rescued, most urgent first.
    onboarding_expiring: list[DigestOnboardingEntry] = Field(default_factory=list)
    onboarding_expiring_count: int = 0
    # Batch groups the bank has not answered past their aging
    # threshold (from the same queue view): the file went out
    # days ago and one group has simply never answered — bank
    # silence, named in the day's summary like any other stall.
    group_receipt_waits: list[GroupReceiptWaitItem] = Field(
        default_factory=list
    )
    group_receipt_wait_count: int = 0
    # The lag drift watch's announcements: segments whose
    # statistic is degrading (recent recall fallen away from
    # the earlier record) and segments standing excluded from
    # the forecast's candidate set by governed decision — plus
    # how many of the day's confirmations the watch's demotions
    # actually shaped. A forecast quietly degrading is an ops
    # event; the digest is where ops events are announced.
    lag_watch: list[DigestLagWatchEntry] = Field(default_factory=list)
    lag_watch_degrading_count: int = 0
    lag_watch_excluded_segments: list[str] = Field(default_factory=list)
    lag_watch_demotions_today: int = 0
    # Provenance: a digest is "computed" at read time by default —
    # fresh every call, and labelled as such. An operator can store
    # a day's first-computed digest (POST /digest/{date}/store);
    # the stored copy carries provenance "stored" with who stored
    # it and when. The stored copy is the one a human forwards or
    # files for audit: the records move on, and yesterday's digest
    # cannot be re-derived from them — a recomputation never wears
    # the "stored" label.
    provenance: str = "computed"  # computed | stored
    stored_at: str = ""  # ISO timestamp, when stored
    stored_by: str = ""  # the operator who stored it


class DigestHistoryEntry(BaseModel):
    """One stored digest's headline counts, with the change
    against the previous stored day: the digest series read as a
    trend — is the building's escalation load growing or
    shrinking? ``changes`` maps each headline count that *moved*
    to this day minus the previous stored day (a count that held
    is absent, not a zero); the earliest stored day has no
    predecessor, so its changes are empty rather than dressed up
    as deltas from zero."""

    as_of: str = ""  # the digest's date
    stored_at: str = ""
    stored_by: str = ""
    recovery_breach_count: int = 0
    posting_escalation_count: int = 0
    stuck_invoice_count: int = 0
    four_eyes_pending_count: int = 0
    forecast_lapse_count: int = 0
    drift_run_count: int = 0
    changes: dict[str, int] = Field(default_factory=dict)


class DigestStoreSweepEntry(BaseModel):
    """One day in the digest auto-store sweep's report: the day
    was ``stored`` by this sweep (its first-computed digest,
    pinned now, attributed to ``system``), or ``already_stored``
    — first-computed had already won for that day, and the sweep
    leaves the stored copy, and its human attribution, untouched."""

    date: str = ""  # ISO date the entry covers
    outcome: str = ""  # stored | already_stored
    stored_by: str = ""  # who the day's stored copy is attributed to


class DigestStoreSweep(BaseModel):
    """The digest auto-store sweep's report: which recent days it
    pinned and which it found already pinned, over a bounded
    lookback ending at ``through_date``. The sweep exists so the
    digest history — the trend the operators read — stops
    depending on someone remembering to press Store every
    morning: a scheduler can run it daily and every missed day
    inside the lookback still lands in the series. Days a human
    stored keep their human attribution; sweep-stored days are
    attributed to ``system`` in the stored copy's own provenance,
    so a backfilled day never masquerades as a morning ritual."""

    through_date: str = ""  # ISO date the lookback ends at
    lookback_days: int = 0  # days covered, today included
    stored_count: int = 0
    already_stored_count: int = 0
    entries: list[DigestStoreSweepEntry] = Field(default_factory=list)


class DigestHistory(BaseModel):
    """The stored digests in date order, headline counts only:
    the day-over-day view over the audit-file copies. Days nobody
    stored are absent — the history never invents a day."""

    entries: list[DigestHistoryEntry] = Field(default_factory=list)
    count: int = 0


# ---------------------------------------------------------------------------
# Duplicate-payment recovery packs
# ---------------------------------------------------------------------------

class RecoveryCaseEvent(BaseModel):
    """One operator transition on a recovery case: the audit entry.
    The lifecycle is worked by humans (the agent contacts nobody and
    books no credit); each transition records who moved the case,
    when, and an optional note."""

    status: str  # the case status this transition moved TO
    actor: str = ""
    at: str = ""  # ISO timestamp
    note: str = ""


# The recovery-case lifecycle, in order. A case opens when its pack
# is drafted; an operator walks it forward one step at a time as the
# supplier is contacted and the credit lands.
RECOVERY_CASE_FLOW = ["opened", "supplier_contacted", "credit_received", "closed"]


class RecoveryEscalation(BaseModel):
    """One SLA-breach escalation for a recovery case: the signed
    ``recovery_escalation`` webhook fired when the case's current
    status outlasted the SLA (``RECOVERY_CASE_SLA_DAYS``).

    The ledger lives on the pack, NOT in ``case_history``, on
    purpose: the aging clock runs from the last history event, so
    an escalation entry there would reset the very status age it
    reports. Dedupe is by (case_status, status_since) — one
    escalation per status crossing; a case that moves and stalls
    again earns a new one. An attempt is recorded whether the
    receiver answered (``sent``) or not (``failed``): the ledger is
    the record that the crossing was escalated, once."""

    case_status: str = ""  # the status that breached
    status_since: str = ""  # ISO date the status began — the crossing's identity
    sla_days: int = 0  # the SLA in force when the breach fired
    status_age_days: Optional[int] = None  # the breached status age
    age_days: Optional[int] = None  # the case's total age at the breach
    dispatched_at: str = ""  # ISO timestamp of the dispatch
    dispatch_status: str = ""  # sent | failed


class RecoveryPack(BaseModel):
    """The recovery case for a duplicate whose original was already
    approved: the duplicate evidence (computed), a supplier recovery
    letter and an internal memo — drafted, verified, and guardrail-
    checked through the standard path, then stored for a human to
    send and act on. The agent recovers nothing itself."""

    pack_id: str
    duplicate_invoice_id: str
    original_invoice_id: str = ""  # the approved twin
    supplier_name: str = ""
    invoice_number: str = ""
    currency: str = "USD"
    amount: float = 0.0  # the duplicated amount, from the stored facts
    exposure_status: str = ""  # twin_approved | both_approved
    evidence: list[str] = Field(default_factory=list)
    drafts: Optional[Drafts] = None
    verification: Optional[VerifyResult] = None
    validation: Optional[Validation] = None
    status: str = "ready"  # ready | blocked (guardrails failed)
    created_at: str = ""
    # The case lifecycle: opened on drafting, then walked forward by
    # operators — supplier_contacted → credit_received → closed.
    # ``case_history`` is the audit trail of those transitions.
    case_status: str = "opened"
    case_history: list[RecoveryCaseEvent] = Field(default_factory=list)
    # SLA-breach escalations fired for this case (see
    # RecoveryEscalation): the webhook ledger, deduped per status
    # crossing. Empty until a sweep finds the case breached.
    escalations: list[RecoveryEscalation] = Field(default_factory=list)


class RecoveryTransitionRequest(BaseModel):
    """An operator transition on a recovery case."""

    to_status: str
    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None
    note: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


# ---------------------------------------------------------------------------
# Multi-invoice bundles (one arrival, several invoices)
# ---------------------------------------------------------------------------

class InvoiceBundleInput(BaseModel):
    """One intake arrival carrying several invoices: typically a
    supplier email plus a combined scan. The documents are segmented
    in code by invoice-number anchors (bundles.py); each segment
    becomes its own invoice run. ``purchase_orders`` /
    ``goods_receipts`` are the system records segments reference by
    PO number, so a bundled invoice can still three-way match."""

    bundle_id: str
    supplier_name: str = ""
    supplier_email_text: str = ""
    notes: str = ""
    submitted_at: str = ""  # ISO datetime/date
    currency: str = "USD"
    documents: list[DocumentInput] = Field(default_factory=list)
    purchase_orders: list[PurchaseOrder] = Field(default_factory=list)
    goods_receipts: list[GoodsReceipt] = Field(default_factory=list)


class BundleInvoiceOutcome(BaseModel):
    invoice_id: str
    invoice_number: str = ""
    status: str = ""  # the run's status, or "error"
    outcome: str = ""  # triage outcome, "" on error
    severity: str = ""
    total_amount: Optional[float] = None
    currency: str = "USD"
    error: Optional[str] = None


class BundleResult(BaseModel):
    """The bundle summary: one row per segmented invoice, counts and
    totals computed in code. Each invoice also lives in the store as
    its own run, with its own gate."""

    bundle_id: str
    supplier_name: str = ""
    segment_count: int = 0
    processed_count: int = 0
    outcomes: list[BundleInvoiceOutcome] = Field(default_factory=list)
    outcome_counts: dict[str, int] = Field(default_factory=dict)
    totals_by_currency: dict[str, float] = Field(default_factory=dict)
    note: str = ""


class AnalyzeRequest(BaseModel):
    invoice: InvoiceInput


class DecisionRequest(BaseModel):
    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None
    reason: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer
