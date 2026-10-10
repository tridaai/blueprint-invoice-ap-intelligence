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
    message: str = ""


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
    autonomy: AutonomyRecommendation = Field(default_factory=AutonomyRecommendation)
    telemetry: Optional[RunTelemetry] = None
    trace: list[TraceStep] = Field(default_factory=list)
    decision: Optional[str] = None  # approved | rejected
    decided_by: Optional[str] = None
    decision_reason: Optional[str] = None
    dispatch_status: Optional[str] = None  # sent | failed | None
    external_action_taken: bool = False
    error: Optional[str] = None


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

    status: str  # proposed | approved | rejected | applied
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
    was made against, so the approver sees old vs new."""

    request_id: str
    supplier_name: str
    proposed: SupplierMasterRecord
    current: Optional[SupplierMasterRecord] = None  # None = new supplier
    source: str = "operator"  # operator | document
    source_invoice_id: str = ""  # set when source == "document"
    reason: str = ""
    proposed_by: str = ""
    proposed_at: str = ""
    status: str = "pending"  # pending | approved | rejected
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
    # Four-eyes on large runs: a run whose net total exceeded
    # PAYMENT_RUN_FOUR_EYES_THRESHOLD when proposed needs a second
    # confirmation — from a different person than the first — before
    # the hand-off file can be generated.
    four_eyes_required: bool = False
    second_confirmed_by: Optional[str] = None
    second_confirmed_at: Optional[str] = None
    note: str = ""
    # The execution hand-off: the confirmed run rendered as a
    # bank-ready CSV and stored through the object-store port. The
    # file is the routing adapter to the customer's banking / ERP
    # system — generating it changes no payment state. These fields
    # are the export's receipt: where the file lives, who generated
    # it, and the sha256 of the exact bytes stored.
    handoff_key: Optional[str] = None
    handoff_sha256: Optional[str] = None
    handoff_generated_at: Optional[str] = None
    handoff_generated_by: Optional[str] = None


class PaymentRunRequest(BaseModel):
    window_days: int = 7
    as_of: str = ""  # ISO date the windows are anchored on; "" = today


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
