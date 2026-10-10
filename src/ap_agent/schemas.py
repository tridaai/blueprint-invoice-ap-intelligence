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


class AnalyzeRequest(BaseModel):
    invoice: InvoiceInput


class DecisionRequest(BaseModel):
    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None
    reason: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer
