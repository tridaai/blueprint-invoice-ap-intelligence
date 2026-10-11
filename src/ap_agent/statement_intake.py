"""Statement document intake — a supplier statement that arrives as
a document is read into the same ``SupplierStatement`` the structured
path reconciles.

Statements land in AP inboxes as documents, not JSON. This module is
the document path: the OCR port reads the document (confidence
included), the vision port classifies it, and deterministic code
parses the text layer — header (supplier, statement date, claimed
total) and the tabular lines (invoice number, date, amount, credit
lines marked as such) — into the reconciliation module's input
shape. The model does none of the parsing: in provider mode the
ports' reads stand behind the same shapes the offline engines
return, and the parse is code either way.

The discipline mirrors the invoice path's:

- **Classification first** (the vision port's verdict, AP-DOC-02's
  shape): a document that verifies as an invoice, purchase order,
  goods receipt, credit note or email is refused by the
  reconciliation path — invoices are not statements, and statements
  are not payable — with the reason carried on the reconciliation
  record, never a silent drop. A document no statement can be read
  from is refused the same way.
- **Every figure carries the confidence of its read.** A line whose
  read falls below the OCR trust threshold is flagged on the
  reconciliation record (``low_confidence``) — reconciled as read,
  but never silently trusted.
- **Credit lines are read as credits**: a row the document marks
  as a credit (the word ``credit``/``CR``, or a negative /
  parenthesised amount) becomes a credit line with a signed
  amount, reconciled as a credit — never matched against an
  invoice as if the supplier were billing it again.
- Amount tokens an OCR read garbled with a confusable character
  (``75.0O`` for ``75.00``) still parse — at a reduced confidence
  that flags the line. A token that does not parse at all keeps
  the line, amount unknown, at the floor confidence.
"""

from __future__ import annotations

import re

from .ocr import LOW_CONFIDENCE_THRESHOLD, OfflineOcrEngine
from .schemas import (
    DocumentInput,
    FieldExtraction,
    InvoiceInput,
    StatementExtraction,
    StatementLine,
    StatementLineRead,
    SupplierStatement,
    normalize_doc_type,
)
from .vision import run_vision

# Document types the vision verdict can return that are NOT a
# statement: each is refused by name, the way triage refuses a
# non-invoice (AP-DOC-02).
_NON_STATEMENT_TYPES = {
    "invoice",
    "purchase_order",
    "goods_receipt",
    "credit_note",
    "supplier_email",
}
_STATEMENT_TYPES = {"remittance_note", "statement"}

_ROW_RE = re.compile(
    r"^(?P<number>[A-Za-z0-9][A-Za-z0-9\-/\.]*)\s+"
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?:(?P<marker>credit|cr)\s+)?"
    r"(?P<amount>[^\s]+)"
    r"(?:\s+(?P<trailing>credit|cr))?\s*$",
    re.IGNORECASE,
)
_TOTAL_RE = re.compile(
    r"^\s*(total|balance\s+due|amount\s+due|total\s+due)\b[^0-9\-]*"
    r"(?P<amount>\(?-?[\d,]+\.\d{2}\)?)",
    re.IGNORECASE,
)
_STATEMENT_ID_RE = re.compile(
    r"statement\s*(?:no\.?|number|#|:)?\s*([A-Za-z0-9][\w\-.]*)",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(r"currency\s*[:\-]?\s*([A-Za-z]{3})", re.IGNORECASE)
_DATE_LABEL_RE = re.compile(
    r"(?:statement\s+date|date)\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE
)
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_CONFUSABLES = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5"})
_CLEAN_AMOUNT_RE = re.compile(r"\d+(\.\d{1,2})?")

# Parse confidences: a cleanly shaped read, a confusable-repaired
# read, and the floor for a token that does not parse at all. The
# line's confidence is the lower of the parse confidence and the
# OCR read's own confidence, so a poor scan flags every line.
_PARSE_CLEAN = 0.97
_PARSE_REPAIRED = 0.45
_PARSE_FAILED = 0.3


def _parse_amount(token: str) -> tuple[float | None, float]:
    """(value, parse confidence) for one amount token. Parenthesised
    and minus-signed amounts are negative (a credit, in statement
    terms). A token that is numeric only after confusable repair
    parses at the reduced confidence."""
    text = (
        token.strip()
        .replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .replace(" ", "")
    )
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative, text = True, text[1:-1]
    if text.startswith("-"):
        negative, text = True, text[1:]
    text = text.replace(",", "")
    if _CLEAN_AMOUNT_RE.fullmatch(text):
        value = float(text)
        return (-value if negative else value), _PARSE_CLEAN
    mapped = text.translate(_CONFUSABLES)
    if mapped != text and _CLEAN_AMOUNT_RE.fullmatch(mapped):
        value = float(mapped)
        return (-value if negative else value), _PARSE_REPAIRED
    return None, _PARSE_FAILED


def _parse_lines(
    text: str, ocr_confidence: float
) -> tuple[list[StatementLineRead], int]:
    """The tabular lines of the statement text, and the count of
    digit-bearing rows that could not be read as lines."""
    reads: list[StatementLineRead] = []
    skipped = 0
    for raw_line in text.splitlines():
        row = raw_line.strip()
        if not row:
            continue
        match = _ROW_RE.match(row)
        if match is None:
            lowered = row.lower()
            if any(ch.isdigit() for ch in row) and not (
                "statement" in lowered
                or _TOTAL_RE.match(row)
                or _DATE_LABEL_RE.search(row)
                or _CURRENCY_RE.search(row)
            ):
                skipped += 1
            continue
        value, parse_confidence = _parse_amount(match.group("amount"))
        marked_credit = bool(match.group("marker") or match.group("trailing"))
        is_credit = marked_credit or (value is not None and value < 0)
        amount = value
        if marked_credit and value is not None:
            amount = -abs(value)  # a marked credit signs negative
        confidence = round(min(parse_confidence, ocr_confidence), 3)
        reads.append(
            StatementLineRead(
                invoice_number=match.group("number"),
                amount=amount,
                date=match.group("date"),
                is_credit=is_credit,
                confidence=confidence,
                low_confidence=confidence < LOW_CONFIDENCE_THRESHOLD,
                raw=row,
            )
        )
    return reads, skipped


def _header_field(
    field: str, value: str | None, confidence: float, status: str
) -> FieldExtraction:
    return FieldExtraction(
        field=field, value=value, confidence=confidence, status=status
    )


def _parse_header(
    document: DocumentInput, text: str, first_row_at: int
) -> tuple[dict, list[FieldExtraction], list[str]]:
    """The statement header: supplier, statement id, statement date,
    currency, claimed total — provided fields first (the submitting
    system's word, claimed with no confidence), the document text
    second (extracted, with the read's confidence)."""
    provided = document.fields or {}
    notes: list[str] = []
    values: dict = {}
    header: list[FieldExtraction] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    supplier = provided.get("supplier_name")
    if supplier:
        values["supplier_name"] = supplier
        header.append(
            _header_field("supplier_name", supplier, 0.0, "source_provided")
        )
    else:
        candidate = next(
            (
                ln
                for ln in lines
                if "statement" not in ln.lower()
                and ":" not in ln  # label lines (Date: …, Currency: …) are not the supplier
                and not _ROW_RE.match(ln)
                and not _TOTAL_RE.match(ln)
            ),
            None,
        )
        if candidate:
            values["supplier_name"] = candidate
            header.append(
                _header_field("supplier_name", candidate, 0.9, "extracted_only")
            )
        else:
            header.append(
                _header_field("supplier_name", None, 0.0, "missing_in_text")
            )

    statement_id = provided.get("statement_id")
    if statement_id:
        values["statement_id"] = statement_id
        header.append(
            _header_field("statement_id", statement_id, 0.0, "source_provided")
        )
    else:
        found = None
        for match in _STATEMENT_ID_RE.finditer(text):
            if any(ch.isdigit() for ch in match.group(1)):
                found = match.group(1)
                break
        if found:
            values["statement_id"] = found
            header.append(
                _header_field("statement_id", found, 0.95, "extracted_only")
            )
        elif document.document_id:
            values["statement_id"] = document.document_id
            header.append(
                _header_field(
                    "statement_id", document.document_id, 0.0, "source_provided"
                )
            )
            notes.append(
                "the document names no statement id — its document id "
                "stands as the statement id"
            )

    statement_date = provided.get("statement_date")
    if statement_date:
        values["statement_date"] = statement_date
        header.append(
            _header_field("statement_date", statement_date, 0.0, "source_provided")
        )
    else:
        labelled = _DATE_LABEL_RE.search(text)
        header_region = text[:first_row_at] if first_row_at > 0 else text
        unlabelled = _ISO_DATE_RE.search(header_region)
        found_date = (
            labelled.group(1)
            if labelled
            else (unlabelled.group(0) if unlabelled else None)
        )
        if found_date:
            values["statement_date"] = found_date
            header.append(
                _header_field("statement_date", found_date, 0.9, "extracted_only")
            )
        else:
            header.append(
                _header_field("statement_date", None, 0.0, "missing_in_text")
            )

    currency = provided.get("currency")
    if currency:
        values["currency"] = currency.upper()
        header.append(
            _header_field("currency", currency.upper(), 0.0, "source_provided")
        )
    else:
        stated = _CURRENCY_RE.search(text)
        if stated:
            values["currency"] = stated.group(1).upper()
            header.append(
                _header_field(
                    "currency", stated.group(1).upper(), 0.9, "extracted_only"
                )
            )
        else:
            symbol = (
                "EUR" if "€" in text else "GBP" if "£" in text else None
            )
            if symbol:
                values["currency"] = symbol
                header.append(
                    _header_field("currency", symbol, 0.7, "extracted_only")
                )
            else:
                header.append(
                    _header_field("currency", None, 0.0, "missing_in_text")
                )
                notes.append(
                    "the document states no currency — the statement "
                    "carries the default (USD)"
                )

    claimed: tuple[float | None, float] | None = None
    for ln in lines:
        if "subtotal" in ln.lower():
            continue
        total_match = _TOTAL_RE.match(ln)
        if total_match:
            claimed = _parse_amount(total_match.group("amount"))
    if claimed is not None and claimed[0] is not None:
        values["claimed_total"] = claimed[0]
        header.append(
            _header_field(
                "claimed_total", f"{claimed[0]:.2f}", claimed[1], "extracted_only"
            )
        )
    else:
        header.append(
            _header_field("claimed_total", None, 0.0, "missing_in_text")
        )
    return values, header, notes


def extract_statement_document(
    document: DocumentInput, *, ocr_engine, vision_engine
) -> tuple[SupplierStatement | None, StatementExtraction]:
    """Read one document into a ``SupplierStatement`` — or refuse it.

    Returns ``(statement, extraction)`` when the document verifies
    as a statement and lines could be read from it, and
    ``(None, extraction)`` with ``extraction.refusal_reason`` set
    when it does not. The extraction is returned either way: the
    read is the record."""
    extraction = StatementExtraction(document_id=document.document_id)
    offline = OfflineOcrEngine()
    try:
        ocr_result = ocr_engine.read(document)
    except Exception as exc:  # OCR is evidence; degrade like run_ocr does
        ocr_result = offline.read(document)
        extraction.notes.append(
            f"OCR engine failed ({exc}) — offline text layer used"
        )
    extraction.ocr_engine = ocr_result.engine
    extraction.ocr_confidence = ocr_result.mean_confidence

    text = document.text_layer
    probe = InvoiceInput(
        invoice_id=document.document_id or "STMT-DOC",
        documents=[document],
    )
    verdict = run_vision(probe, [ocr_result], vision_engine)
    detected = normalize_doc_type(verdict.document_type_detected or "")
    extraction.detected_type = detected or verdict.document_type_detected
    extraction.classification_source = verdict.source

    if detected in _NON_STATEMENT_TYPES:
        extraction.refusal_reason = (
            f"The document verified as '{detected}', not a supplier "
            "statement — it cannot enter statement reconciliation "
            "(AP-DOC-02: statements and remittance advices are not "
            "payable, and an invoice is not a statement)."
        )
        return None, extraction

    reads, skipped = _parse_lines(text, ocr_result.mean_confidence)
    extraction.lines = reads
    extraction.skipped_rows = skipped
    if skipped:
        extraction.notes.append(
            f"{skipped} text row(s) could not be read as statement "
            "lines and were skipped — they are not in the figures"
        )
    if not reads:
        extraction.refusal_reason = (
            "No statement lines could be read from the document — "
            "nothing was reconciled. The document may not be a "
            "statement, or its text layer may be unreadable."
        )
        return None, extraction
    if detected not in _STATEMENT_TYPES and "statement" not in text.lower():
        extraction.refusal_reason = (
            f"The document verified as '{detected or 'unknown'}' and "
            "its text carries no statement header — it could not be "
            "verified as a supplier statement, so it was not reconciled."
        )
        return None, extraction

    first_row_at = min(
        (text.find(read.raw) for read in reads if read.raw), default=-1
    )
    values, header, notes = _parse_header(document, text, first_row_at)
    extraction.header = header
    extraction.notes.extend(notes)
    extraction.accepted = True

    statement = SupplierStatement(
        statement_id=values.get("statement_id") or document.document_id,
        supplier_name=values.get("supplier_name") or "",
        statement_date=values.get("statement_date") or "",
        currency=values.get("currency") or "USD",
        claimed_total=values.get("claimed_total"),
        lines=[
            StatementLine(
                invoice_number=read.invoice_number,
                amount=read.amount,
                date=read.date,
                is_credit=read.is_credit,
                confidence=read.confidence,
            )
            for read in reads
        ],
    )
    return statement, extraction
