"""Document extraction node logic.

In provider mode the LLM reads each document's text and extracts the
typed fields (``schemas.EXTRACTED_FIELDS``) with a per-field
confidence — structured output, via ``ModelBackend.extract_document_fields``.
Deterministic code then cross-checks every extracted value against the
field values the supplier's system provided:

- ``match`` — extracted equals provided.
- ``mismatch`` — both present, different values (a data-quality signal
  the approver sees; the computed facts in ``facts.py`` count them).
- ``missing_in_text`` — provided, but the extractor could not find it
  in the document text.
- ``extracted_only`` — found in the text, absent from the provided fields.

In the default (mock) mode no extraction runs: fields pass through as
``source_provided`` with no confidence claimed. Extraction never fails
a run — a provider error or unusable reply falls back to the provided
fields for that document.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from .schemas import (
    EXTRACTED_FIELDS,
    DocumentExtraction,
    DocumentInput,
    FieldExtraction,
    InvoiceInput,
)


def _provided_extraction(document: DocumentInput) -> DocumentExtraction:
    return DocumentExtraction(
        document_id=document.document_id,
        doc_type=document.doc_type,
        source="provided",
        fields=[
            FieldExtraction(
                field=field,
                value=document.fields.get(field),
                confidence=0.0,
                status="source_provided",
            )
            for field in EXTRACTED_FIELDS
            if document.fields.get(field) is not None
        ],
    )


def _as_number(value) -> Decimal | None:
    """``value`` as an exact number when it reads as one ("1,180.00",
    "1180.0", 1180), else None. Currency codes and symbols around the
    figure are tolerated; anything else is text."""
    text = str(value).strip()
    for token in ("$", "€", "£", "USD", "EUR", "GBP"):
        text = text.replace(token, "")
    text = text.replace(",", "").strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _same_value(provided, extracted) -> bool:
    """Provided and extracted agree: equal as numbers when both read
    as numbers (1000.00 and 1000.0 are the same subtotal), otherwise
    equal as text, ignoring surrounding space and case."""
    left, right = _as_number(provided), _as_number(extracted)
    if left is not None and right is not None:
        return left == right
    return str(provided).strip().casefold() == str(extracted).strip().casefold()


def _cross_check(document: DocumentInput, extracted: dict[str, dict]) -> DocumentExtraction:
    fields: list[FieldExtraction] = []
    for field in EXTRACTED_FIELDS:
        provided = document.fields.get(field)
        entry = extracted.get(field) or {"value": None, "confidence": 0.0}
        value = entry.get("value")
        confidence = entry.get("confidence") or 0.0
        if provided is None and value is None:
            continue
        if value is None:
            status, shown_value, confidence = "missing_in_text", provided, 0.0
        elif provided is None:
            status, shown_value = "extracted_only", value
        elif _same_value(provided, value):
            status, shown_value = "match", value
        else:
            status, shown_value = "mismatch", value
        fields.append(
            FieldExtraction(
                field=field,
                value=None if shown_value is None else str(shown_value),
                confidence=float(confidence),
                status=status,
            )
        )
    return DocumentExtraction(
        document_id=document.document_id,
        doc_type=document.doc_type,
        source="llm",
        fields=fields,
    )


def extract_documents(invoice: InvoiceInput, backend) -> list[DocumentExtraction]:
    """Extract + cross-check every document on the invoice.

    ``backend`` is the active model backend; only real LLM backends
    expose ``extract_document_fields``, so the default mode is a pure
    pass-through by construction.
    """
    extract = getattr(backend, "extract_document_fields", None)
    results: list[DocumentExtraction] = []
    for document in invoice.documents:
        if extract is None or not document.text_layer.strip():
            results.append(_provided_extraction(document))
            continue
        try:
            extracted = extract(
                document.doc_type,
                document.document_id,
                document.text_layer,
                list(EXTRACTED_FIELDS),
            )
        except Exception as exc:  # extraction is evidence, never a run failure
            fallback = _provided_extraction(document)
            fallback.note = (
                f"LLM extraction failed ({exc}) — provided fields used instead"
            )
            results.append(fallback)
            continue
        if extracted is None:
            fallback = _provided_extraction(document)
            fallback.note = "LLM extraction reply was unusable — provided fields used instead"
            results.append(fallback)
        else:
            results.append(_cross_check(document, extracted))
    return results


def extraction_discrepancies(extractions: list[DocumentExtraction]) -> list[str]:
    """Human-readable discrepancy lines for the trace and the diagnosis."""
    lines: list[str] = []
    for extraction in extractions:
        for field in extraction.fields:
            if field.status == "mismatch":
                lines.append(
                    f"{extraction.document_id or extraction.doc_type} field "
                    f"'{field.field}': document text value {field.value!r} "
                    "differs from the provided value"
                )
            elif field.status == "missing_in_text":
                lines.append(
                    f"{extraction.document_id or extraction.doc_type} field "
                    f"'{field.field}': provided value {field.value!r} not found "
                    "in the document text"
                )
    return lines
