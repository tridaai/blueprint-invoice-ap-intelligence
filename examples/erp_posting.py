"""Example output adapter: an approved invoice -> an ERP posting payload.

This is a worked example of the blueprint's output-routing port —
the same seam the approval webhook uses. It takes the JSON of an
**approved** agent result (exactly what ``GET /invoices/{id}``
returns, and what the webhook payload is built from) and shapes it
into a generic double-entry journal payload an ERP import can
consume:

    Dr  6100-PURCHASES                 lines total (net of tax)
    Dr  1310-INPUT-TAX-RECOVERABLE     tax stated
    Cr  2000-ACCOUNTS-PAYABLE-CONTROL  invoice total

Design rules the example demonstrates:

- **Amounts come from the computed facts, never from prose.** The
  journal reads ``facts.invoice_lines_total`` / ``facts.tax_stated``
  / ``facts.invoice_total_stated`` — the three-way match the code
  computed — not the supplier letter, not the packet text.
- **Only approved results post.** Anything still at the gate, or
  rejected, raises: the human decision is the posting's authority.
- **The journal must balance.** If the stated total does not equal
  lines + tax (beyond a cent), the adapter refuses and reports the
  variance — an unreconciled invoice is an exception, not a posting.
- **Idempotency.** The key derives from the invoice's identity
  (agent id + supplier invoice number + total), so re-running the
  adapter for the same approval produces the same key and the ERP
  can safely de-duplicate a retried import.

It is deliberately not a real ERP integration: no vendor SDK, no
network, no credentials, no posting. Adapt the account codes to
your chart of accounts and hand the payload to your ERP's import —
or POST it onward the way the approval webhook does.

**The receipt loop.** A posting is only half the seam: the ERP's
answer has to come back, or the agent's record and the ledger
drift apart. ``build_receipt`` shapes that answer into the body of
``POST /invoices/{invoice_id}/posting-receipt`` — the endpoint the
service exposes for exactly this:

    {"status": "posted", "external_reference": "<ERP doc number>",
     "idempotency_key": "<the posting's key>", "actor": "<operator>"}

or, when the ERP refused the posting, ``"status": "failed"`` with
``"detail"`` carrying its reason. The service records the receipt
on the invoice (and in the audit export); a posted receipt can
only be replayed with the same idempotency key, never replaced by
a different posting — a double posting is corrected in the ERP,
not papered over here.

Usage:

    python examples/erp_posting.py approved-result.json
    curl localhost:8000/invoices/INV-1001 | python examples/erp_posting.py -

    # The ERP's answer, shaped for the receipt endpoint:
    python examples/erp_posting.py --receipt posting.json posted ERP-DOC-1042
    python examples/erp_posting.py --receipt posting.json failed "" "period 2026-09 is closed"
"""

from __future__ import annotations

import hashlib
import json
import sys

# Account codes are placeholders for a generic chart of accounts —
# map them to yours.
ACCOUNT_PURCHASES = "6100-PURCHASES"
ACCOUNT_INPUT_TAX = "1310-INPUT-TAX-RECOVERABLE"
ACCOUNT_AP_CONTROL = "2000-ACCOUNTS-PAYABLE-CONTROL"

TOLERANCE = 0.01


def _money(value: float) -> float:
    return round(float(value), 2)


def idempotency_key(result: dict) -> str:
    """Stable per approved invoice: the same approval re-run through
    the adapter yields the same key, so the receiving ERP can treat
    a retry as the same posting, not a second one."""
    facts = result.get("facts") or {}
    total = facts.get("invoice_total_stated")
    basis = "|".join(
        [
            str(result.get("invoice_id", "")),
            str(result.get("invoice_number", "")),
            f"{float(total):.2f}" if total is not None else "",
        ]
    )
    return "ap-posting-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def build_posting(result: dict) -> dict:
    """Shape one approved agent result into a journal payload.

    Raises ValueError for anything that must not post: a result that
    is not approved, or figures that do not reconcile.
    """
    if result.get("status") != "approved":
        raise ValueError(
            f"invoice {result.get('invoice_id')} is "
            f"{result.get('status') or 'unknown'} — only an approved "
            "result may become a posting; the approver's decision is "
            "the authority"
        )
    facts = result.get("facts") or {}
    lines_total = facts.get("invoice_lines_total")
    tax = facts.get("tax_stated")
    total = facts.get("invoice_total_stated")
    if lines_total is None or total is None:
        raise ValueError(
            "the result carries no computed totals (unreadable or "
            "unmatched invoice) — nothing to post"
        )
    lines_total = _money(lines_total)
    tax = _money(tax or 0.0)
    total = _money(total)
    if abs(lines_total + tax - total) > TOLERANCE:
        raise ValueError(
            f"figures do not reconcile: lines {lines_total:.2f} + tax "
            f"{tax:.2f} != stated total {total:.2f} — an unreconciled "
            "invoice is an exception, not a posting"
        )
    currency = facts.get("currency") or "USD"
    supplier = result.get("supplier_name", "")
    reference = result.get("invoice_number", "")

    journal = [
        {
            "account": ACCOUNT_PURCHASES,
            "debit": lines_total,
            "credit": 0.0,
            "currency": currency,
            "memo": f"{supplier} invoice {reference} — goods/services (net)",
        }
    ]
    if tax:
        journal.append(
            {
                "account": ACCOUNT_INPUT_TAX,
                "debit": tax,
                "credit": 0.0,
                "currency": currency,
                "memo": f"{supplier} invoice {reference} — input tax",
            }
        )
    journal.append(
        {
            "account": ACCOUNT_AP_CONTROL,
            "debit": 0.0,
            "credit": total,
            "currency": currency,
            "memo": f"{supplier} invoice {reference} — payable",
        }
    )
    debits = sum(line["debit"] for line in journal)
    credits = sum(line["credit"] for line in journal)
    assert abs(debits - credits) <= TOLERANCE, "journal must balance"

    return {
        "document_type": "vendor_invoice_posting",
        "external_reference": reference,
        "source_invoice_id": result.get("invoice_id"),
        "supplier": supplier,
        "currency": currency,
        "decided_by": result.get("decided_by"),
        "idempotency_key": idempotency_key(result),
        "journal_lines": journal,
        "total_debit": _money(debits),
        "total_credit": _money(credits),
        "balanced": True,
        # The ERP assigns the posting date on import; the payload
        # carries the decision's provenance instead.
        "source": "trida-blueprint-invoice-ap-intelligence (example adapter)",
    }


RECEIPT_OUTCOMES = ("posted", "failed")


def build_receipt(
    posting: dict,
    *,
    outcome: str,
    external_reference: str = "",
    detail: str = "",
    recorded_by: str = "",
) -> dict:
    """Shape the ERP's outcome for one posting into a receipt: the
    body of ``POST /invoices/{invoice_id}/posting-receipt``.

    The receipt inherits the posting's idempotency key, so the
    service can tie the answer to the exact posting it answers (and
    refuse a second, different posting for the same invoice). A
    posted receipt must name the ERP's document number; a failed
    one must carry its reason — the same guards the endpoint
    enforces, applied here so the example fails fast and close to
    the cause.
    """
    if outcome not in RECEIPT_OUTCOMES:
        raise ValueError(
            f"unknown posting outcome: {outcome!r} — a receipt is "
            "'posted' or 'failed'"
        )
    if outcome == "posted" and not external_reference:
        raise ValueError(
            "a posted receipt needs the ERP's document number as "
            "external_reference — that number is what closes the loop"
        )
    if outcome == "failed" and not detail:
        raise ValueError(
            "a failed receipt needs the ERP's reason as detail"
        )
    return {
        "invoice_id": posting.get("source_invoice_id"),
        "status": outcome,
        "external_reference": external_reference,
        "idempotency_key": posting.get("idempotency_key", ""),
        "detail": detail,
        "actor": recorded_by,
        "source": "trida-blueprint-invoice-ap-intelligence (example adapter)",
    }


def _load_json(path: str) -> dict:
    raw = sys.stdin.read() if path == "-" else open(path, encoding="utf-8").read()
    return json.loads(raw)


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "--receipt":
        # --receipt POSTING_JSON OUTCOME [EXTERNAL_REFERENCE] [DETAIL] [RECORDED_BY]
        if len(argv) < 4:
            print(__doc__.split("Usage:")[1].strip(), file=sys.stderr)
            return 2
        posting = _load_json(argv[2])
        try:
            receipt = build_receipt(
                posting,
                outcome=argv[3],
                external_reference=argv[4] if len(argv) > 4 else "",
                detail=argv[5] if len(argv) > 5 else "",
                recorded_by=argv[6] if len(argv) > 6 else "",
            )
        except ValueError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(receipt, indent=2))
        return 0
    if len(argv) != 2:
        print(__doc__.split("Usage:")[1].strip(), file=sys.stderr)
        return 2
    result = _load_json(argv[1])
    try:
        posting = build_posting(result)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(posting, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
