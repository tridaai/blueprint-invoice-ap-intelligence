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

Usage:

    python examples/erp_posting.py approved-result.json
    curl localhost:8000/invoices/INV-1001 | python examples/erp_posting.py -
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


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.split("Usage:")[1].strip(), file=sys.stderr)
        return 2
    raw = sys.stdin.read() if argv[1] == "-" else open(argv[1], encoding="utf-8").read()
    result = json.loads(raw)
    try:
        posting = build_posting(result)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(posting, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
