"""Payment-run hand-off file: the confirmed run as a bank-ready CSV.

A confirmed payment run is a pay list. The customer's banking or
ERP system is what executes it — and the adapter between this agent
and that system is a file, rendered deterministically from the
confirmed run and stored through the object-store port
(``payment-runs/<run_id>/handoff.csv``). Generating it changes no
payment state: the run stays ``confirmed``, no invoice changes
state, and no byte leaves the customer's own object storage until a
human downloads the file and hands it to their bank/ERP.

**Layout v1** (one header row, one row per run line, one totals
row; amounts are plain decimals with two places, no thousands
separators; dates ISO):

| column | content |
|---|---|
| ``run_id`` | the confirmed run's id (the totals row repeats it) |
| ``line_no`` | 1-based line number; blank on the totals row |
| ``invoice_id`` | the agent's invoice id; ``TOTAL`` on the totals row |
| ``supplier_name`` | as recorded on the invoice |
| ``invoice_number`` | the supplier's invoice number |
| ``due_date`` | ISO date, blank when unknown |
| ``currency`` | the run's currency |
| ``gross_amount`` | the approved invoice total |
| ``discount_amount`` | capturable early-payment discount, else 0.00 |
| ``net_amount`` | gross − discount: the amount to pay |

The renderer is pure — same run in, same bytes out — so the sha256
recorded on the run is a stable content hash of the file the bank
receives.
"""

from __future__ import annotations

import csv
import io

from .schemas import PaymentRun

HANDOFF_COLUMNS = [
    "run_id",
    "line_no",
    "invoice_id",
    "supplier_name",
    "invoice_number",
    "due_date",
    "currency",
    "gross_amount",
    "discount_amount",
    "net_amount",
]

HANDOFF_LAYOUT_VERSION = "v1"


def handoff_key(run_id: str) -> str:
    """The object-store key a run's hand-off file lives under."""
    return f"payment-runs/{run_id}/handoff.csv"


def _money(value: float) -> str:
    return f"{value:.2f}"


def render_handoff_csv(run: PaymentRun) -> bytes:
    """Render a confirmed run as the hand-off CSV (layout v1).

    Pure and deterministic: the totals row re-states the run's
    code-computed totals, and a reader can verify the file by
    summing the line rows."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(HANDOFF_COLUMNS)
    for index, line in enumerate(run.lines, start=1):
        writer.writerow(
            [
                run.run_id,
                index,
                line.invoice_id,
                line.supplier_name,
                line.invoice_number,
                line.due_date,
                run.currency,
                _money(line.gross_amount),
                _money(line.discount_amount),
                _money(line.net_amount),
            ]
        )
    writer.writerow(
        [
            run.run_id,
            "",
            "TOTAL",
            f"{run.invoice_count} invoice(s)",
            "",
            "",
            run.currency,
            _money(run.total_gross),
            _money(run.total_discount),
            _money(run.total_net),
        ]
    )
    return buffer.getvalue().encode("utf-8")
