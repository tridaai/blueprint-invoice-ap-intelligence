"""Payment-run hand-off files: the confirmed run as a bank-ready file.

A confirmed payment run is a pay list. The customer's banking or
ERP system is what executes it — and the adapter between this agent
and that system is a file, rendered deterministically from the
confirmed run and stored through the object-store port.
Generating it changes no payment state: the run stays
``confirmed``, no invoice changes state, and no byte leaves the
customer's own object storage until a human downloads the file
and hands it to their bank/ERP.

Two serializations sit behind the one hand-off seam (the full
mapping lives in ``docs/handoff-formats.md``):

- ``csv`` — layout v1, below: the plain pay list most ERPs import.
- ``pain001`` — an ISO 20022 pain.001.001.09 customer
  credit-transfer initiation document: one ``PmtInf`` for the run,
  one ``CdtTrfTxInf`` per line, amounts at the export-time net
  figures, remittance information carrying the supplier invoice
  number. Bank-account elements the agent never holds (the
  debtor's account, creditor accounts — governed master data, not
  document data) are emitted with documented completion markers
  for the importing system to fill from its own records.

**CSV layout v1** (one header row, one row per run line, one totals
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

The renderers are pure — same run in, same bytes out — so the
sha256 recorded on the run is a stable content hash of the file
the bank receives.

The run they render is the export-time run: before rendering, the
service re-verifies each member's discount (``paymentruns.
discount_drift``) and adjusts any member whose window closed since
confirmation to its full gross amount, itemising the drift on the
run record. The file therefore never nets off a discount the bank
run can no longer take — in either format.
"""

from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET

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

HANDOFF_FORMAT_CSV = "csv"
HANDOFF_FORMAT_PAIN001 = "pain001"
HANDOFF_FORMATS = (HANDOFF_FORMAT_CSV, HANDOFF_FORMAT_PAIN001)

PAIN001_NAMESPACE = "urn:iso:std:iso:20022:tech:xsd:pain.001.001.09"

# Completion markers for the bank-account elements the agent never
# holds: the importing system fills them from its own records at
# import (see docs/handoff-formats.md).
DEBTOR_ACCOUNT_MARKER = "DEBTOR-ACCOUNT-AT-IMPORT"
DEBTOR_AGENT_MARKER = "DEBTOR-AGENT-AT-IMPORT"
CREDITOR_ACCOUNT_MARKER = "SUPPLIER-MASTER-AT-IMPORT"


def handoff_key(run_id: str, format: str = HANDOFF_FORMAT_CSV) -> str:
    """The object-store key a run's hand-off file lives under."""
    if format == HANDOFF_FORMAT_PAIN001:
        return f"payment-runs/{run_id}/handoff.pain001.xml"
    return f"payment-runs/{run_id}/handoff.csv"


def handoff_content_type(format: str) -> str:
    return "application/xml" if format == HANDOFF_FORMAT_PAIN001 else "text/csv"


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


def _el(parent: ET.Element, tag: str, text: str | None = None) -> ET.Element:
    element = ET.SubElement(parent, f"{{{PAIN001_NAMESPACE}}}{tag}")
    if text is not None:
        element.text = text
    return element


def render_handoff_pain001(run: PaymentRun) -> bytes:
    """Render a confirmed run as an ISO 20022 pain.001.001.09
    customer credit-transfer initiation document.

    The mapping (element by element in ``docs/handoff-formats.md``):
    the group header carries the run id as the message id and the
    run's confirmation time as the creation time — deterministic
    for a given run, like the CSV. One ``PmtInf`` covers the whole
    run (single currency, single requested execution date: the
    run's window end); each line becomes one ``CdtTrfTxInf`` at its
    export-time *net* amount, with the supplier invoice number in
    the remittance information and the agent's invoice id as the
    end-to-end id.

    Bank-account elements are the one place the file is explicit
    about what it does not know: the agent holds no bank accounts
    (creditor details are governed supplier-master data; the
    debtor's account belongs to the customer's bank portal), so
    those elements carry the documented ``*-AT-IMPORT`` markers
    for the importing system to complete from its own records.
    """
    ET.register_namespace("", PAIN001_NAMESPACE)
    document = ET.Element(f"{{{PAIN001_NAMESPACE}}}Document")
    initiation = _el(document, "CstmrCdtTrfInitn")
    header = _el(initiation, "GrpHdr")
    _el(header, "MsgId", run.run_id)
    _el(header, "CreDtTm", run.confirmed_at or f"{run.window_start}T00:00:00")
    _el(header, "NbOfTxs", str(run.invoice_count))
    _el(header, "CtrlSum", _money(run.total_net))
    party = _el(header, "InitgPty")
    _el(party, "Nm", "Trida AI Blueprint - AP Agent")

    payment = _el(initiation, "PmtInf")
    _el(payment, "PmtInfId", run.run_id)
    _el(payment, "PmtMtd", "TRF")
    _el(payment, "NbOfTxs", str(run.invoice_count))
    _el(payment, "CtrlSum", _money(run.total_net))
    _el(payment, "ReqExctnDt", run.window_end)
    debtor = _el(payment, "Dbtr")
    _el(debtor, "Nm", "Debtor account holder (completed at import)")
    account = _el(payment, "DbtrAcct")
    other = _el(_el(account, "Id"), "Othr")
    _el(other, "Id", DEBTOR_ACCOUNT_MARKER)
    agent = _el(payment, "DbtrAgt")
    other = _el(_el(_el(agent, "FinInstnId"), "Othr"), "Id")
    other.text = DEBTOR_AGENT_MARKER
    _el(payment, "ChrgBr", "SLEV")

    for line in run.lines:
        tx = _el(payment, "CdtTrfTxInf")
        pmt_id = _el(tx, "PmtId")
        _el(pmt_id, "EndToEndId", line.invoice_id)
        amount = _el(tx, "Amt")
        instructed = _el(amount, "InstdAmt", _money(line.net_amount))
        instructed.set("Ccy", run.currency)
        creditor = _el(tx, "Cdtr")
        _el(creditor, "Nm", line.supplier_name)
        account = _el(tx, "CdtrAcct")
        other = _el(_el(account, "Id"), "Othr")
        _el(other, "Id", CREDITOR_ACCOUNT_MARKER)
        remittance = _el(tx, "RmtInf")
        _el(
            remittance,
            "Ustrd",
            f"Invoice {line.invoice_number} ({line.invoice_id})"[:140],
        )

    tree = ET.ElementTree(document)
    ET.indent(tree, space="  ")
    buffer = io.BytesIO()
    tree.write(buffer, encoding="utf-8", xml_declaration=True)
    return buffer.getvalue()


def render_handoff(run: PaymentRun, format: str = HANDOFF_FORMAT_CSV) -> bytes:
    """Render a confirmed run in one of the hand-off formats —
    the one seam both serializations sit behind."""
    if format == HANDOFF_FORMAT_PAIN001:
        return render_handoff_pain001(run)
    if format == HANDOFF_FORMAT_CSV:
        return render_handoff_csv(run)
    raise ValueError(
        f"Unknown hand-off format {format!r} — expected one of {HANDOFF_FORMATS}"
    )
