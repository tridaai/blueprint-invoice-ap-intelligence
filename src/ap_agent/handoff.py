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

- ``csv`` — layout v2, below: the plain pay list most ERPs import.
- ``pain001`` — an ISO 20022 pain.001.001.09 customer
  credit-transfer initiation document: one ``PmtInf`` for the run,
  one ``CdtTrfTxInf`` per line, amounts at the export-time net
  figures, remittance information carrying the supplier invoice
  number.

Creditor accounts are completed at export from the governed
supplier master — the one bank detail the system may honestly
fill, because the master is where approved bank details live:
where (and only where) the master holds bank details for a
line's supplier, the export emits them (the IBAN where recorded,
else the account number, plus the sort code as the creditor
agent in pain.001); everywhere else the creditor account carries
the documented ``SUPPLIER-MASTER-AT-IMPORT`` marker for the
importing system to complete. The debtor side always carries
its marker — it is the customer's own account, not master data.
The run record's export receipt (``handoff_creditor_entries``)
says, per member, which way it went — with masked labels only;
the full identifiers live in the master and in the file.

**CSV layout v2** (one header row, one row per run line, one totals
row; amounts are plain decimals with two places, no thousands
separators; dates ISO; v2 appends the two creditor columns to v1):

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
| ``creditor_account`` | the master's IBAN/account for the supplier, else the completion marker |
| ``creditor_account_source`` | ``supplier_master`` when completed, else ``at_import`` |

**Batch booking** (a governing hand-off profile's say, never a
renderer default): some banks book per value date, so when the
profile calls for it the pain.001 document groups the run's
lines into one ``PmtInf`` per requested execution date with
``BtchBookg`` true, instead of one ``PmtInf`` for the whole run.
Each line's group date is the later of its own due date (the
run's window end when it carries none) and the profile's
lead-days floor (export date + lead days) — the lead buys the
bank its processing time per group, and no group executes
before its lines fall due. Which date governs is therefore
stated per group, not per file. The CSV stays the flat list —
batch booking is a pain.001 shape. The creditor completion and
the export-time drift re-verification are computed per member
before rendering, so every gate that holds for the file holds
inside each ``PmtInf``: a line lands in exactly one group,
carrying its own completion (or marker), at its export-time net.

The renderers are pure — same run in, same bytes out — so the
sha256 recorded on the run is a stable content hash of the file
the bank receives.

**Bank feedback** closes the batch-booking loop: the bank
answers a batch-booked file per ``PmtInf`` group, not per file,
and the answers are recorded on the run
(``service.record_handoff_group_receipts``) — one receipt per
group, its members stamped from the export receipt this module
writes, so each member carries its group's fate. A rejected
group's later answer supersedes the first; an accepted group is
final. The reads here (:func:`batch_group_members`,
:func:`current_group_receipts`, :func:`group_receipt_by_member`)
derive everything from the run record — the group membership
from the export receipt, never recomputed from the lines,
because the bank answered the file that was sent.

A group the bank **rejected** can be re-presented alone
(``service.reexport_handoff_group``): a single-``PmtInf``
pain.001 carrying the group's own id and only its members,
drift re-verified as of the re-export date, stored under
:func:`handoff_group_reexport_key` and stamped on the run as
a :class:`HandoffGroupReexport` with the bank's rejection
reason carried onto the record. The re-export resets the
group's receipts — the rejected answer answered the first
file, and stands superseded — so the group reads unanswered
against the new file until the bank answers again.

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

from .matching import _parse_date
from .schemas import (
    HandoffCreditorEntry,
    HandoffGroupReceipt,
    PaymentRun,
    PaymentRunLine,
)
from .supplier_master import _iban_norm, _masked

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
    "creditor_account",
    "creditor_account_source",
]

HANDOFF_LAYOUT_VERSION = "v2"

HANDOFF_FORMAT_CSV = "csv"
HANDOFF_FORMAT_PAIN001 = "pain001"
HANDOFF_FORMATS = (HANDOFF_FORMAT_CSV, HANDOFF_FORMAT_PAIN001)

PAIN001_NAMESPACE = "urn:iso:std:iso:20022:tech:xsd:pain.001.001.09"

# Completion markers for the bank-account elements the agent never
# holds: the importing system fills them from its own records at
# import (see docs/handoff-formats.md). The creditor marker is
# what a member carries when the supplier master holds no approved
# bank details for its supplier — at export the master completes
# the members it covers (see creditor_completions).
DEBTOR_ACCOUNT_MARKER = "DEBTOR-ACCOUNT-AT-IMPORT"
DEBTOR_AGENT_MARKER = "DEBTOR-AGENT-AT-IMPORT"
CREDITOR_ACCOUNT_MARKER = "SUPPLIER-MASTER-AT-IMPORT"

# Where a member's creditor account came from, in the CSV column
# and the export receipt.
CREDITOR_SOURCE_MASTER = "supplier_master"
CREDITOR_SOURCE_AT_IMPORT = "at_import"


def creditor_completions(
    run: PaymentRun, master
) -> tuple[dict[str, dict], list[HandoffCreditorEntry]]:
    """Resolve each run member's creditor account against the
    supplier master: (completions by invoice id, receipt entries).

    A member is completed **only** when the master holds a record
    for its supplier AND that record carries bank identifiers —
    the governed, approved details. The completion prefers the
    IBAN (emitted as the pain.001 ``IBAN`` element / the CSV's
    creditor account), else the account number (with the sort code
    as the creditor agent where recorded). A supplier with no
    record, or a record without bank details, completes nothing:
    the member keeps the at-import marker, and the receipt entry
    says so. The master is read, never written, here.
    """
    completions: dict[str, dict] = {}
    entries: list[HandoffCreditorEntry] = []
    for line in run.lines:
        record = master.get_supplier(line.supplier_name) if master else None
        has_account = record is not None and bool(
            (record.iban and _iban_norm(record.iban))
            or (record.account_number or "").strip()
        )
        if not has_account:
            entries.append(
                HandoffCreditorEntry(
                    invoice_id=line.invoice_id,
                    supplier_name=line.supplier_name,
                    source=CREDITOR_SOURCE_AT_IMPORT,
                )
            )
            continue
        if record.iban and _iban_norm(record.iban):
            kind, value = "iban", _iban_norm(record.iban)
        else:
            kind, value = "account", (record.account_number or "").strip()
        completions[line.invoice_id] = {
            "iban": _iban_norm(record.iban) if record.iban else "",
            "account_number": (record.account_number or "").strip(),
            "sort_code": (record.sort_code or "").strip(),
        }
        entries.append(
            HandoffCreditorEntry(
                invoice_id=line.invoice_id,
                supplier_name=line.supplier_name,
                source=CREDITOR_SOURCE_MASTER,
                account_kind=kind,
                account_label=_masked(kind, value),
            )
        )
    return completions, entries


def handoff_key(run_id: str, format: str = HANDOFF_FORMAT_CSV) -> str:
    """The object-store key a run's hand-off file lives under."""
    if format == HANDOFF_FORMAT_PAIN001:
        return f"payment-runs/{run_id}/handoff.pain001.xml"
    return f"payment-runs/{run_id}/handoff.csv"


def handoff_content_type(format: str) -> str:
    return "application/xml" if format == HANDOFF_FORMAT_PAIN001 else "text/csv"


def _money(value: float) -> str:
    return f"{value:.2f}"


def _creditor_cell(
    completion: dict | None,
) -> tuple[str, str]:
    """The CSV's creditor cells for one member: the completed
    account (IBAN preferred) and its source, or the marker."""
    if completion:
        account = completion.get("iban") or completion.get("account_number") or ""
        if account:
            return account, CREDITOR_SOURCE_MASTER
    return CREDITOR_ACCOUNT_MARKER, CREDITOR_SOURCE_AT_IMPORT


def render_handoff_csv(
    run: PaymentRun, creditor_accounts: dict[str, dict] | None = None
) -> bytes:
    """Render a confirmed run as the hand-off CSV (layout v2).

    Pure and deterministic: the totals row re-states the run's
    code-computed totals, and a reader can verify the file by
    summing the line rows. ``creditor_accounts`` (from
    :func:`creditor_completions`) fills the creditor columns;
    without it every member carries the at-import marker."""
    completions = creditor_accounts or {}
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(HANDOFF_COLUMNS)
    for index, line in enumerate(run.lines, start=1):
        account, source = _creditor_cell(completions.get(line.invoice_id))
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
                account,
                source,
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
            "",
            "",
        ]
    )
    return buffer.getvalue().encode("utf-8")


def _el(parent: ET.Element, tag: str, text: str | None = None) -> ET.Element:
    element = ET.SubElement(parent, f"{{{PAIN001_NAMESPACE}}}{tag}")
    if text is not None:
        element.text = text
    return element


def batch_groups(
    run: PaymentRun, floor_date: str = ""
) -> list[tuple[str, list[PaymentRunLine]]]:
    """Group a run's lines by requested execution date for batch
    booking: (execution date, lines) pairs, dates ascending, lines
    in run order within a group.

    A line's group date is the later of its own due date — the
    run's window end when the line carries none (or an
    unparseable one) — and ``floor_date``, the governing
    profile's lead-days floor (export date + lead days). The
    interaction with the profile's lead days is stated, not
    implied: **the floor wins whenever it is later** — the bank
    gets its processing lead on every group — and the due date
    wins whenever IT is later, so no group executes before its
    lines fall due. With no floor, groups are simply the lines'
    due dates. Pure over the run."""
    floor = _parse_date(floor_date) if floor_date else None
    groups: dict[str, list[PaymentRunLine]] = {}
    for line in run.lines:
        due = _parse_date(line.due_date) if line.due_date else None
        if due is None:
            due = _parse_date(run.window_end)
        chosen = due
        if floor is not None and (chosen is None or floor > chosen):
            chosen = floor
        key = chosen.isoformat() if chosen is not None else (run.window_end or "")
        groups.setdefault(key, []).append(line)
    return sorted(groups.items())


def batch_group_members(run: PaymentRun) -> dict[str, list[str]]:
    """The batch-booked export's groups, read back from its
    receipt: execution date → the member invoice ids exported
    under it, in export order.

    The membership is read from the export-time completion
    entries (``handoff_creditor_entries`` — each stamped with
    its group's date at export), NOT recomputed from the run's
    lines: the bank answered the file that was sent, whose
    grouping the receipt is the record of. A run that was not
    batch-booked (or not exported) has no groups — an empty
    dict, never a reconstruction. Pure."""
    if not run.handoff_batch_booking:
        return {}
    groups: dict[str, list[str]] = {}
    for entry in run.handoff_creditor_entries:
        if not entry.execution_date:
            continue
        groups.setdefault(entry.execution_date, []).append(entry.invoice_id)
    return groups


def current_group_receipts(run: PaymentRun) -> dict[str, HandoffGroupReceipt]:
    """Each group's standing receipt: execution date → the
    latest non-superseded :class:`HandoffGroupReceipt`. A group
    whose rejected answer was later superseded reads by its
    newest answer; a group with no answer is absent. Pure."""
    current: dict[str, HandoffGroupReceipt] = {}
    for receipt in run.handoff_group_receipts:
        if receipt.superseded:
            continue
        current[receipt.execution_date] = receipt
    return current


def handoff_group_reexport_key(
    run_id: str, execution_date: str, sequence: int
) -> str:
    """The object-store key a bounced group's solo re-export
    lives under: beside the run's own files, named by the group
    and the re-export's sequence for that group (a group can
    bounce, be re-presented, and bounce again — each file
    keeps its own key)."""
    return (
        f"payment-runs/{run_id}/handoff-group-{execution_date}"
        f"-reexport-{sequence}.pain001.xml"
    )


def group_reexports(run: PaymentRun, execution_date: str) -> list:
    """A group's solo re-exports, in the order they were made.
    Pure over the run record."""
    return [
        entry
        for entry in run.handoff_group_reexports
        if entry.execution_date == execution_date
    ]


def latest_group_reexport(run: PaymentRun, execution_date: str):
    """A group's most recent solo re-export, or None when the
    group has only ever travelled in the whole file. Pure."""
    entries = group_reexports(run, execution_date)
    return entries[-1] if entries else None


def group_file_sent_at(run: PaymentRun, execution_date: str) -> str:
    """When the file the bank is currently answering for this
    group went out: the latest solo re-export's stamp when the
    group was re-presented, else the whole-file export's stamp
    — whichever is later (a fresh whole-file export supersedes
    a solo one, and restarts every group's clock). Pure."""
    sent = run.handoff_generated_at or ""
    reexport = latest_group_reexport(run, execution_date)
    if reexport is not None and reexport.reexported_at > sent:
        sent = reexport.reexported_at
    return sent


def group_receipt_by_member(run: PaymentRun) -> dict[str, dict]:
    """Every batch member's bank answer, by invoice id: the
    standing receipt of the group it exported in — status,
    the group's execution date and PmtInf id, the bank's
    reason or reference. Members whose group has no standing
    answer are absent: unmarked means unanswered, not accepted.
    Pure."""
    current = current_group_receipts(run)
    marked: dict[str, dict] = {}
    for date, receipt in current.items():
        for invoice_id in receipt.member_invoice_ids:
            marked[invoice_id] = {
                "status": receipt.status,
                "execution_date": date,
                "pmt_inf_id": receipt.pmt_inf_id,
                "detail": receipt.detail,
                "bank_reference": receipt.bank_reference,
            }
    return marked


def render_handoff_pain001(
    run: PaymentRun,
    creditor_accounts: dict[str, dict] | None = None,
    requested_execution_date: str = "",
    batch_booking: bool = False,
    batch_floor_date: str = "",
) -> bytes:
    """Render a confirmed run as an ISO 20022 pain.001.001.09
    customer credit-transfer initiation document.

    The mapping (element by element in ``docs/handoff-formats.md``):
    the group header carries the run id as the message id and the
    run's confirmation time as the creation time — deterministic
    for a given run, like the CSV. One ``PmtInf`` covers the whole
    run (single currency, single requested execution date: the
    run's window end, or the caller's ``requested_execution_date``
    when the governing hand-off profile's lead days push it
    later); each line becomes one ``CdtTrfTxInf`` at its
    export-time *net* amount, with the supplier invoice number in
    the remittance information and the agent's invoice id as the
    end-to-end id.

    Bank-account elements are explicit about their provenance: the
    debtor's account and agent always carry the documented
    ``*-AT-IMPORT`` markers (the customer's own account belongs to
    their bank portal, never to master data). A creditor account is
    completed from the governed supplier master when
    ``creditor_accounts`` (from :func:`creditor_completions`)
    covers the member — the ``IBAN`` element where an IBAN is on
    file, else the account number with the sort code as the
    creditor agent — and carries the ``SUPPLIER-MASTER-AT-IMPORT``
    marker everywhere else.

    With ``batch_booking`` (a governing profile's call — see
    :func:`batch_groups`), the one ``PmtInf`` becomes one per
    execution-date group, each with ``BtchBookg`` true, its own
    ``NbOfTxs``/``CtrlSum`` over its own lines, and the group's
    date as its ``ReqExctnDt``; ``batch_floor_date`` is the
    profile's lead-days floor every group date respects. The
    group header keeps the run's totals — the file is still one
    message about one run. The default single-``PmtInf`` output
    is byte-identical to before batch booking existed (a golden
    file pins it).
    """
    completions = creditor_accounts or {}
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

    if batch_booking:
        for group_date, group_lines in batch_groups(run, batch_floor_date):
            _payment_info(
                initiation,
                run,
                group_lines,
                pmt_inf_id=f"{run.run_id}-{group_date}",
                execution_date=group_date,
                ctrl_sum=round(sum(line.net_amount for line in group_lines), 2),
                completions=completions,
                batch=True,
            )
    else:
        _payment_info(
            initiation,
            run,
            list(run.lines),
            pmt_inf_id=run.run_id,
            execution_date=requested_execution_date or run.window_end,
            ctrl_sum=run.total_net,
            completions=completions,
            batch=False,
        )

    tree = ET.ElementTree(document)
    ET.indent(tree, space="  ")
    buffer = io.BytesIO()
    tree.write(buffer, encoding="utf-8", xml_declaration=True)
    return buffer.getvalue()


def _payment_info(
    initiation: ET.Element,
    run: PaymentRun,
    lines: list[PaymentRunLine],
    *,
    pmt_inf_id: str,
    execution_date: str,
    ctrl_sum: float,
    completions: dict[str, dict],
    batch: bool,
) -> None:
    """One ``PmtInf`` block: the whole run (the default) or one
    batch-booking group. The debtor block repeats per ``PmtInf``
    because pain.001 scopes it there; the per-line transaction
    rendering — creditor completion or marker included — is the
    same code either way, so the completion gate holds inside
    every group exactly as it holds for the whole file."""
    payment = _el(initiation, "PmtInf")
    _el(payment, "PmtInfId", pmt_inf_id)
    _el(payment, "PmtMtd", "TRF")
    if batch:
        _el(payment, "BtchBookg", "true")
    _el(payment, "NbOfTxs", str(len(lines)))
    _el(payment, "CtrlSum", _money(ctrl_sum))
    _el(payment, "ReqExctnDt", execution_date)
    debtor = _el(payment, "Dbtr")
    _el(debtor, "Nm", "Debtor account holder (completed at import)")
    account = _el(payment, "DbtrAcct")
    other = _el(_el(account, "Id"), "Othr")
    _el(other, "Id", DEBTOR_ACCOUNT_MARKER)
    agent = _el(payment, "DbtrAgt")
    other = _el(_el(_el(agent, "FinInstnId"), "Othr"), "Id")
    other.text = DEBTOR_AGENT_MARKER
    _el(payment, "ChrgBr", "SLEV")

    for line in lines:
        tx = _el(payment, "CdtTrfTxInf")
        pmt_id = _el(tx, "PmtId")
        _el(pmt_id, "EndToEndId", line.invoice_id)
        amount = _el(tx, "Amt")
        instructed = _el(amount, "InstdAmt", _money(line.net_amount))
        instructed.set("Ccy", run.currency)
        completion = completions.get(line.invoice_id) or {}
        if completion and not completion.get("iban") and completion.get("sort_code"):
            # Account-number completion: the sort code identifies
            # the creditor's branch/agent (pain.001 order: CdtrAgt
            # precedes Cdtr/CdtrAcct).
            agent = _el(tx, "CdtrAgt")
            other = _el(_el(_el(agent, "FinInstnId"), "Othr"), "Id")
            other.text = completion["sort_code"]
        creditor = _el(tx, "Cdtr")
        _el(creditor, "Nm", line.supplier_name)
        account = _el(tx, "CdtrAcct")
        if completion.get("iban"):
            _el(_el(account, "Id"), "IBAN", completion["iban"])
        elif completion.get("account_number"):
            other = _el(_el(account, "Id"), "Othr")
            _el(other, "Id", completion["account_number"])
        else:
            other = _el(_el(account, "Id"), "Othr")
            _el(other, "Id", CREDITOR_ACCOUNT_MARKER)
        remittance = _el(tx, "RmtInf")
        _el(
            remittance,
            "Ustrd",
            f"Invoice {line.invoice_number} ({line.invoice_id})"[:140],
        )


def render_handoff(
    run: PaymentRun,
    format: str = HANDOFF_FORMAT_CSV,
    creditor_accounts: dict[str, dict] | None = None,
    requested_execution_date: str = "",
    batch_booking: bool = False,
    batch_floor_date: str = "",
) -> bytes:
    """Render a confirmed run in one of the hand-off formats —
    the one seam both serializations sit behind.
    ``requested_execution_date`` reaches the pain.001 renderer
    only (the CSV layout has no execution-date column); the
    service computes it from the governing hand-off profile's
    lead days (``handoff_profiles.requested_execution_date``).
    ``batch_booking`` + ``batch_floor_date`` reach it the same
    way when the governing profile books per value date; the
    CSV ignores both — it stays the flat list."""
    if format == HANDOFF_FORMAT_PAIN001:
        return render_handoff_pain001(
            run,
            creditor_accounts,
            requested_execution_date,
            batch_booking=batch_booking,
            batch_floor_date=batch_floor_date,
        )
    if format == HANDOFF_FORMAT_CSV:
        return render_handoff_csv(run, creditor_accounts)
    raise ValueError(
        f"Unknown hand-off format {format!r} — expected one of {HANDOFF_FORMATS}"
    )
