# Payment-run hand-off formats

A confirmed payment run leaves this system as a file — the routing
adapter between the agent's pay list and the customer's banking or
ERP system, which is what actually executes the payments. The file
is rendered deterministically from the confirmed run and stored
through the object-store port; generating it changes no payment
state, in either format.

Both formats sit behind the one hand-off seam
(`ap_agent/handoff.py`), behind the same gates:

- the run must be **confirmed** by a finance lead;
- a run stamped four-eyes (its currency's threshold —
  `PAYMENT_RUN_FOUR_EYES_THRESHOLDS`, falling back to
  `PAYMENT_RUN_FOUR_EYES_THRESHOLD`) must carry its **second
  confirmation**, from a different person;
- every member's discount is **re-verified at export** as of the
  export date — a member whose window closed since confirmation
  is exported at full gross and the drift is itemised on the run
  record (`handoff_drift`), never silently netted off.

Choose the format per export — or let the run's governed
**hand-off profile** choose it. A profile (`GET/POST
/handoff/profiles`, migration 0011) records the payee side's
routing facts per supplier or per currency: preferred format,
a lead-days figure for the pain.001 requested execution date,
and whether the payee's bank rejects at-import creditor
markers. Resolution is deterministic: `?format=csv|pain001`
named on the export call wins; with no format named, the
governing profile's format applies — the run's currency
profile first, then a supplier profile the run's profiled
members agree on — and with neither, the CSV default. The
governing profile's lead days push `ReqExctnDt` later (never
earlier than the window end), whoever picked the format. The
run record's export receipt (`handoff_key`, `handoff_sha256`,
`handoff_generated_at/by`, `handoff_format`) always names the
format of the file it points at, joined now by
`handoff_profile_id` (the profile that governed, if any) and
`handoff_execution_date` (the `ReqExctnDt` written);
`handoff_creditor_entries` records how each member's creditor
account was completed (see "Bank-account elements" below). A
profile whose bank rejects markers also flags the run at
*proposal* time (`handoff_profile_flags`) for any member the
master holds no bank details for — the import refusal is
prevented at the desk, not discovered at the bank. Confirmation also computes
the run's drift forecast (`drift_forecast`): members whose
discount deadlines fall before the expected export date are
flagged while the confirmer can still act — a projection from
the measured proposal→export lag, labelled as a forecast; the
export-time drift re-verification above remains the authority.

## Format 1 — CSV (layout v2)

Object key: `payment-runs/<run_id>/handoff.csv`

One header row, one row per run line, one totals row. Amounts are
plain decimals with two places, no thousands separators; dates ISO.
Layout v2 appends the two creditor columns to v1 — the v1 columns
keep their order and meaning.

| column | content |
|---|---|
| `run_id` | the confirmed run's id (the totals row repeats it) |
| `line_no` | 1-based line number; blank on the totals row |
| `invoice_id` | the agent's invoice id; `TOTAL` on the totals row |
| `supplier_name` | as recorded on the invoice |
| `invoice_number` | the supplier's invoice number |
| `due_date` | ISO date, blank when unknown |
| `currency` | the run's currency |
| `gross_amount` | the approved invoice total |
| `discount_amount` | capturable early-payment discount, else 0.00 |
| `net_amount` | gross − discount: the amount to pay |
| `creditor_account` | the supplier master's IBAN (else account number) for the supplier, else `SUPPLIER-MASTER-AT-IMPORT` |
| `creditor_account_source` | `supplier_master` when completed from the master, else `at_import` |

## Format 2 — ISO 20022 pain.001 (XML)

Object key: `payment-runs/<run_id>/handoff.pain001.xml`

A `pain.001.001.09` customer credit-transfer initiation document:
one payment information block for the run, one credit-transfer
transaction per run line. The mapping:

| pain.001 element | source on the run |
|---|---|
| `GrpHdr/MsgId` | the run id |
| `GrpHdr/CreDtTm` | the run's confirmation timestamp |
| `GrpHdr/NbOfTxs`, `PmtInf/NbOfTxs` | the run's invoice count |
| `GrpHdr/CtrlSum`, `PmtInf/CtrlSum` | the run's total net (export-time figures) |
| `GrpHdr/InitgPty/Nm` | `Trida AI Blueprint - AP Agent` |
| `PmtInf/PmtInfId` | the run id |
| `PmtInf/PmtMtd` | `TRF` (credit transfer) |
| `PmtInf/ReqExctnDt` | the run's window end — pay on or before this date; pushed later to export date + the governing hand-off profile's lead days when that is later (the run stamps the date written as `handoff_execution_date`) |
| `PmtInf/ChrgBr` | `SLEV` (charges follow the service-level agreement) |
| `CdtTrfTxInf/PmtId/EndToEndId` | the agent's invoice id |
| `CdtTrfTxInf/Amt/InstdAmt` (+ `Ccy`) | the line's export-time **net** amount, in the run's currency |
| `CdtTrfTxInf/Cdtr/Nm` | the supplier name as recorded on the invoice |
| `CdtTrfTxInf/CdtrAcct/Id/IBAN` | the master's IBAN for the supplier — only when the master holds one (see below) |
| `CdtTrfTxInf/CdtrAcct/Id/Othr/Id` | the master's account number for the supplier, else the `SUPPLIER-MASTER-AT-IMPORT` marker |
| `CdtTrfTxInf/CdtrAgt/FinInstnId/Othr/Id` | the master's sort code — only on account-number completions |
| `CdtTrfTxInf/RmtInf/Ustrd` | `Invoice <invoice_number> (<invoice_id>)` |

### Bank-account elements — completed from the master, or marked

Creditor bank details are governed supplier-master data: documents
stating bank details are exactly what the fraud screen checks
*against* the master, so a hand-off file never sources account
data from documents. It may, however, source them from the master
itself — at export, each member's supplier is looked up, and
where (and only where) the master holds approved bank details,
the creditor account is completed from them: the IBAN where
recorded, else the account number with the sort code as the
creditor agent. Every other member carries the explicit
`SUPPLIER-MASTER-AT-IMPORT` marker for the importing system to
complete. The run record's export receipt
(`handoff_creditor_entries`) records, per member, which way it
went — with masked labels only; the full identifiers live in the
master record and in the file, never on the run record.

The debtor side is never completed: the customer's own account
belongs to their bank portal, not to master data.

| element | when not completed | marker |
|---|---|---|
| `PmtInf/DbtrAcct/Id/Othr/Id` | always | `DEBTOR-ACCOUNT-AT-IMPORT` |
| `PmtInf/DbtrAgt/FinInstnId/Othr/Id` | always | `DEBTOR-AGENT-AT-IMPORT` |
| `CdtTrfTxInf/CdtrAcct/...` | the master holds no bank details for the supplier | `SUPPLIER-MASTER-AT-IMPORT` |

A bank or ERP that rejects placeholder account ids at import is
doing its job: the markers make the hand-off's remaining
incompleteness visible at exactly the step where the
authoritative data lives, instead of letting the agent invent or
transcribe account numbers.
