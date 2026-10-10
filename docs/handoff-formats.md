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
- a run stamped four-eyes (`PAYMENT_RUN_FOUR_EYES_THRESHOLD`)
  must carry its **second confirmation**, from a different person;
- every member's discount is **re-verified at export** as of the
  export date — a member whose window closed since confirmation
  is exported at full gross and the drift is itemised on the run
  record (`handoff_drift`), never silently netted off.

Choose the format per export: `POST /payment-runs/{run_id}/handoff`
defaults to `?format=csv`; `?format=pain001` selects the XML. The
run record's export receipt (`handoff_key`, `handoff_sha256`,
`handoff_generated_at/by`, `handoff_format`) always names the
format of the file it points at.

## Format 1 — CSV (layout v1)

Object key: `payment-runs/<run_id>/handoff.csv`

One header row, one row per run line, one totals row. Amounts are
plain decimals with two places, no thousands separators; dates ISO.

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
| `PmtInf/ReqExctnDt` | the run's window end — pay on or before this date |
| `PmtInf/ChrgBr` | `SLEV` (charges follow the service-level agreement) |
| `CdtTrfTxInf/PmtId/EndToEndId` | the agent's invoice id |
| `CdtTrfTxInf/Amt/InstdAmt` (+ `Ccy`) | the line's export-time **net** amount, in the run's currency |
| `CdtTrfTxInf/Cdtr/Nm` | the supplier name as recorded on the invoice |
| `CdtTrfTxInf/RmtInf/Ustrd` | `Invoice <invoice_number> (<invoice_id>)` |

### Completed at import — the bank-account elements

The agent holds no bank accounts, deliberately: creditor bank
details are governed supplier-master data (documents stating bank
details are exactly what the fraud screen checks *against* the
master — a hand-off file must never source account data from
documents), and the debtor's account belongs to the customer's own
bank portal. These elements are therefore emitted with explicit
completion markers, and the importing system fills them from its
own records — keyed on the creditor name and remittance
information above — before submission:

| element | marker | completed from |
|---|---|---|
| `PmtInf/DbtrAcct/Id/Othr/Id` | `DEBTOR-ACCOUNT-AT-IMPORT` | the customer's bank portal / ERP company bank setup |
| `PmtInf/DbtrAgt/FinInstnId/Othr/Id` | `DEBTOR-AGENT-AT-IMPORT` | the customer's bank |
| `CdtTrfTxInf/CdtrAcct/Id/Othr/Id` | `SUPPLIER-MASTER-AT-IMPORT` | the ERP's supplier master (the same record the bank-change screen governs) |

A bank or ERP that rejects placeholder account ids at import is
doing its job: the markers make the hand-off's one incompleteness
visible at exactly the step where the authoritative data lives,
instead of letting the agent invent or transcribe account numbers.
