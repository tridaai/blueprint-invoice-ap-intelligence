# Examples

Worked examples of the blueprint's seams — reference code that
shows how a customer wires the agent into their own systems. Like
the agent, the examples move no money and call no real service.

## `erp_posting.py` — approved invoice → ERP posting payload

The output-routing question every evaluator asks first: *"the
approver approved it in your console — how does it get into our
ERP?"* The blueprint's answer is the routing seam: the approval
webhook (payload + optional HMAC signature) for push, the payment
run's hand-off CSV for the pay cycle, and — this example — an
adapter that shapes an approved result into a generic journal
posting for pull/import flows.

```bash
# An approved result's JSON (from the API) -> a posting payload:
curl localhost:8000/invoices/INV-1001 | python examples/erp_posting.py -
```

The journal it builds (account codes are placeholders — map them
to your chart of accounts):

| line | account | debit | credit |
|---|---|---|---|
| goods/services, net | `6100-PURCHASES` | lines total | |
| input tax | `1310-INPUT-TAX-RECOVERABLE` | tax stated | |
| payable | `2000-ACCOUNTS-PAYABLE-CONTROL` | | invoice total |

What the example is careful about, and any real adapter should be
too:

- **Amounts come from the computed facts** (`facts.invoice_lines_total`,
  `facts.tax_stated`, `facts.invoice_total_stated`) — never from the
  letter or packet prose.
- **Only `status == "approved"` results post**; anything else raises.
  The human decision is the posting's authority.
- **Unreconciled figures are refused**: lines + tax must equal the
  stated total within a cent, or the adapter reports the variance
  instead of posting an exception.
- **The idempotency key is stable per approval** (derived from the
  invoice id + supplier invoice number + total), so a retried import
  de-duplicates at the ERP instead of double-posting.
- No vendor SDK, no network, no credentials: turning the payload
  into an actual posting in SAP / NetSuite / Dynamics / anything
  else is a thin transport step on top of this shape, and it belongs
  to the customer's system of record — deliberately outside the
  agent.

### The receipt loop — the ERP answers back

A posting without an answer leaves the agent's record and the
ledger free to drift apart. The same example shapes the answer:

```bash
python examples/erp_posting.py --receipt posting.json posted ERP-DOC-1042
python examples/erp_posting.py --receipt posting.json failed "" "period 2026-09 is closed"
```

which produces the body of `POST /invoices/{invoice_id}/posting-receipt`:

```json
{"invoice_id": "INV-1001", "status": "posted",
 "external_reference": "ERP-DOC-1042",
 "idempotency_key": "ap-posting-…", "actor": "<operator>"}
```

The service records the receipt on the invoice and in the audit
export. The discipline mirrors the posting side: a `posted` receipt
must name the ERP's document number, a `failed` one must carry its
reason, only an approved invoice can carry a receipt at all, and
once an invoice is recorded as posted, a second receipt is refused
unless it replays the same idempotency key — a double posting is
corrected in the ERP, not papered over here. A `failed` receipt may
be superseded by the retry's outcome. The agent still posts
nothing itself; the receipt is bookkeeping that closes the loop.
