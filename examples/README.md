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
