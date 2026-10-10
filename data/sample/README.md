# Sample data

Everything here is **synthetic** — invented suppliers, invoices, purchase orders, and receipts written for this blueprint. No real supplier, invoice, or bank detail is used anywhere in this repository (the "bank account" in INV-1010 is an invented fixture string, present to exercise the fraud screen).

- `sample_invoices.json` — the 12 bundled sample invoices (mirror of the packaged copy in `src/ap_agent/data/`, which is what the CLI / demo / API actually load). Includes the adversarial cases: INV-1010 (a bank-detail change request in the invoice notes — the fraud pattern), INV-1011 (a prompt-injection attempt in the supplier email), and INV-1012 (a supplier note relaying a payment-date demand — the payment-language guardrail trap).
- `policies.json` — the 12 synthetic AP policies behind the retriever (mirror of `src/ap_agent/policies_data.py`).
