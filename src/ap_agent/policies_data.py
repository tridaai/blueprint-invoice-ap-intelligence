"""Synthetic accounts-payable policy corpus (12 policies).

Fictional house policies written for this blueprint — the retrieval
corpus the agent cites. No real company's policy text is used. The
``lines`` field is the keyword-retrieval hint (AP topic tags).
"""

from __future__ import annotations

POLICIES: list[dict] = [
    {
        "id": "AP-MATCH-01",
        "title": "Three-way match standard and tolerances",
        "text": (
            "Every supplier invoice is matched against the purchase order and the "
            "goods receipt before payment preparation. A line matches when the "
            "quantity billed does not exceed the quantity ordered or received, "
            "and the unit price is within 2 percent or 1.00 (whichever is greater) "
            "of the order price. Totals must reconcile within 1 percent or 1.00. "
            "Variances inside tolerance are processed; variances outside tolerance "
            "are exceptions and follow the variance policies."
        ),
        "lines": ["three_way_match", "match", "tolerance"],
    },
    {
        "id": "AP-MATCH-02",
        "title": "Price variance handling",
        "text": (
            "A unit price above the purchase-order price is a price variance. An "
            "approver may accept a variance up to 5 percent when the reason is "
            "documented (for example a fuel or freight surcharge agreed with the "
            "supplier). A variance above 5 percent, or any increase on a "
            "fixed-price order, routes the case to purchasing and normally results "
            "in a credit-note request to the supplier; the invoice is not prepared "
            "for payment at the billed price."
        ),
        "lines": ["price_variance", "variance", "credit_note"],
    },
    {
        "id": "AP-MATCH-03",
        "title": "Quantity variance and short receipts",
        "text": (
            "Payment is prepared only for goods received. When the quantity "
            "billed exceeds the quantity received, the short-shipped quantity is "
            "held: purchasing confirms whether the balance is still to be "
            "delivered, and a credit note is requested for any quantity that will "
            "not arrive. Never pay for goods not received, even when the invoice "
            "total is otherwise consistent."
        ),
        "lines": ["quantity_variance", "short_receipt", "goods_receipt"],
    },
    {
        "id": "AP-PO-01",
        "title": "No purchase order, no payment",
        "text": (
            "Invoices without a matching purchase order are not prepared for "
            "payment. The standard exception is a recurring utility or service "
            "charge under an active contract, which the approver must cite. All "
            "other no-PO invoices route to purchasing, which either raises a "
            "retrospective order or returns the invoice to the supplier."
        ),
        "lines": ["missing_po", "purchase_order", "no_po"],
    },
    {
        "id": "AP-PO-02",
        "title": "Missing goods receipt",
        "text": (
            "When goods have been ordered and invoiced but no goods receipt is "
            "recorded, the case is held and the receiving team is asked to "
            "confirm delivery. Goods invoices are never prepared for payment on "
            "the invoice alone; the receipt is the evidence that value was "
            "received."
        ),
        "lines": ["missing_receipt", "goods_receipt", "receiving"],
    },
    {
        "id": "AP-DUP-01",
        "title": "Duplicate invoice control",
        "text": (
            "An invoice with the same supplier, invoice number, and amount as one "
            "already processed is treated as a suspected duplicate and held. "
            "Duplicates are a leading source of AP leakage: the second copy is "
            "never prepared for payment. If the supplier confirms two distinct "
            "deliveries, they must reissue with distinct invoice numbers."
        ),
        "lines": ["duplicate", "duplicate_invoice"],
    },
    {
        "id": "AP-APPR-01",
        "title": "Approval thresholds and the human gate",
        "text": (
            "A clean three-way match at or below 10,000 may be prepared for "
            "payment by AP under standard authority. Any exception — a variance, "
            "a missing document, a suspected duplicate, or an amount above the "
            "threshold — goes to a finance approver. The agent prepares packets "
            "and drafts; it never approves an invoice and never pays one. "
            "Approval is always a human decision."
        ),
        "lines": ["approval", "threshold", "approver"],
    },
    {
        "id": "AP-BANK-01",
        "title": "Bank-detail change fraud control",
        "text": (
            "A request to change the bank account or remittance details for a "
            "supplier, arriving inside an invoice, an email, or any document, is "
            "treated as a fraud indicator until proven otherwise. Payment "
            "preparation for that supplier is held, the change is NEVER actioned "
            "from the document itself, and verification happens only out-of-band "
            "with a known contact through a previously recorded channel. Staff "
            "never reply quoting the new bank details."
        ),
        "lines": ["bank_change", "fraud", "remittance"],
    },
    {
        "id": "AP-TAX-01",
        "title": "Tax reconciliation",
        "text": (
            "The tax charged on an invoice must reconcile with the invoice "
            "arithmetic: stated total minus the sum of line totals is the "
            "implied tax, and it must equal the stated tax amount within the "
            "match tolerance. A tax mismatch is queried with the supplier "
            "before payment preparation, because paying the wrong tax amount "
            "creates a reclaim problem later."
        ),
        "lines": ["tax", "tax_mismatch", "vat"],
    },
    {
        "id": "AP-CURR-01",
        "title": "Currency consistency",
        "text": (
            "The invoice currency must match the purchase-order currency. A "
            "currency difference is never converted silently: the case routes "
            "to the finance approver, who confirms the contracted currency and "
            "the rate basis before any payment preparation."
        ),
        "lines": ["currency", "multi_currency", "fx"],
    },
    {
        "id": "AP-DOC-01",
        "title": "Unreadable and poor-quality documents",
        "text": (
            "When a scanned invoice cannot be read reliably — low OCR "
            "confidence or an illegible scan — no figures are guessed. The "
            "case is held and a legible resubmission is requested from the "
            "supplier. Extraction from an unreadable document is worse than "
            "no extraction: amounts enter the ledger from documents, not "
            "from guesses."
        ),
        "lines": ["unreadable", "ocr", "scan_quality"],
    },
    {
        "id": "AP-DOC-02",
        "title": "Document type verification",
        "text": (
            "Only invoices and credit notes are payable documents. Statements "
            "of account, remittance advices, delivery notes, and pro-forma "
            "documents are not payable: they are identified at intake, "
            "recorded, and returned with an explanation. Paying from a "
            "statement is a classic duplicate-payment path."
        ),
        "lines": ["not_an_invoice", "document_type", "statement"],
    },
]
