-- 0015_credit_applications: approver-confirmed credit-note applications.
--
-- Credit-note matching (creditnotes.py) is a pure report; the write
-- path is an approver recording a matched credit against its invoice
-- (service.apply_credit_note). One row per credit note id; the full
-- application lives in the JSONB payload, as the store serialises it.
-- Applications net against the invoice's payable position and into
-- payment-run proposals — they never touch the approvals table.

CREATE TABLE IF NOT EXISTS credit_applications (
    seq            BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    credit_note_id TEXT PRIMARY KEY,
    payload        JSONB NOT NULL,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
