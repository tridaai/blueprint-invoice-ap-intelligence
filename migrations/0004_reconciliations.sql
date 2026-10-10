-- 0004_reconciliations: supplier statement reconciliation reports.
--
-- A reconciliation run compares a supplier's statement of account
-- against the processed invoices and produces a report a human
-- confirms (reconciliation.py, service.reconcile_statement). The
-- report is the only thing stored here — confirming one never
-- touches the approvals table. One row per statement id; the full
-- report lives in the JSONB payload, as the store serialises it.

CREATE TABLE IF NOT EXISTS reconciliations (
    seq          BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    statement_id TEXT PRIMARY KEY,
    payload      JSONB NOT NULL,
    confirmed    BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
