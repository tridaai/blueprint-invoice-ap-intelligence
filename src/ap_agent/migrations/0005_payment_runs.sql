-- 0005_payment_runs: proposed payment runs over the approved invoices.
--
-- A payment run groups approved invoices by currency + due window
-- into a pay list a finance lead confirms (paymentruns.py,
-- service.propose_payment_runs). The run is a draft object: storing
-- or confirming one never touches the approvals table, and nothing
-- here executes a payment — the customer's AP system of record does
-- that. One row per run id; the full run lives in the JSONB payload.

CREATE TABLE IF NOT EXISTS payment_runs (
    seq        BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    run_id     TEXT PRIMARY KEY,
    payload    JSONB NOT NULL,
    confirmed  BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
