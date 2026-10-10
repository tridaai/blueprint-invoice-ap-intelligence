-- 0001_core: pgvector extension + the approvals record table.
--
-- The approvals table is the system of record for analyses and human
-- decisions (see store.py, PostgresStore). One row per invoice id;
-- the full record (result + invoice payload + decision fields) lives
-- in the JSONB payload, exactly as the store serialises it. `seq`
-- preserves insertion order for history reads (upserts keep the
-- original seq), `decided` is the indexed decision flag.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS approvals (
    seq        BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    invoice_id TEXT PRIMARY KEY,
    payload    JSONB NOT NULL,
    decided    BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS approvals_decided_idx ON approvals (decided);
