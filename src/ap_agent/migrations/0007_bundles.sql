-- 0007_bundles: multi-invoice bundle summaries.
--
-- A bundle arrival is segmented in code into per-invoice runs
-- (bundles.py); every invoice is its own row in approvals with its
-- own gate. What is stored here is the arrival's summary — segment
-- count, per-invoice outcomes, totals — so the batch can be
-- retrieved as one arrival later. One row per bundle id; the full
-- summary lives in the JSONB payload.

CREATE TABLE IF NOT EXISTS bundles (
    seq        BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    bundle_id  TEXT PRIMARY KEY,
    payload    JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
