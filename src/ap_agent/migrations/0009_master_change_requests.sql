-- 0009_master_change_requests: governed supplier-master changes.
--
-- A supplier bank-detail change is the classic AP fraud vector, so
-- the master never changes by a document's say-so: a change is
-- proposed as a request object (master_changes.py,
-- service.propose_master_change), a second operator approves it
-- with an out-of-band verification reference recorded, and only then
-- does it land in supplier_master — on or after its effective date.
-- One row per request id; the full request (proposal, decision,
-- audit history) lives in the JSONB payload. Until a change is
-- approved AND applied, the fraud screen keeps comparing documents
-- against the previous master record.

CREATE TABLE IF NOT EXISTS master_change_requests (
    seq        BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    request_id TEXT PRIMARY KEY,
    payload    JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
