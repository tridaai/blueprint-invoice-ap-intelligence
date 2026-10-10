-- 0008_supplier_master: supplier master data — the out-of-band record
-- of each supplier's approved bank details, payment terms, and tax
-- id (supplier_master.py, the SupplierMaster port).
--
-- The bank-change fraud screen checks the bank identifiers a
-- document states against this record: a mismatch is the fraud
-- signal, a match is evidence (AP-BANK-01). One row per supplier,
-- keyed by the normalised supplier name; the full record lives in
-- the JSONB payload. The table starts empty in production — the
-- customer loads their verified master data (the offline path seeds
-- the in-memory double with the synthetic samples instead).

CREATE TABLE IF NOT EXISTS supplier_master (
    seq          BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    supplier_key TEXT PRIMARY KEY,
    payload      JSONB NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
