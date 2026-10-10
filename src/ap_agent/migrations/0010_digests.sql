-- 0010_digests: stored escalation digests.
--
-- The digest (digest.py, service.escalation_digest) is composed
-- deterministically from the stored records at read time — but a
-- day's escalations cannot be re-derived once the records move:
-- cases transition, postings close, runs export. The digest a
-- human forwards, or files for audit, must be the digest as it
-- was, not a recomputation. So an operator can store the day's
-- first-computed digest (service.store_digest): one row per date,
-- the full digest in the JSONB payload. First-computed wins — a
-- stored copy is never overwritten, and its provenance fields say
-- who stored it and when. Recomputation stays available at
-- GET /digest, labelled "computed" — it never wears the label.

CREATE TABLE IF NOT EXISTS digests (
    seq         BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    digest_date TEXT PRIMARY KEY,
    payload     JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
