-- 0013_lag_segment_proposals: governed decisions about the
-- drift-forecast's lag statistics, drafted from the drift watch.
--
-- The watch demotes a decaying statistic inside the selection,
-- run by run — automatic, and nobody's decision. The proposal
-- (lag_segment_proposals.py) is the governed path to a standing
-- ruling, mirroring the profile-change discipline: drafted from
-- the watch's evidence by one operator, decided by a second,
-- applied only on approval. An approved retirement excludes the
-- segment from the forecast's candidate set; an approved
-- re-grade re-admits it — the exclusion state is derived from
-- this ledger, so there is no second store. One row per
-- proposal id; the decision trail rides inside the payload.

CREATE TABLE IF NOT EXISTS lag_segment_proposals (
    seq         BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    proposal_id TEXT PRIMARY KEY,
    payload     JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
