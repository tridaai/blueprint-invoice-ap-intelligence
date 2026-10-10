-- 0012_profile_change_proposals: governed changes to hand-off
-- routing profiles, drafted from the effectiveness grades.
--
-- The grades name configuration theatre — a profile that governs
-- nothing, or one whose files are no safer than ungoverned ones —
-- but a read changes nothing. The proposal (profile_proposals.py)
-- is the governed path from the grade to the change, mirroring
-- the master-change discipline: drafted from the evidence by one
-- operator, decided by a second, applied only on approval (a
-- retirement deletes the profile; a marker tightening sets the
-- flag through the profile's own governed path). One row per
-- proposal id; the decision trail rides inside the payload.

CREATE TABLE IF NOT EXISTS profile_change_proposals (
    seq         BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    proposal_id TEXT PRIMARY KEY,
    payload     JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
