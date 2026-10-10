-- 0011_handoff_profiles: governed hand-off routing profiles.
--
-- A payment run leaves as one file in one format, but the right
-- format is a property of the payee side: one ERP ingests the
-- CSV, another bank wants pain.001, a third needs lead days
-- before the requested execution date, and some banks reject
-- files whose creditor account is still the at-import marker.
-- The profile (handoff_profiles.py) is where an operator
-- records that routing knowledge — scoped to one supplier or
-- to a whole currency — with the actor named and an append-only
-- audit trail inside the payload. One row per profile id; the
-- export resolves profiles deterministically, so the file a
-- bank receives is reproducible from the profiles in force.

CREATE TABLE IF NOT EXISTS handoff_profiles (
    seq         BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    profile_id  TEXT PRIMARY KEY,
    payload     JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
