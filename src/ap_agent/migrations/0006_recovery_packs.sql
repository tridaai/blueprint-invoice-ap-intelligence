-- 0006_recovery_packs: duplicate-payment recovery packs.
--
-- When duplicate detection fires on an invoice whose original was
-- already approved, the recovery case — computed evidence, a supplier
-- recovery letter, an internal memo, and the verification +
-- guardrail results over those drafts — is stored here
-- (recovery.py, service.draft_duplicate_recovery). One row per
-- duplicate invoice id; the full pack lives in the JSONB payload.
-- Storing a pack changes no invoice's state and recovers nothing:
-- a human sends the letter and works the case.

CREATE TABLE IF NOT EXISTS recovery_packs (
    seq        BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    invoice_id TEXT PRIMARY KEY,
    payload    JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
