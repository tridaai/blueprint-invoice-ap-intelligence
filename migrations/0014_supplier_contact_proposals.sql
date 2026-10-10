-- 0014_supplier_contact_proposals: governed changes to a
-- supplier's onboarding document checklist, drafted from the
-- re-analysis analytics.
--
-- The analytics name the suppliers re-analysis cannot unblock
-- — reply after reply lands with the outcome unchanged,
-- because the settling document was never collected. The
-- proposal (supplier_contact_proposals.py) is the governed
-- path from that pattern to a checklist change, mirroring the
-- lag-segment discipline: drafted from the analytics' evidence
-- by one operator, decided by a second, applied only on
-- approval. An approved collect puts the supplier on the
-- documents-up-front list; an approved release takes them off
-- — who stands listed is derived from this ledger, so there
-- is no second store. One row per proposal id; the decision
-- trail rides inside the payload.

CREATE TABLE IF NOT EXISTS supplier_contact_proposals (
    seq         BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    proposal_id TEXT PRIMARY KEY,
    payload     JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
