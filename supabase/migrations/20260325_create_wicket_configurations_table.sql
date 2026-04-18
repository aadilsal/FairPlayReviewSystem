-- Migration: Create wicket_configurations table
-- Description: Persist per-match wicket coordinates so users configure once.
-- Created: 2026-03-25

CREATE TABLE IF NOT EXISTS public.wicket_configurations (
    id BIGSERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    configured BOOLEAN NOT NULL DEFAULT FALSE,
    -- Stored as [x, y, w, h] (pixel coordinates in the analyzed video frames)
    near_box JSONB,
    far_box JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT fk_wicket_configurations_match
        FOREIGN KEY (match_id)
        REFERENCES public.matches(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_wicket_configurations_user
        FOREIGN KEY (user_id)
        REFERENCES public.users(id)
        ON DELETE CASCADE,

    CONSTRAINT uq_wicket_configurations_match_user
        UNIQUE (match_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_wicket_configurations_match_id ON public.wicket_configurations(match_id);
CREATE INDEX IF NOT EXISTS idx_wicket_configurations_user_id ON public.wicket_configurations(user_id);
CREATE INDEX IF NOT EXISTS idx_wicket_configurations_configured ON public.wicket_configurations(configured);

DROP TRIGGER IF EXISTS update_wicket_configurations_updated_at ON public.wicket_configurations;
CREATE TRIGGER update_wicket_configurations_updated_at
    BEFORE UPDATE ON public.wicket_configurations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

ALTER TABLE public.wicket_configurations ENABLE ROW LEVEL SECURITY;

-- Keep policies permissive (API enforces ownership via match.user_id checks)
DROP POLICY IF EXISTS "Enable read access for all users" ON public.wicket_configurations;
CREATE POLICY "Enable read access for all users" ON public.wicket_configurations
    FOR SELECT
    USING (true);

DROP POLICY IF EXISTS "Enable insert for all users" ON public.wicket_configurations;
CREATE POLICY "Enable insert for all users" ON public.wicket_configurations
    FOR INSERT
    WITH CHECK (true);

DROP POLICY IF EXISTS "Enable update for all users" ON public.wicket_configurations;
CREATE POLICY "Enable update for all users" ON public.wicket_configurations
    FOR UPDATE
    USING (true);

GRANT SELECT, INSERT, UPDATE ON public.wicket_configurations TO anon, authenticated;
GRANT USAGE, SELECT ON SEQUENCE wicket_configurations_id_seq TO anon, authenticated;

COMMENT ON TABLE public.wicket_configurations IS 'Stores persisted wicket boxes per match so detection/prediction can use fixed coordinates';
COMMENT ON COLUMN public.wicket_configurations.configured IS 'True once near/far boxes are set for the match';
COMMENT ON COLUMN public.wicket_configurations.near_box IS 'Near wicket bbox as JSON array [x,y,w,h]';
COMMENT ON COLUMN public.wicket_configurations.far_box IS 'Far wicket bbox as JSON array [x,y,w,h]';
