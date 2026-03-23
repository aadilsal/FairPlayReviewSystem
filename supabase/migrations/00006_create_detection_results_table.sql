-- Migration: Create detection_results table
-- Description: Stores full pipeline run metadata and artifact paths
-- Created: 2026-03-15

CREATE TABLE IF NOT EXISTS public.detection_results (
    id BIGSERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    input_video_path TEXT,
    output_video_path TEXT,
    metadata_path TEXT,
    summary_stats JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(20) NOT NULL DEFAULT 'processing',
    processing_time_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT fk_detection_results_match
        FOREIGN KEY (match_id)
        REFERENCES public.matches(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_detection_results_user
        FOREIGN KEY (user_id)
        REFERENCES public.users(id)
        ON DELETE CASCADE,

    CONSTRAINT check_detection_result_status
        CHECK (status IN ('processing', 'completed', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_detection_results_match_id ON public.detection_results(match_id);
CREATE INDEX IF NOT EXISTS idx_detection_results_user_id ON public.detection_results(user_id);
CREATE INDEX IF NOT EXISTS idx_detection_results_status ON public.detection_results(status);
CREATE INDEX IF NOT EXISTS idx_detection_results_created_at ON public.detection_results(created_at);

DROP TRIGGER IF EXISTS update_detection_results_updated_at ON public.detection_results;
CREATE TRIGGER update_detection_results_updated_at
    BEFORE UPDATE ON public.detection_results
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

ALTER TABLE public.detection_results ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Enable read access for all users" ON public.detection_results;
CREATE POLICY "Enable read access for all users" ON public.detection_results
    FOR SELECT
    USING (true);

DROP POLICY IF EXISTS "Enable insert for all users" ON public.detection_results;
CREATE POLICY "Enable insert for all users" ON public.detection_results
    FOR INSERT
    WITH CHECK (true);

DROP POLICY IF EXISTS "Enable update for all users" ON public.detection_results;
CREATE POLICY "Enable update for all users" ON public.detection_results
    FOR UPDATE
    USING (true);

GRANT SELECT, INSERT, UPDATE ON public.detection_results TO anon, authenticated;
GRANT USAGE, SELECT ON SEQUENCE detection_results_id_seq TO anon, authenticated;

COMMENT ON TABLE public.detection_results IS 'Stores full pipeline run metadata and artifact paths';
COMMENT ON COLUMN public.detection_results.match_id IS 'Associated match identifier';
COMMENT ON COLUMN public.detection_results.user_id IS 'User who initiated detection';
COMMENT ON COLUMN public.detection_results.summary_stats IS 'Computed aggregate stats for the analyzed video';
COMMENT ON COLUMN public.detection_results.status IS 'Pipeline lifecycle state: processing/completed/failed';
