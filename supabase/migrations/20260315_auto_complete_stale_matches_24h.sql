-- Auto-complete stale in-progress matches after inactivity timeout.
-- Date: 2026-03-15

ALTER TABLE public.matches
ADD COLUMN IF NOT EXISTS completed_by_system BOOLEAN NOT NULL DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS auto_completed_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS completion_reason VARCHAR(100);

CREATE INDEX IF NOT EXISTS idx_matches_status_updated_at ON public.matches(status, updated_at);

CREATE OR REPLACE FUNCTION public.auto_complete_stale_matches(
    timeout_hours INTEGER DEFAULT 24,
    target_user_id BIGINT DEFAULT NULL
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    safe_hours INTEGER := GREATEST(timeout_hours, 1);
    cutoff_time TIMESTAMP WITH TIME ZONE := NOW() - make_interval(hours => GREATEST(timeout_hours, 1));
    completed_count INTEGER := 0;
    notified_count INTEGER := 0;
BEGIN
    WITH stale_matches AS (
        SELECT
            m.id,
            m.user_id,
            COALESCE(NULLIF(m.name, ''), NULLIF(m.teams, ''), CONCAT_WS(' vs ', m.team_a, m.team_b), 'Match') AS display_name
        FROM public.matches m
        WHERE m.status = 'in_progress'
          AND m.updated_at <= cutoff_time
          AND (target_user_id IS NULL OR m.user_id = target_user_id)
    ),
    updated_matches AS (
        UPDATE public.matches m
        SET
            status = 'completed',
            completed_by_system = TRUE,
            auto_completed_at = NOW(),
            completion_reason = 'timeout_24h',
            updated_at = NOW()
        FROM stale_matches s
        WHERE m.id = s.id
        RETURNING m.id, m.user_id, s.display_name
    ),
    inserted_notifications AS (
        INSERT INTO public.notifications (user_id, message, read, created_at, updated_at)
        SELECT
            um.user_id,
            FORMAT('Match "%s" was auto-completed after %s hours of inactivity.', um.display_name, safe_hours),
            FALSE,
            NOW(),
            NOW()
        FROM updated_matches um
        RETURNING id
    )
    SELECT
        (SELECT COUNT(*) FROM updated_matches),
        (SELECT COUNT(*) FROM inserted_notifications)
    INTO completed_count, notified_count;

    RETURN jsonb_build_object(
        'timeout_hours', safe_hours,
        'completed_count', completed_count,
        'notified_count', notified_count,
        'target_user_id', target_user_id
    );
END;
$$;

GRANT EXECUTE ON FUNCTION public.auto_complete_stale_matches(INTEGER, BIGINT) TO authenticated;

-- Optional scheduler setup if pg_cron is enabled in your Supabase project.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        IF NOT EXISTS (SELECT 1 FROM cron.job WHERE jobname = 'matches-auto-complete-24h') THEN
            PERFORM cron.schedule(
                'matches-auto-complete-24h',
                '*/30 * * * *',
                'SELECT public.auto_complete_stale_matches(24, NULL);'
            );
        END IF;
    END IF;
EXCEPTION
    WHEN undefined_table OR invalid_schema_name THEN
        RAISE NOTICE 'pg_cron metadata is unavailable; schedule this SQL manually in Supabase Cron UI.';
END
$$;
