-- Ensure matches are user-owned so API can scope data per authenticated user.
-- Date: 2026-03-15

ALTER TABLE public.matches
ADD COLUMN IF NOT EXISTS user_id BIGINT;

-- Backfill legacy rows to a deterministic owner (oldest user) so NOT NULL can be enforced.
-- If users table is empty, this update is a no-op and constraint enforcement is skipped.
WITH first_user AS (
    SELECT id FROM public.users ORDER BY id ASC LIMIT 1
)
UPDATE public.matches m
SET user_id = fu.id
FROM first_user fu
WHERE m.user_id IS NULL;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM public.users LIMIT 1)
       AND NOT EXISTS (SELECT 1 FROM public.matches WHERE user_id IS NULL) THEN
        ALTER TABLE public.matches
        ALTER COLUMN user_id SET NOT NULL;
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'fk_matches_user'
          AND conrelid = 'public.matches'::regclass
    ) THEN
        ALTER TABLE public.matches
        ADD CONSTRAINT fk_matches_user
        FOREIGN KEY (user_id)
        REFERENCES public.users(id)
        ON DELETE CASCADE;
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_matches_user_id ON public.matches(user_id);
CREATE INDEX IF NOT EXISTS idx_matches_user_date ON public.matches(user_id, date DESC);
