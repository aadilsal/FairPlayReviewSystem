-- Align matches schema with frontend payload
-- Frontend sends: name, teams, venue(nullable), date, status(optional)
-- Date: 2026-03-15

ALTER TABLE public.matches
ADD COLUMN IF NOT EXISTS name VARCHAR(255),
ADD COLUMN IF NOT EXISTS teams VARCHAR(255),
ADD COLUMN IF NOT EXISTS venue VARCHAR(255);

-- Make legacy split-team columns nullable for compatibility with teams payload.
ALTER TABLE public.matches
ALTER COLUMN team_a DROP NOT NULL,
ALTER COLUMN team_b DROP NOT NULL;

-- Backfill teams from legacy columns where needed.
UPDATE public.matches
SET teams = COALESCE(teams, CONCAT_WS(' vs ', team_a, team_b))
WHERE teams IS NULL;

-- Backfill name for existing rows.
UPDATE public.matches
SET name = COALESCE(name, teams)
WHERE name IS NULL;

-- Normalize default status for frontend.
ALTER TABLE public.matches
ALTER COLUMN status SET DEFAULT 'upcoming';

-- Replace status check to include frontend status 'upcoming'.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'check_match_status'
          AND conrelid = 'public.matches'::regclass
    ) THEN
        ALTER TABLE public.matches DROP CONSTRAINT check_match_status;
    END IF;
END
$$;

ALTER TABLE public.matches
ADD CONSTRAINT check_match_status
CHECK (status IN ('upcoming', 'scheduled', 'in_progress', 'completed', 'cancelled', 'postponed'));
