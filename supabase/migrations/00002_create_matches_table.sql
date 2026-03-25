-- Migration: Create matches table
-- Description: Stores cricket match information
-- Created: 2026-03-01

CREATE TABLE IF NOT EXISTS public.matches (
    id BIGSERIAL PRIMARY KEY,
    team_a VARCHAR(255) NOT NULL,
    team_b VARCHAR(255) NOT NULL,
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'scheduled',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_matches_date ON public.matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_status ON public.matches(status);
CREATE INDEX IF NOT EXISTS idx_matches_created_at ON public.matches(created_at);

-- Add status constraint to ensure valid values
ALTER TABLE public.matches 
ADD CONSTRAINT check_match_status 
CHECK (status IN ('scheduled', 'in_progress', 'completed', 'cancelled', 'postponed'));

-- Add updated_at trigger
CREATE TRIGGER update_matches_updated_at
    BEFORE UPDATE ON public.matches
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Enable Row Level Security (RLS)
ALTER TABLE public.matches ENABLE ROW LEVEL SECURITY;

-- Create policies for RLS
-- Anyone can read matches
CREATE POLICY "Enable read access for all users" ON public.matches
    FOR SELECT
    USING (true);

-- Only authenticated users can create matches
CREATE POLICY "Authenticated users can insert matches" ON public.matches
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Only authenticated users can update matches
CREATE POLICY "Authenticated users can update matches" ON public.matches
    FOR UPDATE
    TO authenticated
    USING (true);

-- Only authenticated users can delete matches
CREATE POLICY "Authenticated users can delete matches" ON public.matches
    FOR DELETE
    TO authenticated
    USING (true);

-- Grant permissions
GRANT SELECT ON public.matches TO anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.matches TO authenticated;
GRANT USAGE, SELECT ON SEQUENCE matches_id_seq TO anon;
GRANT USAGE, SELECT ON SEQUENCE matches_id_seq TO authenticated;

COMMENT ON TABLE public.matches IS 'Stores cricket match information';
COMMENT ON COLUMN public.matches.id IS 'Unique match identifier';
COMMENT ON COLUMN public.matches.team_a IS 'Name of first team';
COMMENT ON COLUMN public.matches.team_b IS 'Name of second team';
COMMENT ON COLUMN public.matches.date IS 'Match date and time';
COMMENT ON COLUMN public.matches.status IS 'Current status of the match';
COMMENT ON COLUMN public.matches.created_at IS 'Match creation timestamp';
COMMENT ON COLUMN public.matches.updated_at IS 'Last update timestamp';
