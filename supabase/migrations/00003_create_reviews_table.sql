-- Migration: Create reviews table
-- Description: Stores user reviews and analysis for matches
-- Created: 2026-03-01

CREATE TABLE IF NOT EXISTS public.reviews (
    id BIGSERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    content TEXT NOT NULL,
    analysis TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Foreign key constraints
    CONSTRAINT fk_reviews_match
        FOREIGN KEY (match_id)
        REFERENCES public.matches(id)
        ON DELETE CASCADE,
    
    CONSTRAINT fk_reviews_user
        FOREIGN KEY (user_id)
        REFERENCES public.users(id)
        ON DELETE CASCADE
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_reviews_match_id ON public.reviews(match_id);
CREATE INDEX IF NOT EXISTS idx_reviews_user_id ON public.reviews(user_id);
CREATE INDEX IF NOT EXISTS idx_reviews_created_at ON public.reviews(created_at);
CREATE INDEX IF NOT EXISTS idx_reviews_match_user ON public.reviews(match_id, user_id);

-- Add updated_at trigger
CREATE TRIGGER update_reviews_updated_at
    BEFORE UPDATE ON public.reviews
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Enable Row Level Security (RLS)
ALTER TABLE public.reviews ENABLE ROW LEVEL SECURITY;

-- Create policies for RLS
-- Anyone can read reviews
CREATE POLICY "Enable read access for all users" ON public.reviews
    FOR SELECT
    USING (true);

-- Authenticated users can create reviews
CREATE POLICY "Authenticated users can insert reviews" ON public.reviews
    FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid()::text = user_id::text);

-- Users can update their own reviews
CREATE POLICY "Users can update own reviews" ON public.reviews
    FOR UPDATE
    TO authenticated
    USING (auth.uid()::text = user_id::text);

-- Users can delete their own reviews
CREATE POLICY "Users can delete own reviews" ON public.reviews
    FOR DELETE
    TO authenticated
    USING (auth.uid()::text = user_id::text);

-- Grant permissions
GRANT SELECT ON public.reviews TO anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.reviews TO authenticated;
GRANT USAGE, SELECT ON SEQUENCE reviews_id_seq TO anon;
GRANT USAGE, SELECT ON SEQUENCE reviews_id_seq TO authenticated;

COMMENT ON TABLE public.reviews IS 'Stores user reviews and AI analysis for matches';
COMMENT ON COLUMN public.reviews.id IS 'Unique review identifier';
COMMENT ON COLUMN public.reviews.match_id IS 'Reference to the match being reviewed';
COMMENT ON COLUMN public.reviews.user_id IS 'Reference to the user who created the review';
COMMENT ON COLUMN public.reviews.content IS 'Review content/description';
COMMENT ON COLUMN public.reviews.analysis IS 'AI-generated fairplay analysis';
COMMENT ON COLUMN public.reviews.created_at IS 'Review creation timestamp';
COMMENT ON COLUMN public.reviews.updated_at IS 'Last update timestamp';
