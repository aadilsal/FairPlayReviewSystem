-- Final synchronization migration
-- Date: 2026-03-15

-- 1. Update Reviews Table
ALTER TABLE public.reviews 
ADD COLUMN IF NOT EXISTS match_name VARCHAR(255);

-- 2. Update Detection Results Table
ALTER TABLE public.detection_results
ADD COLUMN IF NOT EXISTS result_data JSONB DEFAULT '{}'::jsonb;

-- Populate match_name for existing reviews from matches table
UPDATE public.reviews r
SET match_name = m.name
FROM public.matches m
WHERE r.match_id = m.id AND r.match_name IS NULL;
