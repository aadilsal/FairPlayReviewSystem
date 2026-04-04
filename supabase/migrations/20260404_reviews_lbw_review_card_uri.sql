-- LBW review card image stored in Supabase Storage (review-videos bucket); this column holds the object path.
ALTER TABLE public.reviews
ADD COLUMN IF NOT EXISTS lbw_review_card_uri TEXT;

COMMENT ON COLUMN public.reviews.lbw_review_card_uri IS 'Storage object path for LBW review card JPEG (same bucket convention as video_uri)';
