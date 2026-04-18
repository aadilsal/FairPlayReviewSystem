-- Migration: Add annotated image path/object fields to wicket_configurations
-- Description: Persist annotated wicket detection preview image reference.
-- Created: 2026-03-25

ALTER TABLE public.wicket_configurations
ADD COLUMN IF NOT EXISTS annotated_image_path TEXT,
ADD COLUMN IF NOT EXISTS annotated_image_object_path TEXT,
ADD COLUMN IF NOT EXISTS source_image_object_path TEXT;

COMMENT ON COLUMN public.wicket_configurations.annotated_image_path IS 'Local filesystem path of the annotated wicket image (debug/dev only)';
COMMENT ON COLUMN public.wicket_configurations.annotated_image_object_path IS 'Supabase Storage object path for annotated wicket image (preferred for frontend)';
COMMENT ON COLUMN public.wicket_configurations.source_image_object_path IS 'Supabase Storage object path for original uploaded source image';
