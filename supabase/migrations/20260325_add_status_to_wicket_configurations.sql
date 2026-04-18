-- Migration: Add status fields to wicket_configurations
-- Description: Enable async/background wicket detection with polling.
-- Created: 2026-03-25

ALTER TABLE public.wicket_configurations
ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'idle',
ADD COLUMN IF NOT EXISTS error_message TEXT;

-- Add/replace check constraint
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'check_wicket_config_status'
          AND conrelid = 'public.wicket_configurations'::regclass
    ) THEN
        ALTER TABLE public.wicket_configurations DROP CONSTRAINT check_wicket_config_status;
    END IF;
END
$$;

ALTER TABLE public.wicket_configurations
ADD CONSTRAINT check_wicket_config_status
CHECK (status IN ('idle', 'processing', 'completed', 'failed'));

CREATE INDEX IF NOT EXISTS idx_wicket_configurations_status ON public.wicket_configurations(status);

COMMENT ON COLUMN public.wicket_configurations.status IS 'Async processing state for auto-detection: idle/processing/completed/failed';
COMMENT ON COLUMN public.wicket_configurations.error_message IS 'Error detail when status=failed';
