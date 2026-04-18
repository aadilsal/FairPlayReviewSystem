-- Migration to harmonize schema with frontend requirements
-- Date: 2026-03-15

-- 1. Update Matches Table
ALTER TABLE public.matches 
ADD COLUMN IF NOT EXISTS name VARCHAR(255),
ADD COLUMN IF NOT EXISTS teams VARCHAR(255),
ADD COLUMN IF NOT EXISTS venue VARCHAR(255);

-- Update existing matches to populate teams from team_a and team_b
UPDATE public.matches 
SET teams = team_a || ' vs ' || team_b 
WHERE teams IS NULL AND team_a IS NOT NULL AND team_b IS NOT NULL;

-- 2. Update Reviews Table
ALTER TABLE public.reviews
ADD COLUMN IF NOT EXISTS over VARCHAR(50),
ADD COLUMN IF NOT EXISTS original_decision VARCHAR(100),
ADD COLUMN IF NOT EXISTS decision VARCHAR(100),
ADD COLUMN IF NOT EXISTS impact VARCHAR(100),
ADD COLUMN IF NOT EXISTS pitch VARCHAR(100),
ADD COLUMN IF NOT EXISTS wickets VARCHAR(100),
ADD COLUMN IF NOT EXISTS video_uri TEXT;

-- 3. Create Notification Settings Table
CREATE TABLE IF NOT EXISTS public.notification_settings (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL UNIQUE,
    match_alerts BOOLEAN DEFAULT TRUE,
    review_updates BOOLEAN DEFAULT TRUE,
    system_notifications BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT fk_notification_settings_user
        FOREIGN KEY (user_id)
        REFERENCES public.users(id)
        ON DELETE CASCADE
);

-- Add updated_at trigger for notification_settings
DROP TRIGGER IF EXISTS update_notification_settings_updated_at ON public.notification_settings;
CREATE TRIGGER update_notification_settings_updated_at
    BEFORE UPDATE ON public.notification_settings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Enable RLS for notification_settings
ALTER TABLE public.notification_settings ENABLE ROW LEVEL SECURITY;

-- RLS Policies for notification_settings
DROP POLICY IF EXISTS "Users can view own notification settings" ON public.notification_settings;
CREATE POLICY "Users can view own notification settings" ON public.notification_settings
    FOR SELECT
    USING (user_id IN (
        SELECT id FROM public.users WHERE auth_user_id = auth.uid()
    ));

DROP POLICY IF EXISTS "Users can update own notification settings" ON public.notification_settings;
CREATE POLICY "Users can update own notification settings" ON public.notification_settings
    FOR UPDATE
    USING (user_id IN (
        SELECT id FROM public.users WHERE auth_user_id = auth.uid()
    ));

DROP POLICY IF EXISTS "System can insert notification settings" ON public.notification_settings;
CREATE POLICY "System can insert notification settings" ON public.notification_settings
    FOR INSERT
    WITH CHECK (true);

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON public.notification_settings TO anon, authenticated;
