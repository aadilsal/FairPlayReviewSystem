-- ========================================
-- FairPlay Review System - Complete Database Setup
-- ========================================
-- This script creates all tables, indexes, and policies for the FairPlay Review System
-- Run this in the Supabase SQL Editor to set up the entire database
-- Created: 2026-03-01
-- ========================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ========================================
-- 1. CREATE UPDATED_AT TRIGGER FUNCTION
-- ========================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- 2. CREATE USERS TABLE
-- ========================================

CREATE TABLE IF NOT EXISTS public.users (
    id BIGSERIAL PRIMARY KEY,
    auth_user_id UUID UNIQUE,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    avatar TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Users table indexes
CREATE INDEX IF NOT EXISTS idx_users_username ON public.users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON public.users(created_at);
CREATE INDEX IF NOT EXISTS idx_users_auth_user_id ON public.users(auth_user_id) WHERE auth_user_id IS NOT NULL;

-- Users table constraints (idempotent)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'check_email_format'
          AND conrelid = 'public.users'::regclass
    ) THEN
        ALTER TABLE public.users
        ADD CONSTRAINT check_email_format
        CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$');
    END IF;
END
$$;

-- Users table trigger
DROP TRIGGER IF EXISTS update_users_updated_at ON public.users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON public.users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Keep public.users in sync with Supabase Auth users
CREATE OR REPLACE FUNCTION public.handle_auth_user_created()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    generated_username TEXT;
    derived_username TEXT;
BEGIN
    generated_username := split_part(NEW.email, '@', 1) || '_' || left(replace(NEW.id::text, '-', ''), 8);
    derived_username := COALESCE(
        NULLIF(btrim(NEW.raw_user_meta_data ->> 'username'), ''),
        NULLIF(btrim(NEW.raw_user_meta_data ->> 'name'), ''),
        generated_username
    );

INSERT INTO public.users (auth_user_id, username, email, password_hash, avatar)
    VALUES (
        NEW.id,
        derived_username,
        NEW.email,
        NEW.encrypted_password,
        NEW.raw_user_meta_data ->> 'avatar'
    )
    ON CONFLICT (email)
    DO UPDATE SET
        auth_user_id = COALESCE(public.users.auth_user_id, EXCLUDED.auth_user_id),
        username = CASE
            WHEN public.users.username IS NULL
              OR btrim(public.users.username) = ''
              OR public.users.username = generated_username
            THEN EXCLUDED.username
            ELSE public.users.username
        END,
        password_hash = COALESCE(EXCLUDED.password_hash, public.users.password_hash),
        avatar = COALESCE(EXCLUDED.avatar, public.users.avatar),
        updated_at = NOW();

    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.handle_auth_user_updated()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    generated_username TEXT;
    derived_username TEXT;
BEGIN
    generated_username := split_part(NEW.email, '@', 1) || '_' || left(replace(NEW.id::text, '-', ''), 8);
    derived_username := COALESCE(
        NULLIF(btrim(NEW.raw_user_meta_data ->> 'username'), ''),
        NULLIF(btrim(NEW.raw_user_meta_data ->> 'name'), ''),
        generated_username
    );

    UPDATE public.users
    SET
        email = NEW.email,
        username = CASE
            WHEN public.users.username IS NULL
              OR btrim(public.users.username) = ''
              OR public.users.username = generated_username
            THEN derived_username
            ELSE public.users.username
        END,
        password_hash = COALESCE(NEW.encrypted_password, public.users.password_hash),
        avatar = COALESCE(NEW.raw_user_meta_data ->> 'avatar', public.users.avatar),
        updated_at = NOW()
    WHERE auth_user_id = NEW.id;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
AFTER INSERT ON auth.users
FOR EACH ROW
EXECUTE FUNCTION public.handle_auth_user_created();

DROP TRIGGER IF EXISTS on_auth_user_updated ON auth.users;
CREATE TRIGGER on_auth_user_updated
AFTER UPDATE OF email, raw_user_meta_data, encrypted_password ON auth.users
FOR EACH ROW
EXECUTE FUNCTION public.handle_auth_user_updated();

-- Backfill any existing auth users into public.users
UPDATE public.users u
SET auth_user_id = a.id
FROM auth.users a
WHERE u.auth_user_id IS NULL
  AND lower(u.email) = lower(a.email);

INSERT INTO public.users (auth_user_id, username, email, avatar)
SELECT
    a.id,
    COALESCE(
        NULLIF(btrim(a.raw_user_meta_data ->> 'username'), ''),
        NULLIF(btrim(a.raw_user_meta_data ->> 'name'), ''),
        split_part(a.email, '@', 1) || '_' || left(replace(a.id::text, '-', ''), 8)
    ),
    a.email,
    a.raw_user_meta_data ->> 'avatar'
FROM auth.users a
LEFT JOIN public.users u
    ON u.auth_user_id = a.id OR lower(u.email) = lower(a.email)
WHERE u.id IS NULL;

-- Backfill usernames for existing synced rows that still use auto-generated fallback values.
WITH auth_map AS (
    SELECT
        a.id AS auth_user_id,
        lower(a.email) AS email_l,
        split_part(a.email, '@', 1) || '_' || left(replace(a.id::text, '-', ''), 8) AS generated_username,
        COALESCE(
            NULLIF(btrim(a.raw_user_meta_data ->> 'username'), ''),
            NULLIF(btrim(a.raw_user_meta_data ->> 'name'), '')
        ) AS desired_username
    FROM auth.users a
)
UPDATE public.users u
SET
    username = m.desired_username,
    updated_at = NOW()
FROM auth_map m
WHERE m.desired_username IS NOT NULL
  AND (
      u.auth_user_id = m.auth_user_id
      OR (u.auth_user_id IS NULL AND lower(u.email) = m.email_l)
  )
  AND (
      u.username IS NULL
      OR btrim(u.username) = ''
      OR u.username = m.generated_username
  );

-- Backfill password_hash from auth.users for already-synced rows.
UPDATE public.users u
SET
    password_hash = a.encrypted_password,
    updated_at = NOW()
FROM auth.users a
WHERE (u.auth_user_id = a.id OR (u.auth_user_id IS NULL AND lower(u.email) = lower(a.email)))
  AND a.encrypted_password IS NOT NULL
  AND (u.password_hash IS NULL OR btrim(u.password_hash) = '');

-- ========================================
-- 3. CREATE MATCHES TABLE
-- ========================================

CREATE TABLE IF NOT EXISTS public.matches (
    id BIGSERIAL PRIMARY KEY,
    team_a VARCHAR(255) NOT NULL,
    team_b VARCHAR(255) NOT NULL,
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'scheduled',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Matches table indexes
CREATE INDEX IF NOT EXISTS idx_matches_date ON public.matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_status ON public.matches(status);
CREATE INDEX IF NOT EXISTS idx_matches_created_at ON public.matches(created_at);

-- Matches table constraints (idempotent)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'check_match_status'
          AND conrelid = 'public.matches'::regclass
    ) THEN
        ALTER TABLE public.matches
        ADD CONSTRAINT check_match_status
        CHECK (status IN ('scheduled', 'in_progress', 'completed', 'cancelled', 'postponed'));
    END IF;
END
$$;

-- Matches table trigger
DROP TRIGGER IF EXISTS update_matches_updated_at ON public.matches;
CREATE TRIGGER update_matches_updated_at
    BEFORE UPDATE ON public.matches
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ========================================
-- 4. CREATE REVIEWS TABLE
-- ========================================

CREATE TABLE IF NOT EXISTS public.reviews (
    id BIGSERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    content TEXT NOT NULL,
    analysis TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT fk_reviews_match
        FOREIGN KEY (match_id)
        REFERENCES public.matches(id)
        ON DELETE CASCADE,
    
    CONSTRAINT fk_reviews_user
        FOREIGN KEY (user_id)
        REFERENCES public.users(id)
        ON DELETE CASCADE
);

-- Reviews table indexes
CREATE INDEX IF NOT EXISTS idx_reviews_match_id ON public.reviews(match_id);
CREATE INDEX IF NOT EXISTS idx_reviews_user_id ON public.reviews(user_id);
CREATE INDEX IF NOT EXISTS idx_reviews_created_at ON public.reviews(created_at);
CREATE INDEX IF NOT EXISTS idx_reviews_match_user ON public.reviews(match_id, user_id);

-- Reviews table trigger
DROP TRIGGER IF EXISTS update_reviews_updated_at ON public.reviews;
CREATE TRIGGER update_reviews_updated_at
    BEFORE UPDATE ON public.reviews
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ========================================
-- 5. CREATE NOTIFICATIONS TABLE
-- ========================================

CREATE TABLE IF NOT EXISTS public.notifications (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    message TEXT NOT NULL,
    read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT fk_notifications_user
        FOREIGN KEY (user_id)
        REFERENCES public.users(id)
        ON DELETE CASCADE
);

-- Notifications table indexes
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON public.notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_read ON public.notifications(read);
CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON public.notifications(created_at);
CREATE INDEX IF NOT EXISTS idx_notifications_user_unread ON public.notifications(user_id, read) WHERE read = FALSE;

-- Notifications table trigger
DROP TRIGGER IF EXISTS update_notifications_updated_at ON public.notifications;
CREATE TRIGGER update_notifications_updated_at
    BEFORE UPDATE ON public.notifications
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ========================================
-- 6. CREATE DETECTION_RESULTS TABLE
-- ========================================

CREATE TABLE IF NOT EXISTS public.detection_results (
    id BIGSERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    input_video_path TEXT,
    output_video_path TEXT,
    metadata_path TEXT,
    summary_stats JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(20) NOT NULL DEFAULT 'processing',
    processing_time_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT fk_detection_results_match
        FOREIGN KEY (match_id)
        REFERENCES public.matches(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_detection_results_user
        FOREIGN KEY (user_id)
        REFERENCES public.users(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_detection_results_match_id ON public.detection_results(match_id);
CREATE INDEX IF NOT EXISTS idx_detection_results_user_id ON public.detection_results(user_id);
CREATE INDEX IF NOT EXISTS idx_detection_results_status ON public.detection_results(status);
CREATE INDEX IF NOT EXISTS idx_detection_results_created_at ON public.detection_results(created_at);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'check_detection_result_status'
          AND conrelid = 'public.detection_results'::regclass
    ) THEN
        ALTER TABLE public.detection_results
        ADD CONSTRAINT check_detection_result_status
        CHECK (status IN ('processing', 'completed', 'failed'));
    END IF;
END
$$;

DROP TRIGGER IF EXISTS update_detection_results_updated_at ON public.detection_results;
CREATE TRIGGER update_detection_results_updated_at
    BEFORE UPDATE ON public.detection_results
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ========================================
-- 7. ENABLE ROW LEVEL SECURITY (RLS)
-- ========================================

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reviews ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.detection_results ENABLE ROW LEVEL SECURITY;

-- ========================================
-- 8. CREATE RLS POLICIES - USERS
-- ========================================

DROP POLICY IF EXISTS "Enable insert for signup" ON public.users;
CREATE POLICY "Enable insert for signup" ON public.users
    FOR INSERT
    WITH CHECK (true);

DROP POLICY IF EXISTS "Users can view all profiles" ON public.users;
CREATE POLICY "Users can view all profiles" ON public.users
    FOR SELECT
    USING (true);

DROP POLICY IF EXISTS "Users can update own profile" ON public.users;
CREATE POLICY "Users can update own profile" ON public.users
    FOR UPDATE
    USING (true);

-- ========================================
-- 9. CREATE RLS POLICIES - MATCHES
-- ========================================

DROP POLICY IF EXISTS "Enable read access for all users" ON public.matches;
CREATE POLICY "Enable read access for all users" ON public.matches
    FOR SELECT
    USING (true);

DROP POLICY IF EXISTS "Enable insert for all users" ON public.matches;
CREATE POLICY "Enable insert for all users" ON public.matches
    FOR INSERT
    WITH CHECK (true);

DROP POLICY IF EXISTS "Enable update for all users" ON public.matches;
CREATE POLICY "Enable update for all users" ON public.matches
    FOR UPDATE
    USING (true);

DROP POLICY IF EXISTS "Enable delete for all users" ON public.matches;
CREATE POLICY "Enable delete for all users" ON public.matches
    FOR DELETE
    USING (true);

-- ========================================
-- 10. CREATE RLS POLICIES - REVIEWS
-- ========================================

DROP POLICY IF EXISTS "Enable read access for all users" ON public.reviews;
CREATE POLICY "Enable read access for all users" ON public.reviews
    FOR SELECT
    USING (true);

DROP POLICY IF EXISTS "Enable insert for all users" ON public.reviews;
CREATE POLICY "Enable insert for all users" ON public.reviews
    FOR INSERT
    WITH CHECK (true);

DROP POLICY IF EXISTS "Enable update for all users" ON public.reviews;
CREATE POLICY "Enable update for all users" ON public.reviews
    FOR UPDATE
    USING (true);

DROP POLICY IF EXISTS "Enable delete for all users" ON public.reviews;
CREATE POLICY "Enable delete for all users" ON public.reviews
    FOR DELETE
    USING (true);

-- ========================================
-- 11. CREATE RLS POLICIES - NOTIFICATIONS
-- ========================================

DROP POLICY IF EXISTS "Enable insert for all users" ON public.notifications;
CREATE POLICY "Enable insert for all users" ON public.notifications
    FOR INSERT
    WITH CHECK (true);

DROP POLICY IF EXISTS "Users can view all notifications" ON public.notifications;
CREATE POLICY "Users can view all notifications" ON public.notifications
    FOR SELECT
    USING (true);

DROP POLICY IF EXISTS "Enable update for all users" ON public.notifications;
CREATE POLICY "Enable update for all users" ON public.notifications
    FOR UPDATE
    USING (true);

DROP POLICY IF EXISTS "Enable delete for all users" ON public.notifications;
CREATE POLICY "Enable delete for all users" ON public.notifications
    FOR DELETE
    USING (true);

-- ========================================
-- 12. CREATE RLS POLICIES - DETECTION RESULTS
-- ========================================

DROP POLICY IF EXISTS "Enable read access for all users" ON public.detection_results;
CREATE POLICY "Enable read access for all users" ON public.detection_results
    FOR SELECT
    USING (true);

DROP POLICY IF EXISTS "Enable insert for all users" ON public.detection_results;
CREATE POLICY "Enable insert for all users" ON public.detection_results
    FOR INSERT
    WITH CHECK (true);

DROP POLICY IF EXISTS "Enable update for all users" ON public.detection_results;
CREATE POLICY "Enable update for all users" ON public.detection_results
    FOR UPDATE
    USING (true);

-- ========================================
-- 13. GRANT PERMISSIONS
-- ========================================

GRANT SELECT, INSERT, UPDATE ON public.users TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.matches TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.reviews TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.notifications TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE ON public.detection_results TO anon, authenticated;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;

-- ========================================
-- 14. ADD COMMENTS TO TABLES
-- ========================================

COMMENT ON TABLE public.users IS 'Stores user account information';
COMMENT ON COLUMN public.users.auth_user_id IS 'Linked Supabase Auth user UUID';
COMMENT ON TABLE public.matches IS 'Stores cricket match information';
COMMENT ON TABLE public.reviews IS 'Stores user reviews and AI analysis for matches';
COMMENT ON TABLE public.notifications IS 'Stores user notifications';
COMMENT ON TABLE public.detection_results IS 'Stores full pipeline run metadata and artifact paths';

-- ========================================
-- SETUP COMPLETE
-- ========================================
-- All tables, indexes, constraints, and policies have been created
-- Your database is now ready to use with the FairPlay Review System API
