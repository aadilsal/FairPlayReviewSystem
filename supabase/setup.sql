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
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    avatar TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Users table indexes
CREATE INDEX IF NOT EXISTS idx_users_username ON public.users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON public.users(created_at);

-- Users table constraints
ALTER TABLE public.users 
ADD CONSTRAINT check_email_format 
CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$');

-- Users table trigger
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON public.users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

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

-- Matches table constraints
ALTER TABLE public.matches 
ADD CONSTRAINT check_match_status 
CHECK (status IN ('scheduled', 'in_progress', 'completed', 'cancelled', 'postponed'));

-- Matches table trigger
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
CREATE TRIGGER update_notifications_updated_at
    BEFORE UPDATE ON public.notifications
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ========================================
-- 6. ENABLE ROW LEVEL SECURITY (RLS)
-- ========================================

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reviews ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notifications ENABLE ROW LEVEL SECURITY;

-- ========================================
-- 7. CREATE RLS POLICIES - USERS
-- ========================================

CREATE POLICY "Enable insert for signup" ON public.users
    FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Users can view all profiles" ON public.users
    FOR SELECT
    USING (true);

CREATE POLICY "Users can update own profile" ON public.users
    FOR UPDATE
    USING (true);

-- ========================================
-- 8. CREATE RLS POLICIES - MATCHES
-- ========================================

CREATE POLICY "Enable read access for all users" ON public.matches
    FOR SELECT
    USING (true);

CREATE POLICY "Enable insert for all users" ON public.matches
    FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Enable update for all users" ON public.matches
    FOR UPDATE
    USING (true);

CREATE POLICY "Enable delete for all users" ON public.matches
    FOR DELETE
    USING (true);

-- ========================================
-- 9. CREATE RLS POLICIES - REVIEWS
-- ========================================

CREATE POLICY "Enable read access for all users" ON public.reviews
    FOR SELECT
    USING (true);

CREATE POLICY "Enable insert for all users" ON public.reviews
    FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Enable update for all users" ON public.reviews
    FOR UPDATE
    USING (true);

CREATE POLICY "Enable delete for all users" ON public.reviews
    FOR DELETE
    USING (true);

-- ========================================
-- 10. CREATE RLS POLICIES - NOTIFICATIONS
-- ========================================

CREATE POLICY "Enable insert for all users" ON public.notifications
    FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Users can view all notifications" ON public.notifications
    FOR SELECT
    USING (true);

CREATE POLICY "Enable update for all users" ON public.notifications
    FOR UPDATE
    USING (true);

CREATE POLICY "Enable delete for all users" ON public.notifications
    FOR DELETE
    USING (true);

-- ========================================
-- 11. GRANT PERMISSIONS
-- ========================================

GRANT SELECT, INSERT, UPDATE ON public.users TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.matches TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.reviews TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.notifications TO anon, authenticated;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;

-- ========================================
-- 12. ADD COMMENTS TO TABLES
-- ========================================

COMMENT ON TABLE public.users IS 'Stores user account information';
COMMENT ON TABLE public.matches IS 'Stores cricket match information';
COMMENT ON TABLE public.reviews IS 'Stores user reviews and AI analysis for matches';
COMMENT ON TABLE public.notifications IS 'Stores user notifications';

-- ========================================
-- SETUP COMPLETE
-- ========================================
-- All tables, indexes, constraints, and policies have been created
-- Your database is now ready to use with the FairPlay Review System API
