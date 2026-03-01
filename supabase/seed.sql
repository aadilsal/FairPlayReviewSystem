-- ========================================
-- FairPlay Review System - Sample/Seed Data
-- ========================================
-- This script inserts sample data for testing purposes
-- Run this AFTER setup.sql if you want to populate with test data
-- Created: 2026-03-01
-- ========================================

-- ========================================
-- 1. INSERT SAMPLE USERS
-- ========================================
-- Note: Password is 'password123' hashed with bcrypt
-- You should change these passwords in production

INSERT INTO public.users (username, email, password_hash, avatar) VALUES
('john_doe', 'john@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5NU7nnpHJD.5u', 'https://ui-avatars.com/api/?name=John+Doe'),
('jane_smith', 'jane@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5NU7nnpHJD.5u', 'https://ui-avatars.com/api/?name=Jane+Smith'),
('cricket_fan', 'fan@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5NU7nnpHJD.5u', 'https://ui-avatars.com/api/?name=Cricket+Fan'),
('admin_user', 'admin@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5NU7nnpHJD.5u', 'https://ui-avatars.com/api/?name=Admin+User')
ON CONFLICT (email) DO NOTHING;

-- ========================================
-- 2. INSERT SAMPLE MATCHES
-- ========================================

INSERT INTO public.matches (team_a, team_b, date, status) VALUES
('Pakistan', 'India', '2026-03-15 14:00:00+00', 'scheduled'),
('England', 'Australia', '2026-03-20 10:00:00+00', 'scheduled'),
('South Africa', 'New Zealand', '2026-03-10 12:00:00+00', 'completed'),
('West Indies', 'Sri Lanka', '2026-03-05 09:00:00+00', 'completed'),
('Bangladesh', 'Afghanistan', '2026-03-25 15:00:00+00', 'scheduled'),
('India', 'Australia', '2026-04-01 13:00:00+00', 'scheduled'),
('Pakistan', 'England', '2026-04-05 11:00:00+00', 'scheduled')
ON CONFLICT DO NOTHING;

-- ========================================
-- 3. INSERT SAMPLE REVIEWS
-- ========================================

INSERT INTO public.reviews (match_id, user_id, content, analysis) VALUES
(3, 1, 'Great match! The umpiring was fair throughout.', 'Analysis: No controversial decisions detected. Fair play rating: 9/10'),
(3, 2, 'Some close LBW calls, but overall good sportsmanship.', 'Analysis: Minor appeals, all handled correctly. Fair play rating: 8/10'),
(4, 1, 'Questionable run-out decision in the 15th over.', 'Analysis: Review suggests possible error. Fair play rating: 6/10'),
(4, 3, 'Excellent bowling performance, clean match.', 'Analysis: No fairness concerns detected. Fair play rating: 10/10'),
(3, 4, 'Both teams showed great spirit!', 'Analysis: Exemplary conduct throughout. Fair play rating: 10/10')
ON CONFLICT DO NOTHING;

-- ========================================
-- 4. INSERT SAMPLE NOTIFICATIONS
-- ========================================

INSERT INTO public.notifications (user_id, message, read) VALUES
(1, 'Welcome to FairPlay Review System!', false),
(1, 'New match scheduled: Pakistan vs India on March 15', false),
(2, 'Your review for South Africa vs New Zealand has been analyzed', true),
(2, 'Welcome to FairPlay Review System!', false),
(3, 'New analysis available for West Indies vs Sri Lanka', false),
(4, 'System maintenance scheduled for tonight', false)
ON CONFLICT DO NOTHING;

-- ========================================
-- SEED DATA INSERTED
-- ========================================
-- Test data has been inserted into all tables
-- You can now test your API endpoints with this data

-- ========================================
-- SAMPLE QUERIES FOR TESTING
-- ========================================

-- Get all users
-- SELECT * FROM public.users;

-- Get upcoming matches
-- SELECT * FROM public.matches WHERE status = 'scheduled' ORDER BY date;

-- Get reviews with user and match info
-- SELECT 
--     r.id, r.content, r.analysis,
--     u.username, u.email,
--     m.team_a, m.team_b, m.date
-- FROM public.reviews r
-- JOIN public.users u ON r.user_id = u.id
-- JOIN public.matches m ON r.match_id = m.id;

-- Get unread notifications for a user
-- SELECT * FROM public.notifications WHERE user_id = 1 AND read = false;
