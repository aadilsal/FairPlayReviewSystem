-- Migration: Drop UNIQUE constraint on username
-- Description: Allow duplicate usernames; users are differentiated by user_id (id)
-- Created: 2025-03-24

-- Drop the unique constraint on username (PostgreSQL default: users_username_key)
ALTER TABLE public.users DROP CONSTRAINT IF EXISTS users_username_key;

-- Update column comment to reflect new semantics
COMMENT ON COLUMN public.users.username IS 'Display name; not unique - users identified by id';
