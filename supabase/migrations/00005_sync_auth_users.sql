-- Migration: Sync Supabase Auth users into public.users
-- Description: Keeps auth.users and public.users synchronized on signup/profile updates
-- Created: 2026-03-15

-- 1) Extend users schema for Supabase Auth linkage
ALTER TABLE public.users
ADD COLUMN IF NOT EXISTS auth_user_id UUID;

ALTER TABLE public.users
ALTER COLUMN password_hash DROP NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_auth_user_id
ON public.users(auth_user_id)
WHERE auth_user_id IS NOT NULL;

-- 2) Align existing rows by email where possible
UPDATE public.users u
SET auth_user_id = a.id
FROM auth.users a
WHERE u.auth_user_id IS NULL
  AND lower(u.email) = lower(a.email);

-- 3) Trigger function: create profile row after auth signup
CREATE OR REPLACE FUNCTION public.handle_auth_user_created()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    derived_username TEXT;
BEGIN
    derived_username := COALESCE(
        NEW.raw_user_meta_data ->> 'username',
        split_part(NEW.email, '@', 1) || '_' || left(replace(NEW.id::text, '-', ''), 8)
    );

    INSERT INTO public.users (auth_user_id, username, email, avatar)
    VALUES (
        NEW.id,
        derived_username,
        NEW.email,
        NEW.raw_user_meta_data ->> 'avatar'
    )
    ON CONFLICT (email)
    DO UPDATE SET
        auth_user_id = COALESCE(public.users.auth_user_id, EXCLUDED.auth_user_id),
        username = COALESCE(public.users.username, EXCLUDED.username),
        avatar = COALESCE(EXCLUDED.avatar, public.users.avatar),
        updated_at = NOW();

    RETURN NEW;
END;
$$;

-- 4) Trigger function: keep email/avatar in sync when auth user updates
CREATE OR REPLACE FUNCTION public.handle_auth_user_updated()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    UPDATE public.users
    SET
        email = NEW.email,
        avatar = COALESCE(NEW.raw_user_meta_data ->> 'avatar', public.users.avatar),
        updated_at = NOW()
    WHERE auth_user_id = NEW.id;

    RETURN NEW;
END;
$$;

-- 5) Recreate triggers idempotently
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
AFTER INSERT ON auth.users
FOR EACH ROW
EXECUTE FUNCTION public.handle_auth_user_created();

DROP TRIGGER IF EXISTS on_auth_user_updated ON auth.users;
CREATE TRIGGER on_auth_user_updated
AFTER UPDATE OF email, raw_user_meta_data ON auth.users
FOR EACH ROW
EXECUTE FUNCTION public.handle_auth_user_updated();

-- 6) Backfill rows that only exist in auth.users
INSERT INTO public.users (auth_user_id, username, email, avatar)
SELECT
    a.id,
    COALESCE(
        a.raw_user_meta_data ->> 'username',
        split_part(a.email, '@', 1) || '_' || left(replace(a.id::text, '-', ''), 8)
    ),
    a.email,
    a.raw_user_meta_data ->> 'avatar'
FROM auth.users a
LEFT JOIN public.users u
    ON u.auth_user_id = a.id OR lower(u.email) = lower(a.email)
WHERE u.id IS NULL;
