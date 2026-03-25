-- Migration: Improve auth-to-users username sync and backfill full names
-- Description: Prefer username/name metadata from auth.users for public.users.username and sync password_hash from auth.users.encrypted_password
-- Created: 2026-03-15

-- Recreate create trigger function with better username derivation.
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

-- Recreate update trigger function to sync username when profile metadata changes.
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

-- Backfill usernames for already-synced rows where username is auto-generated/empty.
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

-- Note: OAuth or magic-link users may not have encrypted_password in auth.users.