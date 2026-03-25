-- Fix user trigger to support 'name' metadata and improve robustness
-- Date: 2026-03-15

CREATE OR REPLACE FUNCTION public.handle_auth_user_created()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    derived_username TEXT;
BEGIN
    -- Look for 'username' or 'name' in metadata, fallback to email-based slug
    derived_username := COALESCE(
        NEW.raw_user_meta_data ->> 'username',
        NEW.raw_user_meta_data ->> 'name',
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
        username = EXCLUDED.username, -- Overwrite with metadata name if available
        avatar = COALESCE(EXCLUDED.avatar, public.users.avatar),
        updated_at = NOW();

    RETURN NEW;
END;
$$;
