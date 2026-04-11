-- Add client command nonce support to match_commands.
-- This fixes schema-cache errors when the frontend selects or writes command_nonce.

ALTER TABLE public.match_commands
ADD COLUMN IF NOT EXISTS command_nonce TEXT;

COMMENT ON COLUMN public.match_commands.command_nonce IS
    'Client-generated nonce for deduplicating match command submissions.';