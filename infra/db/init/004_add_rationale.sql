-- Add rationale to items if it doesn't exist yet
ALTER TABLE IF EXISTS items
  ADD COLUMN IF NOT EXISTS rationale TEXT;
