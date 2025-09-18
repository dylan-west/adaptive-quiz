-- Ensure one interaction per user+item
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'uniq_interactions_user_item'
  ) THEN
    ALTER TABLE interactions
    ADD CONSTRAINT uniq_interactions_user_item UNIQUE (user_id, item_id);
  END IF;
END $$;