-- Display-only state from the same scoreboard observation as the live score.
alter table public.live_predictions add column if not exists live_context jsonb;
comment on column public.live_predictions.live_context is
  'Observed clock status, possession side, down, distance and field position. Timestamp must match updated_at; never use model-imputed possession for display.';
