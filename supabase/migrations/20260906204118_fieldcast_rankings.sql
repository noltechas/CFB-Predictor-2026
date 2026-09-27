-- Published model output only. Neither training nor game predictions read it.
create table public.fieldcast_rankings (
  season integer not null check (season between 2000 and 2200),
  period_start date not null,
  as_of timestamptz not null,
  generated_at timestamptz not null,
  model_version text not null,
  model_fingerprint text not null,
  method_version text not null,
  reference_hash text not null,
  reference_count integer not null check (reference_count > 0),
  n_members integer not null check (n_members > 0),
  entries jsonb not null check (jsonb_typeof(entries) = 'array'),
  diagnostics jsonb not null check (jsonb_typeof(diagnostics) = 'object'),
  primary key (season, period_start)
);
alter table public.fieldcast_rankings enable row level security;
revoke all on public.fieldcast_rankings from anon, authenticated;
grant select on public.fieldcast_rankings to anon, authenticated;
create policy fieldcast_rankings_read on public.fieldcast_rankings
  for select to anon, authenticated using (true);
comment on table public.fieldcast_rankings is
  'Weekly output of the frozen pregame ensemble, measured against a common FBS pool. Never a model input.';
