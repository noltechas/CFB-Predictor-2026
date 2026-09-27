-- One complete, owner-published list per football week. App clients only read.
create table public.editorial_rankings (
  season integer not null check (season between 2000 and 2200),
  ranking_week integer not null check (ranking_week between 0 and 40),
  author text not null check (length(btrim(author)) between 1 and 120),
  title text not null check (length(btrim(title)) between 1 and 200),
  introduction text check (length(introduction) <= 4000),
  published_at timestamptz not null default now(),
  entries jsonb not null check (jsonb_typeof(entries) = 'array' and jsonb_array_length(entries) between 1 and 1000),
  primary key (season, ranking_week)
);
alter table public.editorial_rankings enable row level security;
revoke all on public.editorial_rankings from public, anon, authenticated;
grant select on public.editorial_rankings to anon, authenticated;
grant all on public.editorial_rankings to service_role;
create policy editorial_rankings_read on public.editorial_rankings
  for select to anon, authenticated using (true);
comment on table public.editorial_rankings is
  'Chas’s published editorial rankings. Separate from model rankings and never used as model input.';
