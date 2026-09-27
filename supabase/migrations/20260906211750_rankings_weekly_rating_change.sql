-- A Sunday refresh and the following Thursday can belong to different
-- football weeks. Keep the previous football week's rating intact.
alter table public.fieldcast_rankings add column ranking_week integer;
update public.fieldcast_rankings
  set ranking_week = greatest(0, (diagnostics->>'week_context')::integer - 1);
alter table public.fieldcast_rankings alter column ranking_week set not null;
alter table public.fieldcast_rankings add constraint fieldcast_rankings_week_check
  check (ranking_week between 0 and 40);
alter table public.fieldcast_rankings add column snapshot_kind text not null default 'weekly';
update public.fieldcast_rankings set snapshot_kind = 'preseason' where ranking_week = 0;
alter table public.fieldcast_rankings add constraint fieldcast_rankings_kind_check
  check ((ranking_week = 0 and snapshot_kind in ('preseason', 'reconstructed_preseason'))
      or (ranking_week > 0 and snapshot_kind = 'weekly'));
alter table public.fieldcast_rankings drop constraint fieldcast_rankings_pkey;
alter table public.fieldcast_rankings add primary key (season, ranking_week);
comment on column public.fieldcast_rankings.ranking_week is
  '0 = preseason; otherwise the source continuous football week. Comparisons require the immediately previous week.';
