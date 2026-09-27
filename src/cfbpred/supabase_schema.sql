-- Serving layer for the CFB prediction engine.
--
-- Design note: this database holds REFERENCE data and MODEL OUTPUTS, not the
-- training corpus. The 1.57M-row play table stays in local Parquet and is read
-- with DuckDB, because feature engineering and cross-validation scan it many
-- times and local reads measured ~9x faster with no network variance. What
-- lives here is what benefits from being hosted: things a chart or an app reads.

create schema if not exists cfb;

-- ---------- reference ----------

create table if not exists cfb.teams (
  team_id         integer primary key,
  school          text not null,
  mascot          text,
  abbreviation    text,
  conference      text,
  classification  text,
  color           text,
  alt_color       text,
  logo            text
);

create table if not exists cfb.venues (
  venue_id      integer primary key,
  name          text,
  city          text,
  state         text,
  timezone      text,
  latitude      double precision,
  longitude     double precision,
  elevation     double precision,
  capacity      integer,
  dome          boolean,
  grass         boolean
);

create table if not exists cfb.games (
  game_id            bigint primary key,
  season             smallint not null,
  week               smallint,
  season_type        text,
  start_date         timestamptz,
  start_time_tbd     boolean,
  neutral_site       boolean,
  conference_game    boolean,
  venue_id           integer references cfb.venues(venue_id),
  home_team_id       integer,
  home_team          text,
  home_conference    text,
  home_classification text,
  home_points        smallint,
  away_team_id       integer,
  away_team          text,
  away_conference    text,
  away_classification text,
  away_points        smallint,
  -- derived, stored for convenience
  margin             smallint generated always as (home_points - away_points) stored,
  total              smallint generated always as (home_points + away_points) stored
);
create index if not exists games_season_week_idx on cfb.games (season, week);

-- Point-in-time opponent-adjusted ratings (stage 4 output).
create table if not exists cfb.team_ratings (
  season        smallint not null,
  through_week  smallint not null,
  team_id       integer not null,
  team          text not null,
  off_adj       real,
  def_adj       real,
  overall       real,
  plays         integer,
  model_version text not null,
  primary key (season, through_week, team_id, model_version)
);

-- ---------- model bookkeeping ----------

create table if not exists cfb.model_runs (
  run_id          uuid primary key default gen_random_uuid(),
  model_version   text not null,
  trained_at      timestamptz not null default now(),
  train_seasons   int4range,
  test_seasons    int4range,
  n_params        integer,
  notes           text,
  -- headline held-out metrics, so a run is self-describing
  margin_rmse     real,
  wp_logloss_all  real,
  wp_logloss_q1   real,
  wp_ece_q1       real
);

-- ---------- outputs ----------

-- One row per game: the pregame prediction.
create table if not exists cfb.game_predictions (
  run_id        uuid not null references cfb.model_runs(run_id) on delete cascade,
  game_id       bigint not null references cfb.games(game_id),
  model_version text not null,
  home_win_prob real not null check (home_win_prob between 0 and 1),
  pred_margin   real not null,
  pred_total    real not null,
  pred_home_pts smallint not null,
  pred_away_pts smallint not null,
  primary key (run_id, game_id)
);

-- The win probability curve: one row per play. This is what the chart reads.
create table if not exists cfb.play_predictions (
  run_id            uuid not null references cfb.model_runs(run_id) on delete cascade,
  game_id           bigint not null,
  play_index        integer not null,
  period            smallint,
  clock_seconds     integer,
  game_seconds_left integer,
  home_lead         smallint,
  pos_is_home       boolean,
  down              smallint,
  distance          smallint,
  yards_to_goal     smallint,
  home_win_prob     real not null check (home_win_prob between 0 and 1),
  pred_margin       real,
  pred_total        real,
  play_text         text,
  primary key (run_id, game_id, play_index)
);
create index if not exists play_pred_game_idx on cfb.play_predictions (run_id, game_id, play_index);

-- Optional: the full discrete margin distribution behind a prediction, so the
-- consistency of WP with the margin PMF is inspectable rather than asserted.
create table if not exists cfb.margin_pmf (
  run_id     uuid not null references cfb.model_runs(run_id) on delete cascade,
  game_id    bigint not null,
  play_index integer not null,
  probs      real[] not null,   -- 115 bins: [-inf,-57], -56..56, [57,inf]
  primary key (run_id, game_id, play_index)
);

-- ---------- read-only public access for the chart ----------
alter table cfb.teams             enable row level security;
alter table cfb.venues            enable row level security;
alter table cfb.games             enable row level security;
alter table cfb.team_ratings      enable row level security;
alter table cfb.model_runs        enable row level security;
alter table cfb.game_predictions  enable row level security;
alter table cfb.play_predictions  enable row level security;
alter table cfb.margin_pmf        enable row level security;

do $$
declare t text;
begin
  foreach t in array array['teams','venues','games','team_ratings','model_runs',
                           'game_predictions','play_predictions','margin_pmf']
  loop
    execute format(
      'drop policy if exists %I on cfb.%I', 'anon_read_' || t, t);
    execute format(
      'create policy %I on cfb.%I for select to anon, authenticated using (true)',
      'anon_read_' || t, t);
  end loop;
end $$;

grant usage on schema cfb to anon, authenticated;
grant select on all tables in schema cfb to anon, authenticated;
