"""Shared paths, constants and the play-by-play column allowlist."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PQ = DATA / "parquet"
PBP = PQ / "pbp"
CFBD = PQ / "cfbd"
FEAT = PQ / "features"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
DOCS = ROOT / "docs"

for _p in (RAW, PQ, PBP, CFBD, FEAT, MODELS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data spans
# Play-level: the release product only. The repo product's schema swings between
# 364 and 469 columns season to season and its timeout column is broken
# (25-35% of plays at 3 timeouts, against a true rate near 70%).
PBP_SEASONS = list(range(2014, 2026))
# Game-level: schedules cover 2001-2025 with 100% score coverage and zero ties.
GAME_SEASONS = list(range(2001, 2026))

REL_BASE = ("https://github.com/sportsdataverse/sportsdataverse-data/"
            "releases/download/cfbfastR_cfb_pbp")
SCHED_BASE = ("https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/"
              "main/schedules/parquet")
LINES_URL = ("https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/"
             "main/betting/parquet/cfb_line_odds.parquet")

# ---------------------------------------------------------------- allowlist
# 80 of 362 columns. The full set costs ~14 GB across the corpus; this costs
# under 400 MB. Grouped by purpose so additions are deliberate.
PBP_KEEP = [
    # identity / ordering
    "year", "week", "season_type", "game_id", "id_play",
    "game_play_number", "half_play_number", "half", "period",
    # clock
    "clock_minutes", "clock_seconds", "TimeSecsRem", "adj_TimeSecsRem",
    # down / distance / field — start AND end state, because EPA is
    # EP(end state) - EP(start state) and the end state must not be inferred.
    "down", "distance", "yards_to_goal", "yards_gained", "Goal_To_Go",
    "down_end", "distance_end", "yards_to_goal_end", "TimeSecsRem_end",
    "Goal_To_Go_end", "Under_two_end",
    # possession & score
    "pos_team", "def_pos_team", "pos_team_score", "def_pos_team_score",
    "pos_score_diff_start", "offense_score", "defense_score",
    # timeouts (repaired downstream)
    "pos_team_timeouts_rem_before", "def_pos_team_timeouts_rem_before",
    "off_timeouts_rem_before", "def_timeouts_rem_before",
    # situational flags
    "Under_two", "pos_team_receives_2H_kickoff", "end_of_half",
    "change_of_pos_team", "play_type", "scoring", "scoring_play",
    # teams / venue / context
    "home", "away", "home_team_id", "away_team_id",
    "home_team_division", "away_team_division",
    "home_team_conference", "away_team_conference",
    "venue_id", "neutral_site", "conference_game", "start_date",
    # market + prior (RENAMED on ingest — see ingest_pbp.py)
    "spread", "over_under", "home_team_pregame_elo", "away_team_pregame_elo",
    # upstream model outputs: VALIDATION TARGETS ONLY, never features
    "EPA", "ep_before", "ep_after", "wp_before", "wp_after", "ppa",
    "No_Score_before", "FG_before", "Opp_FG_before", "Opp_Safety_before",
    "Opp_TD_before", "Safety_before", "TD_before",
    # drive structure (needed for next-score labelling)
    "drive_id", "drive_number", "drive_result", "drive_pts",
    # play outcome flags
    "rush", "pass", "sack", "int", "fumble_vec", "turnover",
    "touchdown", "safety", "fg_made", "fg_inds", "punt", "kickoff_play",
    "penalty_flag", "passer_player_name", "rusher_player_name",
]

# Columns that must never reach a feature matrix. Enforced by tests.
FORBIDDEN_FEATURES = {
    # outcome-derived, sit right beside legitimate columns
    "home_postgame_elo", "away_postgame_elo",
    "home_post_win_prob", "away_post_win_prob",
    "excitement_index", "highlights", "notes", "attendance",
    # in-sample fitted values from an upstream model trained on these seasons
    "EPA", "ep_before", "ep_after", "wp_before", "wp_after", "ppa",
    "No_Score_before", "FG_before", "Opp_FG_before", "Opp_Safety_before",
    "Opp_TD_before", "Safety_before", "TD_before",
    # the label itself and anything algebraically equivalent
    "home_points", "away_points", "margin", "total",
    "pos_team_score", "def_pos_team_score", "offense_score", "defense_score",
    # the market: benchmark, and a feature only in the explicit market variant
    "spread_close", "over_under_close",
}

# ------------------------------------------------------- modelling constants
MARGIN_LO, MARGIN_HI = -56, 56          # covers 99.26% of games
N_MARGIN_BINS = (MARGIN_HI - MARGIN_LO + 1) + 2   # 115, incl. two tail buckets
TOTAL_LO, TOTAL_HI = 10, 110
N_TOTAL_BINS = (TOTAL_HI - TOTAL_LO + 1) + 2      # 103
# Remaining-points grid for the delta-parameterised state tower: 0..110 inner
# bins plus ONE upper tail. No lower tail on purpose — remaining points cannot
# be negative, so under this head "total below the current score" is not a
# pattern to learn away but a bin that does not exist.
REM_HI = 110
N_REM_BINS = REM_HI + 2                           # 112: 0..110 + upper tail

# Fitted floor-plus-diffusion scale for remaining margin movement.
# sd(t) = sqrt(SD_FLOOR^2 + SD_COEF * t^SD_POW); weighted RMSE 0.248 vs 1.026
# for the sqrt(t) form the brief proposed. The floor matters: at 30 s left the
# true SD is 3.86 and sqrt(t) predicts 1.76.
SD_FLOOR, SD_COEF, SD_POW = 4.0, 0.00197, 1.5

# CFBD's garbage-time rule, read from their API server source. Keep a play if:
GARBAGE_TIME_SQL = """(
     period = 1
  OR (period = 2 AND NOT scoring_play AND abs(pos_score_diff_start) <= 38)
  OR (period = 3 AND NOT scoring_play AND abs(pos_score_diff_start) <= 28)
  OR (period = 4 AND NOT scoring_play AND abs(pos_score_diff_start) <= 22)
  OR (period = 2 AND scoring_play AND abs(pos_score_diff_start) <= 45)
  OR (period = 3 AND scoring_play AND abs(pos_score_diff_start) <= 35)
  OR (period = 4 AND scoring_play AND abs(pos_score_diff_start) <= 28)
)"""

# Splits. Season-level, never play-level: every play in a game shares one label.
# User decision (2026-08-27): train on every completed season through 2025;
# 2026 is the live season — its games are never trained on, and as they
# complete they grade the model as a true held-out set. Validation is
# GAME-level (the brief allows season- or game-level splits): a deterministic
# ~14% slice of games by id hash is held out of gradients for early stopping
# and checkpoint selection — with no completed test season at train time, it
# is also what run metrics are reported on.
TRAIN_SEASONS = list(range(2014, 2026))   # 2014-2025
VAL_SEASONS: list = []                    # season-level val retired
VAL_MOD, VAL_REM = 7, 3                   # game_id % VAL_MOD == VAL_REM -> val
TEST_SEASONS = [2026]

# The season currently being played: its schedule (including unplayed games)
# comes from CFBD via update_season.py into data/parquet/schedule.parquet,
# separate from games.parquet, which stays completed-games-only.
CURRENT_SEASON = 2026

# Measured benchmarks. A model that beats the first line is presumed leaky.
BENCH = {
    "closing_spread_rmse": 15.693,
    "closing_spread_mae": 12.364,
    "closing_spread_su": 0.7404,
    "closing_total_rmse": 16.184,
    "elo_baseline_rmse": 16.459,
    "elo_baseline_ll": 0.53053,
    "leak_tripwire_rmse": 15.5,
    "cfbfastr_wp_ll": {"all": 0.43844, "q1": 0.61661, "q2": 0.49140,
                       "q3": 0.36767, "q4": 0.25762},
}
