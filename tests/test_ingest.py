"""Stage 1 gates. These must pass before anything downstream is trusted.

A bug here is silent: cfbfastR's own changelog notes that a truncated game
"models perfectly cleanly and produces EPA, drive results and a box score that
all look reasonable and are all wrong."
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfbpred.config import FORBIDDEN_FEATURES, PBP, PBP_SEASONS, PQ  # noqa: E402

PBP_GLOB = str(PBP / "pbp_*.parquet")
GAMES = str(PQ / "games.parquet")
PINNED_GAMES = str(Path(__file__).resolve().parents[1] /
                   "models/player_sequence_production/refreshes/"
                   "20260920T122506160000Z/source/snapshot_schedule.parquet")
REVIEWED_SCHEDULE = str(Path(__file__).resolve().parents[1] /
                        "models/experiments/complete_history_20260922_v2/"
                        "source_schedule.parquet")

# Row counts verified against the upstream release files.
EXPECTED_ROWS = {
    2014: 158_188, 2015: 160_236, 2016: 158_600, 2017: 158_663,
    2018: 160_514, 2019: 159_920, 2020: 99_955, 2021: 163_444,
    2022: 252_307, 2023: 254_090, 2024: 277_048, 2025: 293_200,
}


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    c.execute(f"CREATE VIEW pbp AS SELECT * FROM read_parquet('{PBP_GLOB}')")
    c.execute(f"CREATE VIEW games AS SELECT * FROM read_parquet('{GAMES}')")
    # Keep the original pinned schedule as a regression fixture.  The
    # canonical games table now contains the reviewed complete-history overlay;
    # changing its row counts must not silently rewrite the ingest baseline.
    c.execute(f"CREATE VIEW pinned_games AS SELECT * FROM read_parquet('{PINNED_GAMES}')")
    c.execute("""CREATE VIEW fbs AS SELECT * FROM games
                 WHERE home_division='fbs' AND away_division='fbs'""")
    c.execute("""CREATE VIEW pinned_fbs AS SELECT * FROM pinned_games
                 WHERE home_division='fbs' AND away_division='fbs'""")
    yield c
    c.close()


# ------------------------------------------------------------------ coverage

def test_all_seasons_present():
    for y in PBP_SEASONS:
        assert (PBP / f"pbp_{y}.parquet").exists(), f"missing pbp_{y}"


def test_row_counts_match_source(con):
    got = dict(con.execute(
        'SELECT "year", count(*) FROM pbp GROUP BY 1').fetchall())
    for y, want in EXPECTED_ROWS.items():
        assert got[y] == want, f"{y}: {got[y]:,} rows, expected {want:,}"


def test_fbs_game_and_play_counts(con):
    scheduled = con.execute(
        "SELECT count(*) FROM pinned_fbs WHERE season>=2014 AND season<=2025").fetchone()[0]
    assert scheduled == 8_739, f"expected 8,739 scheduled, got {scheduled:,}"
    # 62 scheduled games have no play-by-play at all; that gap is the thing to
    # pin, because it silently shrinks the training set if it ever grows.
    with_pbp = con.execute("""
        SELECT count(DISTINCT g.game_id) FROM pinned_fbs g
        JOIN pbp p USING (game_id) WHERE g.season >= 2014 AND g.season<=2025""").fetchone()[0]
    assert with_pbp == 8_677, f"expected 8,677 games with pbp, got {with_pbp:,}"
    n_plays = con.execute("""
        SELECT count(*) FROM pbp p JOIN pinned_fbs g USING (game_id)
        WHERE p.period <= 4 AND p.adj_TimeSecsRem IS NOT NULL
          AND p.pos_score_diff_start IS NOT NULL""").fetchone()[0]
    assert n_plays == 1_574_157, f"expected 1,574,157 regulation plays, got {n_plays:,}"


def test_game_level_span(con):
    n = con.execute("SELECT count(*) FROM pinned_fbs WHERE season<=2025").fetchone()[0]
    assert n == 17_472, f"expected 17,472 FBS-vs-FBS games 2001-25, got {n:,}"
    lo, hi = con.execute("SELECT min(season), max(season) FROM pinned_games WHERE season<=2025").fetchone()
    assert (lo, hi) == (2001, 2025)


# ------------------------------------------------------------------- labels

def test_no_ties_in_authoritative_scores(con):
    """CFB has had overtime since 1996. A tie means the label is wrong."""
    n = con.execute("SELECT count(*) FROM fbs WHERE margin = 0").fetchone()[0]
    assert n == 0, f"{n} tie games — labels are not authoritative"


def test_every_game_has_a_score(con):
    n = con.execute("""SELECT count(*) FROM games
                       WHERE home_points IS NULL OR away_points IS NULL""").fetchone()[0]
    assert n == 0


def test_margin_and_total_share_parity(con):
    """away = (total - margin)/2 must be an integer. Holds on all 17,472."""
    n = con.execute("""SELECT count(*) FROM fbs
                       WHERE (abs(margin) % 2) <> (total % 2)""").fetchone()[0]
    assert n == 0, f"{n} games violate parity"


# ----------------------------------------------------------------- timeouts

def test_timeouts_in_range(con):
    bad = con.execute("""
        SELECT count(*) FROM pbp
        WHERE pos_team_timeouts_rem_before NOT BETWEEN 0 AND 3
           OR def_pos_team_timeouts_rem_before NOT BETWEEN 0 AND 3
           OR pos_team_timeouts_rem_before IS NULL
           OR def_pos_team_timeouts_rem_before IS NULL""").fetchone()[0]
    assert bad == 0, f"{bad:,} plays with out-of-range or null timeouts"


def test_timeouts_monotone_within_half(con):
    """Timeouts can only decrease within a half."""
    bad = con.execute("""
        WITH s AS (
          SELECT pos_team_timeouts_rem_before t,
                 lag(pos_team_timeouts_rem_before) OVER (
                   PARTITION BY game_id, half, pos_team ORDER BY play_seq) p
          FROM pbp WHERE period <= 4)
        SELECT count(*) FROM s WHERE p IS NOT NULL AND t > p""").fetchone()[0]
    assert bad == 0, f"{bad:,} increasing-timeout transitions"


def test_timeouts_start_each_half_at_three(con):
    """Both halves should open with three. Pre-repair this was ~99.5%."""
    pct = con.execute("""
        SELECT avg(CASE WHEN pos_team_timeouts_rem_before = 3 THEN 1.0 ELSE 0 END)
        FROM pbp WHERE half_play_number <= 2 AND period <= 4""").fetchone()[0]
    assert pct > 0.985, f"only {pct:.3%} of half-opening plays have 3 timeouts"


# ------------------------------------------------------------------ leakage

def test_forbidden_columns_absent_from_games(con):
    cols = {c[0] for c in con.execute("DESCRIBE games").fetchall()}
    leaked = cols & {"home_postgame_elo", "away_postgame_elo", "home_post_win_prob",
                     "away_post_win_prob", "excitement_index", "attendance",
                     "highlights", "notes"}
    assert not leaked, f"outcome-derived columns present: {sorted(leaked)}"


def test_market_columns_are_renamed(con):
    """`spread` must never survive unlabelled — it is the CLOSING line."""
    cols = {c[0] for c in con.execute("DESCRIBE pbp").fetchall()}
    assert "spread" not in cols and "over_under" not in cols
    assert "spread_close" in cols and "over_under_close" in cols


def test_pregame_elo_is_constant_within_game(con):
    """If it varied within a game it would not be a pregame quantity."""
    n = con.execute("""
        SELECT count(*) FROM (
          SELECT game_id FROM pbp GROUP BY game_id
          HAVING count(DISTINCT home_team_pregame_elo) > 1)""").fetchone()[0]
    assert n == 0


# ----------------------------------------------------- known-defect accounting

def test_pbp_score_defects_are_bounded_and_classified(con):
    """PBP running scores disagree with the authoritative label on ~2.8% of
    games. That is a known upstream defect; this test pins the rate so a
    regression is visible, and asserts we never take labels from here."""
    r = con.execute("""
        WITH d AS (
          SELECT p.game_id,
                 any_value(g.home_points) hp, any_value(g.away_points) ap,
                 max(CASE WHEN p.pos_team = p.home THEN p.pos_team_score
                          ELSE p.def_pos_team_score END) hp_pbp,
                 max(CASE WHEN p.pos_team = p.away THEN p.pos_team_score
                          ELSE p.def_pos_team_score END) ap_pbp
          FROM pbp p JOIN pinned_fbs g USING (game_id) GROUP BY p.game_id)
        SELECT count(*),
               sum(CASE WHEN hp_pbp = hp AND ap_pbp = ap THEN 1 ELSE 0 END),
               sum(CASE WHEN hp_pbp = ap AND ap_pbp = hp AND hp <> ap THEN 1 ELSE 0 END)
        FROM d""").fetchone()
    total, exact, swapped = r
    assert total == 8_677
    assert exact / total > 0.965, f"only {exact / total:.1%} of games score-match"
    assert swapped <= 12, f"{swapped} home/away-swapped games (was 8)"


def test_reviewed_history_overlay_is_present(con):
    """The reviewed schedule additions are checked by provenance, separately
    from the pinned ingest regression counts above."""
    added, installed = con.execute(f"""
        WITH reviewed AS (
          SELECT game_id FROM read_parquet('{REVIEWED_SCHEDULE}')
          EXCEPT SELECT game_id FROM pinned_games
        )
        SELECT count(*), count(g.game_id)
        FROM reviewed r LEFT JOIN games g USING (game_id)
    """).fetchone()
    assert added == 7_981
    assert installed == added
