type ScoreState = { game_id: number; state: string; home_points: number; away_points: number };
export type SavedFinal = {
  game_id: number; state: string; curve_done: boolean | null;
  home_points: number | null; away_points: number | null;
  updated_at?: string;
};

/** Completed, unchanged curves need neither logits nor inference. Corrections
 * and incomplete histories remain eligible; explicit replays bypass this. */
export function scoringWork<T extends ScoreState>(states: T[], saved: SavedFinal[], supported?: Set<number>): T[] {
  const byId = new Map(saved.map(row => [Number(row.game_id), row]));
  const eligible = states.filter(state => {
    if (supported && !supported.has(state.game_id)) return false;
    const row = byId.get(state.game_id);
    return state.state !== "post" || row?.state !== "post" || row.curve_done !== true
      || row.home_points !== state.home_points || row.away_points !== state.away_points;
  });
  // Drain completed-game backlogs over several ticks. All live games remain
  // eligible; a single final-heavy invocation must not exceed Edge CPU limits.
  const live = eligible.filter(state => state.state !== "post");
  // Oldest attempt first: games with unavailable feeds cannot hold the first
  // sixteen slots forever and starve the rest of the completed slate.
  const finals = eligible.filter(state => state.state === "post").sort((a, b) =>
    (byId.get(a.game_id)?.updated_at ?? "").localeCompare(byId.get(b.game_id)?.updated_at ?? ""));
  return [...finals.slice(0, 16), ...live];
}
