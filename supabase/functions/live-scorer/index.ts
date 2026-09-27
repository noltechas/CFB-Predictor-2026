/**
 * The live scorer, as a scheduled edge function.
 *
 * Mirrors live_scorer.py's cycle: poll both ESPN scoreboards, rebuild each
 * in-progress game's play sequence, price every snap with the frozen
 * ensemble, and write live_curve + live_predictions. The pregame half of the
 * network never runs here — its logits are published per game by
 * publish_pregame_logits.py and read back below.
 *
 * `?dry=1` scores everything and writes nothing, for comparing against the
 * Mac scorer before cutover.
 */
import { createClient } from "npm:@supabase/supabase-js@2";
import { fetchPlayRows, liveStates, LiveState, scoreboardContext, stateFromSummary } from "../_shared/espn.ts";
import { deriveState, PlayRow } from "../_shared/features.ts";
import { Meta, Net, readouts } from "../_shared/net.ts";
import { scoringWork, SavedFinal } from "../_shared/scorer_work.ts";
import { prepareCurve, finalHistoryComplete, resumeCurve, staleRows, StoredCurve } from "../_shared/curve_history.ts";

const URL_ = Deno.env.get("SUPABASE_URL")!;
const KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL_, KEY, { auth: { persistSession: false } });

const CONCURRENCY = 6;

let net: Net | null = null;
let loadingNet: Promise<Net> | null = null;
async function getNet(): Promise<Net> {
  if (net) return net;
  if (!loadingNet) loadingNet = (async () => {
    // Frozen weights ship with the function. Cold starts must not download
    // 4.5 MB from Storage; update the assets and redeploy to change the model.
    const [w, m] = await Promise.all([
      Deno.readFile(new URL("../../model/weights.bin", import.meta.url)),
      Deno.readTextFile(new URL("../../model/weights.json", import.meta.url)),
    ]);
    net = new Net(w.buffer.slice(w.byteOffset, w.byteOffset + w.byteLength) as ArrayBuffer,
      JSON.parse(m) as Meta);
    return net;
  })().finally(() => { loadingNet = null; });
  return loadingNet;
}

/** Run tasks with a cap, so twenty live games do not open twenty sockets. */
async function pooled<T, R>(items: T[], n: number, fn: (t: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let i = 0;
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, async () => {
    while (i < items.length) {
      const k = i++;
      out[k] = await fn(items[k]);
    }
  }));
  return out;
}


Deno.serve(async (req) => {
  const t0 = performance.now();
  const dry = new URL(req.url).searchParams.get("dry") === "1";
  try {
    // Authenticated deployment check: validates bundled assets without
    // reading/writing application data or fetching any Storage objects.
    if (new URL(req.url).searchParams.get("check") === "model") {
      await getNet();
      return Response.json({ ok: true, model: "bundled", storageRequests: 0 });
    }
    // ?game=<id> replays one game off its summary feed, whatever the
    // scoreboard says — the only way to exercise this outside a game window
    const only = new URL(req.url).searchParams.get("game");
    const states = only
      ? [await stateFromSummary(Number(only))].filter(Boolean) as LiveState[]
      : await liveStates();
    if (!states.length) {
      return Response.json({ ok: true, dry, live: 0, note: "nothing in progress" });
    }

    // Check ownership before any large reads or model loading.
    if (!dry && !only) {
      const cutoff = new Date(Date.now() - 3 * 60_000).toISOString();
      const { data: macRows, error } = await db.from("live_predictions")
        .select("game_id").is("source", null).gt("updated_at", cutoff).limit(1);
      if (error) throw new Error(`ownership: ${error.message}`);
      if (macRows?.length) return Response.json({ ok: true,
        deferred: "live_scorer.py is running", ms: Math.round(performance.now() - t0) });
    }
    const finalIds = states.filter(s => s.state === "post").map(s => s.game_id);
    let prevState: SavedFinal[] = [];
    if (!only && finalIds.length) {
      const { data, error } = await db.from("live_predictions")
        .select("game_id,state,curve_done,home_points,away_points,updated_at").in("game_id", finalIds);
      if (error) throw new Error(`final state: ${error.message}`);
      prevState = (data ?? []) as SavedFinal[];
    }
    const { data: supportedRows, error: supportedError } = await db.from("game_pregame_logits")
      .select("game_id").in("game_id", states.map(s => s.game_id));
    if (supportedError) throw new Error(`supported games: ${supportedError.message}`);
    const work = only ? states : scoringWork(states, prevState,
      new Set((supportedRows ?? []).map(r => Number(r.game_id))));
    if (!work.length) return Response.json({ ok: true, dry, live: 0,
      note: "completed curves already current", modelLoaded: false });

    // Only fetch arrays for live games, unfinished curves, or corrected finals.
    const ids = work.map((s) => s.game_id);
    const { data: logitRows, error: lerr } = await db
      .from("game_pregame_logits")
      .select("game_id, n_members, margin_logits, total_logits")
      .in("game_id", ids);
    if (lerr) throw new Error(`logits: ${lerr.message}`);
    const logits = new Map<number, { pm: Float32Array[]; pt: Float32Array[] }>();
    for (const row of logitRows ?? []) {
      const M = row.n_members;
      const mw = row.margin_logits.length / M, tw = row.total_logits.length / M;
      logits.set(Number(row.game_id), {
        pm: Array.from({ length: M }, (_, k) =>
          Float32Array.from(row.margin_logits.slice(k * mw, (k + 1) * mw))),
        pt: Array.from({ length: M }, (_, k) =>
          Float32Array.from(row.total_logits.slice(k * tw, (k + 1) * tw))),
      });
    }

    // ?stage=N stops after a given phase, to locate a resource ceiling
    const stage = Number(new URL(req.url).searchParams.get("stage") ?? 0);
    const mem = () => {
      try { return Math.round((Deno as any).memoryUsage().rss / 1e6); } catch { return -1; }
    };
    if (stage === 1) {
      return Response.json({ ok: true, stage, seen: states.length, rssMB: mem(),
                             ms: Math.round(performance.now() - t0) });
    }

    const known = work.filter((s) => logits.has(s.game_id));
    const live = known.filter((s) => s.state === "in");
    const done = known.filter((s) => s.state === "post");
    // Explicit historical replay uses the same final-state and completeness
    // rules as scheduled work. Score once, then optionally publish the rows.
    if (only) {
      const g = known[0];
      if (!g) return Response.json({ ok: false, error: "no published pregame call for that game" });
      const prows = prepareCurve(await fetchPlayRows(g), g);
      const write = new URL(req.url).searchParams.get("write") === "1" && !dry;
      if (!prows.length) return Response.json({ ok: false, error: "no play feed" });
      const n = await getNet(), feats = deriveState(prows);
      const { pm, pt } = logits.get(g.game_id)!;
      const stamp = new Date().toISOString();
      const rows = prows.map((p, i) => {
        const o = n.run(feats[i], p.t_left / 3600, Math.abs(p.home_lead) / 21,
          p.home_lead, p.total_pts_before, pm, pt);
        const ro = readouts(o.margin, o.total);
        return { game_id: g.game_id, play_index: i, updated_at: stamp,
          t_left: p.t_left, period: p.period, clock: p.clock, down: p.down,
          distance: p.distance, situation: p.sit ?? "", play_text: p.play_text ?? "",
          home_points: p.hp, away_points: p.ap, home_win_prob: ro.wp,
          pred_margin: ro.em, pred_total: ro.et,
          total_pmf: Array.from(o.total, v => Math.round(v * 1e6) / 1e6) };
      });
      if (write) {
        for (let i = 0; i < rows.length; i += 50) {
          const { error } = await db.from("live_curve").upsert(rows.slice(i, i + 50),
            { onConflict: "game_id,play_index" });
          if (error) throw new Error(`repair ${g.game_id}: ${error.message}`);
        }
        const { error } = await db.from("live_curve").delete().eq("game_id", g.game_id)
          .gte("play_index", rows.length);
        if (error) throw new Error(`repair tail: ${error.message}`);
        const { error: finalError } = await db.from("live_predictions")
          .update({ curve_done: finalHistoryComplete(g, prows) }).eq("game_id", g.game_id);
        if (finalError) throw new Error(`repair completion: ${finalError.message}`);
        return Response.json({ ok: true, repaired: g.game_id, snaps: rows.length,
          ms: Math.round(performance.now() - t0) });
      }
      return Response.json({ ok: true, replay: g.game_id, n: rows.length,
        ms: Math.round(performance.now() - t0), rows });
    }

    // finals get their scores carried immediately, ahead of the weekly ingest
    if (!dry && done.length) {
      const { error } = await db.from("live_predictions").upsert(done.map((r) => ({
        game_id: r.game_id, updated_at: new Date().toISOString(), state: "post",
        period: r.period, clock: "FINAL",
        home_points: r.home_points, away_points: r.away_points, situation: "",
        source: "edge", live_context: null, curve_done: false,
      })), { onConflict: "game_id" });
      if (error) throw new Error(`final scores: ${error.message}`);
    }

    // scoringWork already excluded unchanged, verified final histories.
    const todo = [...live, ...done];

    // Fetch in parallel (network-bound), then score and WRITE one game at a
    // time. Holding every game's per-snap PMFs at once is what tripped
    // WORKER_RESOURCE_LIMIT; this caps peak memory at a single game.
    const fetched = (await pooled(todo, CONCURRENCY, async (r) => {
      const prows = prepareCurve(await fetchPlayRows(r), r);
      if (!prows.length) return null;              // no drives feed below FCS
      return { r, prows };
    })).filter(Boolean) as { r: LiveState; prows: PlayRow[] }[];

    if (stage === 2) {
      return Response.json({ ok: true, stage, fetched: fetched.length,
                             rows: fetched.reduce((a, f) => a + f.prows.length, 0),
                             rssMB: mem(), ms: Math.round(performance.now() - t0) });
    }

    if (!fetched.length) return Response.json({ ok: true, dry, live: live.length,
      finals: done.length, note: "no curve work available", modelLoaded: false });

    // A busy Saturday is ~1500 snaps x 16 members, which trips the worker's
    // CPU ceiling in one invocation. So: every live game gets its HEADLINE
    // re-priced every tick (one snap each, so the app is never stale), and a
    // fixed budget is then spent backfilling curve rows. One game per tick is
    // rebuilt in full, oldest curve first, which is what keeps ESPN's clock
    // backfill self-correcting.
    const MAX_SNAPS = 120;
    let budget = MAX_SNAPS;

    const gameIds = fetched.map((f) => f.r.game_id);
    // via an aggregate, NOT by selecting the rows: PostgREST caps a response
    // at 1000 rows, so reading 25 games x ~250 snaps came back truncated and
    // max(play_index) was too low — which resumed writing over existing rows
    const { data: curveState, error: cserr } = await db
      .rpc("live_curve_state", { ids: gameIds });
    if (cserr) throw new Error(`curve state: ${cserr.message}`);
    const have = new Map<number, { max: number; n: number; oldest: string }>();
    for (const row of curveState ?? []) {
      have.set(Number(row.game_id), {
        max: Number(row.max_idx), n: Number(row.n), oldest: row.oldest as string,
      });
    }
    // Keep the existing prefix; repeatedly rebuilding it exhausts the budget
    // before later quarters can ever be written.
    const refreshTarget = -1;
    // ...but re-price any stored point the fresh parse has since re-timed.
    // Without this, a point placed by a bad stamp stayed there for good. A
    // failed read only skips corrections this tick; it never blocks scoring.
    const stored = new Map<number, StoredCurve>();
    const { data: timeRows, error: terr } = await db.rpc("live_curve_times", { ids: gameIds });
    for (const row of terr ? [] : timeRows ?? []) stored.set(Number(row.game_id), row as StoredCurve);

    const stamp = new Date().toISOString();
    let snaps = 0;
    const summary: Record<string, unknown>[] = [];
    const prepared = fetched.map(({ r, prows }) => ({ r, prows, feats: deriveState(prows) }));
    if (stage === 3) {
      return Response.json({ ok: true, stage, games: prepared.length,
                             rows: prepared.reduce((a, p) => a + p.prows.length, 0),
                             ms: Math.round(performance.now() - t0) });
    }

    const n = await getNet();
    const priceOne = (g: typeof prepared[number], i: number) => {
      const p = g.prows[i];
      const { pm, pt } = logits.get(g.r.game_id)!;
      const o = n.run(g.feats[i], p.t_left / 3600, Math.abs(p.home_lead) / 21,
                      p.home_lead, p.total_pts_before, pm, pt);
      snaps++;
      return { p, ro: readouts(o.margin, o.total), total: o.total };
    };

    // --- pass 1: the headline for every game in progress
    for (const g of prepared) {
      if (g.r.state !== "in") continue;
      const { p, ro } = priceOne(g, g.prows.length - 1);
      budget--;
      if (!dry) {
        const { error } = await db.from("live_predictions").upsert({
          game_id: g.r.game_id, updated_at: stamp, state: "in",
          period: g.r.period, clock: g.r.clock,
          home_points: g.r.home_points, away_points: g.r.away_points,
          situation: p.sit ?? "",
          live_context: scoreboardContext(g.r, stamp),
          home_win_prob: ro.wp, pred_margin: ro.em, pred_total: ro.et,
          source: "edge", curve_done: false,
        }, { onConflict: "game_id" });
        if (error) throw new Error(`live headline: ${error.message}`);
      }
      summary.push({
        game_id: g.r.game_id,
        score: `${g.r.away_abbr} ${g.r.away_points} @ ${g.r.home_abbr} ${g.r.home_points}`,
        clock: `Q${g.r.period} ${g.r.clock}`,
        wp: +ro.wp.toFixed(4), margin: +ro.em.toFixed(2), total: +ro.et.toFixed(2),
      });
    }

    // --- pass 2: curve backfill until the budget is gone. Finished games go
    // first — a live game will be offered again next tick, a final one only
    // stays in the list because its curve is short.
    let wrote = 0, retimed = 0;
    // a shrunken feed means ESPN restated the game; rebuild from scratch
    // Reprice the last stored point, which may be the temporary live
    // state or a corrected final. Never restart a long game every tick.
    const plan = new Map(prepared.map((g) => {
      const { full, from } = resumeCurve(have.get(g.r.game_id), g.prows.length, false, true);
      const redo = full ? [] : staleRows(stored.get(g.r.game_id), g.prows, from);
      const idx: number[] = [];
      for (let i = from; i < g.prows.length; i++) idx.push(i);   // new points first
      idx.push(...redo);
      return [g.r.game_id, { full, idx, redo: redo.length }];
    }));
    const order = [...prepared].sort((a, b) => {
      const fa = a.r.state === "in" ? 1 : 0, fb = b.r.state === "in" ? 1 : 0;
      if (fa !== fb) return fa - fb;
      return plan.get(a.r.game_id)!.idx.length - plan.get(b.r.game_id)!.idx.length;
    });
    for (const g of order) {
      if (budget <= 0) break;
      const { full, idx, redo } = plan.get(g.r.game_id)!;
      const markDone = async () => {
        if (dry || !finalHistoryComplete(g.r, g.prows)) return;
        const { error } = await db.from("live_predictions")
          .update({ curve_done: true }).eq("game_id", g.r.game_id);
        if (error) throw new Error(`curve completion: ${error.message}`);
      };
      if (!idx.length) { await markDone(); continue; }
      if (full && !dry) {
        const { error } = await db.from("live_curve").delete().eq("game_id", g.r.game_id);
        if (error) throw new Error(`curve reset: ${error.message}`);
      }

      const batch: Record<string, unknown>[] = [];
      let wroteThis = 0;
      const flush = async () => {
        if (dry || !batch.length) { batch.length = 0; return; }
        const { error } = await db.from("live_curve")
          .upsert(batch, { onConflict: "game_id,play_index" });
        if (error) throw new Error(`curve ${g.r.game_id}: ${error.message}`);
        batch.length = 0;
      };
      for (let k = 0; k < idx.length && budget > 0; k++) {
        const i = idx[k];
        if (k >= idx.length - redo) retimed++;
        const { p, ro, total } = priceOne(g, i);
        budget--;
        const pmf = new Array(total.length);
        for (let j = 0; j < total.length; j++) pmf[j] = Math.round(total[j] * 1e6) / 1e6;
        batch.push({
          game_id: g.r.game_id, play_index: p.play_index, updated_at: stamp,
          t_left: p.t_left, period: p.period, clock: p.clock,
          down: p.down, distance: p.distance, situation: p.sit ?? "",
          play_text: p.play_text ?? "", home_points: p.hp, away_points: p.ap,
          home_win_prob: ro.wp, pred_margin: ro.em, pred_total: ro.et, total_pmf: pmf,
        });
        if (batch.length >= 50) await flush();
        wrote++; wroteThis++;
      }
      await flush();
      // only complete if the budget carried us through every pending point
      if (wroteThis >= idx.length) await markDone();
    }

    if (!dry) {
      const iso = (d: number) => new Date(Date.now() - d * 864e5).toISOString();
      await db.from("live_predictions").delete().lt("updated_at", iso(8));
      await db.from("live_curve").delete().lt("updated_at", iso(120));
      await db.from("live_curve").update({ total_pmf: null })
        .lt("updated_at", iso(14)).not("total_pmf", "is", null);
    }

    return Response.json({
      ok: true, dry,
      seen: states.length, known: known.length, live: live.length,
      finals: done.length, headlines: summary.length, curveRows: wrote, retimed, snaps,
      retimeRead: terr ? terr.message : "ok",
      budgetLeft: budget,
      ms: Math.round(performance.now() - t0), rssMB: mem(),
      refreshed: refreshTarget, games: summary,
    });
  } catch (e) {
    return Response.json({ ok: false, error: String(e), ms: Math.round(performance.now() - t0) },
                         { status: 500 });
  }
});
