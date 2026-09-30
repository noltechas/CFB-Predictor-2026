/* Fourth Down Engine — app logic.
   Reads Supabase REST (anon): games, teams, game_predictions, game_lines,
   play_predictions, play_raw, model_runs. All joins happen client-side.

   Conventions: CFBD spread is the HOME handicap (negative = home favored),
   so the market's expected home margin is -spread. The model's is
   pred_margin. Both are judged against the actual margin. */

const CFG = await fetch("./config.json").then(r => r.json());
const HEADERS = { apikey: CFG.key, Authorization: `Bearer ${CFG.key}` };

async function api(path) {
  const r = await fetch(`${CFG.url}/rest/v1/${path}`, { headers: HEADERS });
  if (!r.ok) throw new Error(`API ${r.status} on ${path.split("?")[0]}`);
  return r.json();
}

const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = v => `${(v * 100).toFixed(1)}%`;
const signed = (v, d = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;
const mono = s => `<span class="num">${s}</span>`;
const loadingHtml = label => `<div class="loading" role="status" aria-live="polite">
  <span class="spinner" aria-hidden="true"></span>
  <span class="loading-label">${esc(label)}</span></div>`;
const railSkeleton = (n = 7) => Array.from({ length: n }, () => `<div class="sk-row">
  <div class="sk" style="height:14px;width:72%"></div>
  <div class="sk" style="height:14px;width:58%"></div></div>`).join("");

/* ------------------------------------------------------------- theme */
const THEME_KEY = "fde-theme";
$("theme-toggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem(THEME_KEY, next); } catch (e) {}
  if (state.sel) renderDetail(state.sel);   // swap logo variants
  renderList();
});
const isDark = () => document.documentElement.dataset.theme === "dark";

/* ------------------------------------------------------------- odds math */
const mlProb = ml => ml < 0 ? -ml / (-ml + 100) : 100 / (ml + 100);
function vegasWP(line) {
  if (!line || line.home_moneyline == null || line.away_moneyline == null) return null;
  const h = mlProb(line.home_moneyline), a = mlProb(line.away_moneyline);
  return h / (h + a);                       // de-vigged
}
const marketMargin = line =>
  line && line.spread != null ? -line.spread : null;

function fmtSpread(spread, homeAbbr, awayAbbr) {
  if (spread === 0) return "PK";
  const fav = spread < 0 ? homeAbbr : awayAbbr;
  return `${fav} −${Math.abs(spread).toFixed(1).replace(/\.0$/, "")}`;
}

/* expected HOME margin -> betting notation: the favorite carries the minus */
function fmtFav(homeMargin, homeAbbr, awayAbbr, dec = 1) {
  if (Math.abs(homeMargin) < 0.05) return "PK";
  const fav = homeMargin > 0 ? homeAbbr : awayAbbr;
  return `${fav} −${Math.abs(homeMargin).toFixed(dec)}`;
}

/* ---- pricing off the model's own distribution -------------------------
   game_pmf holds the live season's full pregame PMFs (the same object every
   readout is a partial sum of): 115 margin bins (-56..56 + two ±62 tails)
   and 103 total bins (10..110 + two tails). Cover probability against ANY
   current line is an exact partial sum — no curve fitting, no averages. */
const MARGIN_CENTRES = [-62,
  ...Array.from({ length: 113 }, (_, i) => i - 56), 62];
const TOTAL_CENTRES = [4,
  ...Array.from({ length: 101 }, (_, i) => i + 10), 116];

/* P(value > line) and P(value == line) from a discrete PMF */
function tailProb(centres, probs, line) {
  let over = 0, push = 0;
  for (let i = 0; i < centres.length; i++) {
    if (Math.abs(centres[i] - line) < 1e-9) push += probs[i];
    else if (centres[i] > line) over += probs[i];
  }
  return { over, push };
}

/* flat $100 at −110: win pays $90.91, push refunds */
const evAt110 = (pWin, pPush) =>
  pWin * (100 / 1.1) - (1 - pWin - pPush) * 100;
const BREAK_EVEN_110 = 110 / 210;                       // 52.4%

/* judge who was closer; returns {chip class, label} */
function closerVerdict(modelErr, vegasErr) {
  if (Math.abs(modelErr - vegasErr) < 1e-9)
    return { cls: "push", label: "Dead heat" };
  return modelErr < vegasErr
    ? { cls: "brand", label: "Model closer" }
    : { cls: "vegas", label: "Vegas closer" };
}

/* ------------------------------------------------------------- state */
const state = {
  teams: new Map(),      // team_id -> row
  byName: new Map(),     // school -> row
  games: [],             // joined rows for current selection
  sel: null,
  filter: "",
  sort: "kick",          // "kick" | "edge" (model-vs-line disagreement)
  division: "all",       // "all" | "fbs" (FBS-involved) | "fcs" (FCS-only)
  liveView: "live",      // live games' board: "live" read | "pre"game call
};

function teamOf(g, side) {
  return state.teams.get(g[`${side}_team_id`]) ||
         state.byName.get(g[`${side}_team`]) || null;
}
function teamColor(t) {
  const c = t?.color;
  if (!c || !/^#[0-9a-fA-F]{6}$/.test(c) || c.toLowerCase() === "#ffffff") return null;
  return c;
}
function logoUrl(t) {
  if (!t) return null;
  return (isDark() && t.logo_dark) ? t.logo_dark : t.logo;
}
function abbr(t, name) {
  return t?.abbreviation || String(name ?? "").slice(0, 4).toUpperCase();
}
function logoImg(t, name, cls, monoCls, px) {
  const url = logoUrl(t);
  const mono = monoHtml(t, name, monoCls, !url);
  if (!url) return mono;
  // the monogram sits hidden behind the img and takes over if the load fails
  return `<img class="${cls}" src="${esc(url)}" alt="" loading="lazy"
               width="${px}" height="${px}"
               onerror="this.style.display='none';this.nextElementSibling.style.display='grid'">${mono}`;
}
function monoHtml(t, name, cls, visible) {
  const bg = teamColor(t) || "#7A8072";
  return `<span class="${cls}" style="background:${bg};display:${visible ? "grid" : "none"}">${esc(abbr(t, name))}</span>`;
}

/* ------------------------------------------------------------- boot */
async function boot() {
  // the rail should never sit empty while the first fetches are in flight
  $("game-list").innerHTML = railSkeleton();
  try {
    const [runs, teams] = await Promise.all([
      api("model_runs?select=model_version,trained_at,margin_rmse,wp_logloss_q1,notes&order=trained_at.desc&limit=1"),
      api("teams?select=team_id,school,abbreviation,conference,color,alt_color,logo,logo_dark&limit=1000"),
    ]);
    for (const t of teams) { state.teams.set(t.team_id, t); state.byName.set(t.school, t); }
    const run = runs[0] ?? null;
    const version = run?.model_version ?? "Model version unavailable";
    const hasMetrics = Number.isFinite(run?.wp_logloss_q1) && Number.isFinite(run?.margin_rmse);
    const metrics = hasMetrics
      ? `Q1 LL ${run.wp_logloss_q1.toFixed(3)} · mRMSE ${run.margin_rmse.toFixed(1)}`
      : "evaluation metrics unavailable";
    $("run-text").textContent = `${version} · ${metrics}${run?.notes ? ` · ${run.notes}` : ""}`;
    $("run-dot").classList.toggle("ok", Boolean(run));

    const [[hi], [lo]] = await Promise.all([
      api("game_predictions?select=season&order=season.desc&limit=1"),
      api("game_predictions?select=season&order=season.asc&limit=1"),
    ]);
    const years = [];
    for (let y = hi.season; y >= lo.season; y--) years.push(y);
    $("season").innerHTML = years.map(y => `<option>${y}</option>`).join("");
    $("season").onchange = loadWeeks;
    $("week").onchange = loadGames;
    $("search").addEventListener("input", e => {
      state.filter = e.target.value.trim().toLowerCase();
      renderList();
    });
    document.querySelectorAll("#rail-sort .seg-btn").forEach(b =>
      b.addEventListener("click", () => {
        state.sort = b.dataset.sort;
        document.querySelectorAll("#rail-sort .seg-btn").forEach(x =>
          x.setAttribute("aria-selected", x.dataset.sort === state.sort));
        renderList();
      }));
    document.querySelectorAll("#rail-div .seg-btn").forEach(b =>
      b.addEventListener("click", () => {
        state.division = b.dataset.div;
        document.querySelectorAll("#rail-div .seg-btn").forEach(x =>
          x.setAttribute("aria-selected", x.dataset.div === state.division));
        renderList();
      }));
    await loadWeeks();
  } catch (e) {
    $("run-text").textContent = `offline — ${e.message}`;
    $("run-dot").classList.add("err");
  }
}

async function loadWeeks() {
  const s = $("season").value;
  // PostgREST caps at 1,000 rows and CFBD restarts postseason weeks at 1,
  // so the max regular week is queried directly and bowls get their own entry.
  const [[mx], post] = await Promise.all([
    api(`games?select=week&season=eq.${s}&season_type=eq.regular&order=week.desc&limit=1`),
    api(`games?select=game_id&season=eq.${s}&season_type=eq.postseason&limit=1`),
  ]);
  const opts = [];
  for (let w = 1; w <= (mx?.week ?? 15); w++) opts.push(`<option value="${w}">Week ${w}</option>`);
  if (post.length) opts.push(`<option value="post">Postseason</option>`);
  $("week").innerHTML = opts.join("");
  await loadGames();
}

async function loadGames() {
  const s = $("season").value, w = $("week").value;
  $("game-list").innerHTML = railSkeleton();
  $("rail-title").textContent = w === "post" ? "Postseason" : `Week ${w}`;
  $("rail-count").textContent = "";
  $("week-ledger").innerHTML = "";

  const stype = w === "post" ? `&season_type=eq.postseason`
                             : `&week=eq.${w}&season_type=eq.regular`;
  // every game the model prices is eligible; rows without a published
  // prediction (lower-division matchups outside the population) drop below
  const games = await api(
    `games?select=game_id,season,week,season_type,start_date,start_time_tbd,neutral_site,venue_id,` +
    `home_team_id,home_team,home_points,away_team_id,away_team,away_points,` +
    `home_classification,away_classification` +
    `&season=eq.${s}${stype}&limit=2000`);
  const ids = games.map(g => g.game_id);
  if (!ids.length) {
    $("game-list").innerHTML = `<div class="rail-empty">No games found.</div>`;
    return;
  }
  const idList = ids.join(",");
  const [preds, lines, scals, liveRows] = await Promise.all([
    api(`game_predictions?select=game_id,home_win_prob,pred_home_pts,pred_away_pts,pred_margin,pred_total,split&game_id=in.(${idList})&limit=1000`),
    api(`game_lines?select=game_id,provider,spread,spread_open,over_under,over_under_open,home_moneyline,away_moneyline&game_id=in.(${idList})&limit=1000`),
    // roster presence per side: null n_players means CFBD hadn't published
    // that team's roster when features were built, so the model ran blind
    api(`game_scalars?select=game_id,h_n_players,a_n_players&game_id=in.(${idList})&limit=1000`)
      .catch(() => []),
    api(`live_predictions?select=game_id,updated_at,state,period,clock,home_points,away_points,situation,home_win_prob,pred_margin,pred_total&game_id=in.(${idList})&limit=400`)
      .catch(() => []),
  ]);
  const pBy = new Map(preds.map(p => [p.game_id, p]));
  const lBy = new Map(lines.map(l => [l.game_id, l]));
  const sBy = new Map(scals.map(s => [s.game_id, s]));
  state.games = games
    .filter(g => pBy.has(g.game_id))
    .map(g => {
      const s = sBy.get(g.game_id);
      const rosterGap = s
        ? [s.h_n_players == null ? g.home_team : null,
           s.a_n_players == null ? g.away_team : null].filter(Boolean)
        : [];
      return { ...g, pred: pBy.get(g.game_id), line: lBy.get(g.game_id) ?? null, rosterGap };
    })
    .sort((a, b) => (a.start_date ?? "").localeCompare(b.start_date ?? "") ||
                    a.game_id - b.game_id);
  attachLive(state.games, liveRows);
  state.sel = null;
  renderLedger();
  renderList();
  if (state.games.length) select(state.games[0].game_id, { scroll: false });
  armLiveTimer();
}

/* A game whose kickoff has passed but that has no fresh live row and no
   final. Something upstream is down: the scorer, its database link, or the
   feed. Calling this "Scheduled" is a lie the dashboard told for 3h20m on
   2026-08-29 while a game ran to halftime, so it now says so out loud. */
function liveGap(g) {
  if (g.home_points != null || g.live || g.livePost) return false;
  if (!g.start_date || g.start_time_tbd) return false;
  // a few minutes' grace: kickoff timestamps drift and the first snap has
  // to reach the feed before there is anything to report
  return Date.now() - new Date(g.start_date).getTime() > 6 * 60e3;
}

/* ------------------------------------------------- live in-game overlay */
/* live_predictions is written by live_scorer.py while games are in
   progress: current score/clock/situation plus the model's read of that
   exact state. A row counts only while fresh — the scorer stops touching a
   game at the final whistle, so staleness is the end-of-game signal. */
function attachLive(games, liveRows) {
  // 'in' rows are live only while fresh; 'post' rows are facts (the final
  // score, ahead of the weekly official ingest) and stay attached
  const fresh = l => Date.now() - new Date(l.updated_at).getTime() < 30 * 60e3;
  const rows = liveRows || [];
  const lv = new Map(rows.filter(l => l.state !== "post" && fresh(l))
    .map(l => [l.game_id, l]));
  const fin = new Map(rows.filter(l => l.state === "post")
    .map(l => [l.game_id, l]));
  for (const g of games) {
    g.live = g.home_points == null ? (lv.get(g.game_id) ?? null) : null;
    g.livePost = g.home_points == null ? (fin.get(g.game_id) ?? null) : null;
  }
}

async function refreshLive() {
  if (!state.games.length) return;
  const idList = state.games.map(g => g.game_id).join(",");
  try {
    const rows = await api(
      `live_predictions?select=game_id,updated_at,state,period,clock,home_points,away_points,situation,home_win_prob,pred_margin,pred_total&game_id=in.(${idList})&limit=400`);
    attachLive(state.games, rows);
    renderList();
    if (state.sel?.live || state.sel?.livePost) renderDetail(state.sel);
  } catch (e) { /* next tick retries */ }
  armLiveTimer();
}

function armLiveTimer() {
  clearTimeout(state.liveTimer);
  // keep polling while anything is live, or could go live (kicked off in
  // the last 5 hours without a final score yet)
  const now = Date.now();
  const watch = state.games.some(g => g.live
    || (g.home_points == null && g.start_date
        && now - new Date(g.start_date).getTime() > -120e3
        && now - new Date(g.start_date).getTime() < 5 * 3600e3));
  if (watch) state.liveTimer = setTimeout(refreshLive, 30_000);
}

/* ------------------------------------------------------- per-game judging */
function judge(g) {
  const played = g.home_points != null && g.away_points != null;
  const margin = played ? g.home_points - g.away_points : null;
  const total  = played ? g.home_points + g.away_points : null;
  const p = g.pred, l = g.line;
  const out = { played, margin, total };

  if (played) {
    out.suHit = (p.home_win_prob > 0.5) === (margin > 0);
    out.modelMarginErr = Math.abs(p.pred_margin - margin);
    out.modelTotalErr  = Math.abs(p.pred_total - total);
  }
  if (l && l.spread != null) {
    out.mm = marketMargin(l);                       // vegas expected home margin
    out.modelPickHome = p.pred_margin > out.mm;     // model's side of the spread
    out.edge = p.pred_margin - out.mm;              // signed disagreement (home view)
    out.absEdge = Math.abs(out.edge);
    if (played) {
      out.vegasMarginErr = Math.abs(out.mm - margin);
      out.cover = margin > out.mm ? "home" : margin < out.mm ? "away" : "push";
      out.atsHit = out.cover === "push" ? null
                 : out.cover === (out.modelPickHome ? "home" : "away");
    }
  }
  if (l && l.over_under != null) {
    out.totalEdge = p.pred_total - l.over_under;
    out.ouPick = p.pred_total > l.over_under ? "over"
               : p.pred_total < l.over_under ? "under" : null;
    if (played) {
      out.vegasTotalErr = Math.abs(l.over_under - total);
      out.ouResult = total > l.over_under ? "over" : total < l.over_under ? "under" : "push";
      out.ouHit = out.ouPick == null || out.ouResult === "push" ? null
                : out.ouPick === out.ouResult;
    }
  }
  return out;
}

function renderLedger() {
  let su = 0, suN = 0, ats = 0, atsN = 0, closer = 0, closerN = 0;
  let e7 = 0, e4 = 0, lined = 0, gapped = 0;
  for (const g of state.games) {
    const j = judge(g);
    if (!j.played) {
      if (j.absEdge != null) {
        lined++;
        // an edge computed without one side's roster isn't a real edge
        if (j.absEdge >= 4 && g.rosterGap?.length) gapped++;
        else if (j.absEdge >= 7) e7++;
        else if (j.absEdge >= 4) e4++;
      }
      continue;
    }
    suN++; if (j.suHit) su++;
    if (j.atsHit != null) { atsN++; if (j.atsHit) ats++; }
    if (j.vegasMarginErr != null) {
      closerN++;
      if (j.modelMarginErr < j.vegasMarginErr) closer++;
    }
  }
  const chips = [];
  if (suN) chips.push(`<span class="ledger-chip">Winners <b>${su}–${suN - su}</b></span>`);
  if (atsN) chips.push(`<span class="ledger-chip">ATS <b>${ats}–${atsN - ats}</b></span>`);
  if (closerN) chips.push(
    `<span class="ledger-chip brand">Closer than Vegas <b>${closer}/${closerN}</b></span>`);
  // upcoming games: how loudly the model disagrees with the current line
  if (e7) chips.push(`<span class="ledger-chip brand">Edges 7+ <b>${e7}</b></span>`);
  if (e4) chips.push(`<span class="ledger-chip">Edges 4–7 <b>${e4}</b></span>`);
  if (gapped) chips.push(`<span class="ledger-chip warn"
    title="Disagreements of 4+ points in games where a team's roster wasn't published when the model ran — not counted as edges">Roster gaps <b>${gapped}</b></span>`);
  if (!suN && lined && !e7 && !e4 && !gapped) chips.push(
    `<span class="ledger-chip">No big disagreements yet <b>${lined} lined</b></span>`);
  $("week-ledger").innerHTML = chips.join("");
}

/* ------------------------------------------------------------- rail list */
const DAY_FMT = new Intl.DateTimeFormat("en-US",
  { weekday: "short", month: "short", day: "numeric" });
const TIME_FMT = new Intl.DateTimeFormat("en-US",
  { hour: "numeric", minute: "2-digit" });

/* local kickoff time; CFBD parks TBD kicks on a placeholder timestamp,
   so the flag decides whether the clock reading is real */
function kickTime(g) {
  if (!g.start_date || g.start_time_tbd) return null;
  return TIME_FMT.format(new Date(g.start_date));
}

function renderList() {
  const list = $("game-list");
  const fbsInv = g => g.home_classification === "fbs" || g.away_classification === "fbs";
  let games = state.games.filter(g =>
    (state.division === "all"
      || (state.division === "fbs" ? fbsInv(g) : !fbsInv(g)))
    && (!state.filter ||
      g.home_team.toLowerCase().includes(state.filter) ||
      g.away_team.toLowerCase().includes(state.filter)));
  $("rail-count").textContent =
    `${games.length}${games.length !== state.games.length ? ` of ${state.games.length}` : ""} games`;
  if (!games.length) {
    list.innerHTML = `<div class="rail-empty">No matchups fit that filter.</div>`;
    return;
  }
  let html = "", i = 0;
  if (state.sort === "edge") {
    games = [...games].sort((a, b) =>
      (judge(b).absEdge ?? -1) - (judge(a).absEdge ?? -1));
    html += `<div class="day-head">By model-vs-line disagreement</div>`;
    for (const g of games) html += rowHtml(g, i++);
  } else {
    let lastDay = "";
    for (const g of games) {
      const day = g.start_date ? DAY_FMT.format(new Date(g.start_date)) : "Scheduled";
      if (day !== lastDay) { html += `<div class="day-head">${esc(day)}</div>`; lastDay = day; }
      html += rowHtml(g, i++);
    }
  }
  list.innerHTML = html;
  list.querySelectorAll(".game-row").forEach(el =>
    el.addEventListener("click", () => select(+el.dataset.gid)));
}

function rowHtml(g, i) {
  const j = judge(g);
  const p = g.pred;
  const ht = teamOf(g, "home"), at = teamOf(g, "away");
  const chips = [];
  if (j.played)
    chips.push(`<span class="chip ${j.suHit ? "hit" : "miss"}">${j.suHit ? "Called" : "Missed"}</span>`);
  if (j.atsHit != null)
    chips.push(`<span class="chip ${j.atsHit ? "hit" : "miss"}">ATS ${j.atsHit ? "✓" : "✗"}</span>`);
  else if (j.cover === "push")
    chips.push(`<span class="chip push">Push</span>`);
  // upcoming game the model quarrels with: mark the disagreement and its side
  if (!j.played && j.absEdge != null && j.absEdge >= 4) {
    const pick = j.modelPickHome ? abbr(ht, g.home_team) : abbr(at, g.away_team);
    const gap = g.rosterGap?.length > 0;
    chips.push(`<span class="chip ${gap ? "warn" : j.absEdge >= 7 ? "edge-hot" : "edge-warm"}"
      title="${gap
        ? `Model departs from the line by ${j.absEdge.toFixed(1)}, but ${esc(g.rosterGap.join(" and "))} had no published roster when it ran — treat with suspicion`
        : `Model departs from the line by ${j.absEdge.toFixed(1)} points toward ${esc(pick)}`}">Edge ${esc(pick)} ${j.absEdge.toFixed(1)}${gap ? " ⚠" : ""}</span>`);
  }
  // any upcoming game priced without a side's roster gets flagged, edge or not
  if (!j.played && g.rosterGap?.length && !(j.absEdge != null && j.absEdge >= 4))
    chips.push(`<span class="chip warn"
      title="No published ${esc(g.rosterGap.join(" or "))} roster when the model ran — its margin may be compressed">⚠ Roster</span>`);
  if (g.line?.spread != null)
    chips.push(`<span class="chip vegas">${esc(fmtSpread(g.line.spread, abbr(ht, g.home_team), abbr(at, g.away_team)))}</span>`);

  const teamLine = (t, name, pts, won) => `
    <span class="row-team">
      ${logoImg(t, name, "row-logo", "row-mono", 22)}
      <span class="row-name">${esc(name)}</span>
      ${pts != null ? `<span class="row-score num ${won ? "won" : ""}">${pts}</span>` : ""}
    </span>`;

  const lv = g.live, fin = g.livePost;
  const aPts = lv ? lv.away_points : fin ? fin.away_points : g.away_points;
  const hPts = lv ? lv.home_points : fin ? fin.home_points : g.home_points;
  return `<button type="button" class="game-row" role="option" data-gid="${g.game_id}"
      aria-selected="${state.sel?.game_id === g.game_id}"
      style="animation-delay:${Math.min(i * 18, 320)}ms">
    <span class="row-teams">
      ${teamLine(at, g.away_team, aPts, (j.played && j.margin < 0) || (fin && hPts < aPts))}
      ${teamLine(ht, g.home_team, hPts, (j.played && j.margin > 0) || (fin && hPts > aPts))}
    </span>
    <span class="row-right">
      ${lv ? `<span class="row-kick live">● Q${lv.period} ${esc(lv.clock)}</span>`
           : fin ? `<span class="row-kick">FINAL</span>`
           : liveGap(g) ? `<span class="row-kick stale" title="Kickoff has passed but no live read is reaching the dashboard">NO LIVE DATA</span>`
           : kickTime(g) ? `<span class="row-kick">${kickTime(g)}</span>`
           : j.played ? "" : `<span class="row-kick">TBD</span>`}
      <span class="row-wp" title="${esc(g.home_team)} win probability${lv ? " · live" : ""}">
        ${logoImg(ht, g.home_team, "wp-logo", "wp-mono", 17)}${pct(lv ? lv.home_win_prob : p.home_win_prob)}</span>
      <span class="row-chips">${chips.join("")}</span>
    </span>
  </button>`;
}

function select(gid, { scroll = true } = {}) {
  state.sel = state.games.find(g => g.game_id === gid) ?? null;
  document.querySelectorAll(".game-row").forEach(el =>
    el.setAttribute("aria-selected", +el.dataset.gid === gid));
  if (state.sel) {
    renderDetail(state.sel);
    if (scroll && matchMedia("(max-width: 980px)").matches)
      $("detail").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

/* --------------------------------------------------------------- detail */
const GAME_FMT = new Intl.DateTimeFormat("en-US",
  { weekday: "long", month: "long", day: "numeric", year: "numeric" });

async function renderDetail(g) {
  $("detail-empty").hidden = true;
  const body = $("detail-body");
  body.hidden = false;

  // Switching games: show the pane's silhouette at once instead of leaving
  // the previous game frozen while game_pmf is fetched. Same-game re-renders
  // (live ticks, theme toggle, view flips) stay in place with no flash.
  if (state.detailGid !== g.game_id) {
    state.detailGid = g.game_id;
    body.innerHTML = `<div class="detail-in sk-detail">
      <div class="sk" style="height:108px"></div>
      <div class="sk" style="height:22px;width:46%"></div>
      <div class="sk-board">
        <div class="sk" style="height:200px"></div>
        <div class="sk" style="height:200px"></div>
        <div class="sk" style="height:200px"></div>
      </div>
      ${loadingHtml("Pricing the matchup")}
    </div>`;
  }

  const j = judge(g);
  const p = g.pred, l = g.line;
  const ht = teamOf(g, "home"), at = teamOf(g, "away");
  const hc = teamColor(ht) || "var(--brand)";
  const ac = teamColor(at) || "var(--ink-3)";
  const hAb = abbr(ht, g.home_team), aAb = abbr(at, g.away_team);
  const kick = kickTime(g);
  const date = (g.start_date ? GAME_FMT.format(new Date(g.start_date)) : "")
    + (kick ? ` · ${kick}` : "");

  const side = (t, name, pts, won, home, sub) => `
    <div class="sb-side ${home ? "home" : ""}">
      ${logoImg(t, name, "sb-logo", "sb-mono", 64)}
      <div class="sb-id">
        <div class="sb-school">${esc(name)}</div>
        <div class="sb-sub">${home ? (g.neutral_site ? "Neutral" : "Home") : (g.neutral_site ? "Neutral" : "Away")}${t?.conference ? ` · ${esc(t.conference)}` : ""}</div>
      </div>
      <div class="sb-scorecol">
        <div class="sb-score ${won ? "won" : ""}" id="sb-${home ? "h" : "a"}">${pts ?? "–"}</div>
        ${sub != null ? `<div class="sb-pred" title="${esc(sub.label)}">${sub.val}</div>` : ""}
      </div>
    </div>`;

  const chips = [
    `<span class="chip ghost">${g.season} · ${g.season_type === "postseason" ? "Postseason" : `Week ${g.week}`}</span>`,
    j.played
      ? `<span class="chip ${p.split === "train" ? "push" : "brand"}">${p.split === "train" ? "In-sample" : p.split === "test" ? "Held out" : "Validation"}</span>`
      : g.livePost
        ? `<span class="chip ghost">Final · grading lands with the official ingest</span>`
        : g.live
          ? `<span class="chip brand">In progress · pregame call locked before kickoff</span>`
          : liveGap(g)
            ? `<span class="chip warn" title="The pregame call below stands; only the live in-game reads are missing.">⚠ Kicked off — no live reads reaching the dashboard</span>`
          : `<span class="chip brand">Upcoming · call locked pregame</span>`,
  ];
  if (j.played) chips.push(`<span class="chip ${j.suHit ? "hit" : "miss"}">${j.suHit ? "Winner called" : "Winner missed"}</span>`);
  if (!j.played && j.absEdge != null && j.absEdge >= 4)
    chips.push(`<span class="chip ${g.rosterGap?.length ? "warn" : j.absEdge >= 7 ? "edge-hot" : "edge-warm"}">Disagrees with the line by ${j.absEdge.toFixed(1)}</span>`);
  if (!j.played && g.rosterGap?.length)
    chips.push(`<span class="chip warn"
      title="CFBD hadn't published this roster when features were built; the model saw league-average roster values instead, which compresses FBS-FCS margins">⚠ No ${esc(g.rosterGap.join(" / "))} roster yet</span>`);
  if (l?.provider) chips.push(`<span class="chip vegas">Line · ${esc(l.provider)}</span>`);

  // the model's own distribution: prices the current line on upcoming games
  // and draws the total-points curve (live season only — no PMF elsewhere)
  let pmf = null;
  try {
    const [row] = await api(
      `game_pmf?select=margin_pmf,total_pmf&game_id=eq.${g.game_id}`);
    pmf = row ?? null;
  } catch (e) { /* curve and EV notes are decoration */ }
  // the user may have clicked another game while that fetch was in flight —
  // painting now would put game A's numbers under game B's header
  if (state.sel?.game_id !== g.game_id) return;

  const lv = g.live, fin = g.livePost, gapNow = liveGap(g);
  // live games can flip the board between the model's read RIGHT NOW and
  // the pregame call it locked before kickoff
  const useLive = !!(lv && lv.pred_margin != null)
    && state.liveView !== "pre";
  const dp = useLive
    ? { ...p, pred_margin: lv.pred_margin, pred_total: lv.pred_total,
        home_win_prob: lv.home_win_prob }
    : p;
  const gd = useLive ? { ...g, pred: dp } : g;
  const jd = useLive ? judge(gd) : j;
  // under-score numbers: the live projected final in live view, the pregame
  // prediction in pregame view and once the game is done
  let subH = null, subA = null;
  if (useLive) {
    const [prH, prA] = projScore(lv.pred_margin, lv.pred_total);
    subH = { val: prH, label: "Model live projected final" };
    subA = { val: prA, label: "Model live projected final" };
  } else if ((j.played || fin || lv) && p) {
    subH = { val: p.pred_home_pts, label: "Pregame prediction" };
    subA = { val: p.pred_away_pts, label: "Pregame prediction" };
  }
  const aPts = lv ? lv.away_points : fin ? fin.away_points : g.away_points;
  const hPts = lv ? lv.home_points : fin ? fin.home_points : g.home_points;
  const aWon = (j.played && j.margin < 0) || (fin && aPts > hPts);
  const hWon = (j.played && j.margin > 0) || (fin && hPts > aPts);
  // the board is re-rendered on chart hover, so its inputs live on state.
  // A live game's total card draws the model's CURRENT distribution (the
  // last scored snap), not the pregame shape — loadCurve fills it in below.
  // The cached copy is seeded here so the 30-second live refresh doesn't
  // flash back to the gauge while the curve refetches.
  const cached = useLive && state.livePmf?.gid === g.game_id
    ? state.livePmf : null;
  state.boardCtx = {
    g: gd, j: jd, hAb, aAb, useLive,
    pmf: useLive ? (cached ? { total_pmf: cached.total_pmf } : null) : pmf,
    onBoard: cached?.onBoard ?? null,
  };
  body.innerHTML = `<div class="detail-in">
    <div class="scorebug" style="--home-color:${hc};--away-color:${ac}">
      ${side(at, g.away_team, aPts, aWon, false, subA)}
      <div class="sb-center">
        <div class="sb-status${lv ? " live" : gapNow ? " stale" : ""}" id="sb-status">${j.played || fin ? "Final" : lv ? "● LIVE" : gapNow ? "No live data" : "Scheduled"}</div>
        <div class="sb-clock" id="sb-clock">${lv ? `Q${lv.period} · ${esc(lv.clock)}` : ""}</div>
        <div class="sb-date">${esc(date)}</div>
        <div class="sb-venue" id="venue"></div>
      </div>
      ${side(ht, g.home_team, hPts, hWon, true, subH)}
    </div>
    <div class="detail-chips">${chips.join("")}</div>

    <div class="sect"><h3>Model vs Vegas</h3>
      <span class="sect-note">${useLive
        ? "the model’s read right now · closing line for scale"
        : "closing line as the benchmark, never an input"}</span>
      ${lv ? `<div class="seg view-seg" id="view-seg" role="tablist" aria-label="Prediction view">
        <button class="seg-btn" data-view="live" role="tab" aria-selected="${useLive}">Live</button>
        <button class="seg-btn" data-view="pre" role="tab" aria-selected="${!useLive}">Pregame</button>
      </div>` : ""}</div>
    <div class="board" id="board">
      ${boardCards(null)}
    </div>
    ${useLive ? "" : ticketHtml(g, j, hAb, aAb, pmf)}

    <div class="sect"><h3>Win probability, snap by snap</h3>
      <span class="sect-note" id="chart-note"></span></div>
    <div class="card chart-card">
      <div class="chart-legend">
        <span class="legend-item"><span class="legend-swatch" style="background:${hc}"></span>${esc(g.home_team)}</span>
        <span class="legend-item"><span class="legend-swatch" style="background:${ac}"></span>${esc(g.away_team)}</span>
      </div>
      <div class="chart-wrap" id="chart-wrap">
        ${loadingHtml("Loading the curve")}
      </div>
    </div>

    <p class="note">The projected score is the median margin composed with the median total from
    one discrete distribution over final margins; win probability is the same distribution’s
    mass above zero, so the three can never disagree. The market’s number is shown for scale —
    it is never an input.</p>
  </div>`;

  body.querySelectorAll("#view-seg .seg-btn").forEach(b =>
    b.addEventListener("click", () => {
      state.liveView = b.dataset.view;
      renderDetail(g);
    }));
  loadVenue(g);
  loadCurve(g, ht, at, hc, ac);
}

/* The three Model-vs-Vegas cards. With a hover point they show the model's
   read at THAT snap instead of the headline call — the market numbers stay
   put, so the cards read as "here is what the model thought at Q3 05:11
   against the same line". Verdict chips are suppressed while scrubbing:
   "covered / winner called" are judgements on the pregame call, not on a
   mid-game state. */
function boardCards(hoverPt) {
  const c = state.boardCtx;
  if (!c) return "";
  if (!hoverPt)
    return marginCard(c.g, c.j, c.hAb, c.aAb, null, c.useLive)
         + totalCard(c.g, c.j, c.pmf, null, c.onBoard)
         + wpCard(c.g, c.j, c.hAb, c.aAb, null, c.useLive);
  const gh = { ...c.g, pred: { ...c.g.pred,
    pred_margin: hoverPt.pred_margin,
    pred_total: hoverPt.pred_total,
    home_win_prob: hoverPt.home_win_prob } };
  const jh = judge(gh);
  const lab = clockStr(hoverPt.game_seconds_left);
  // the snap's OWN distribution over the final total when the scorer stored
  // one (it narrows as the clock runs down); otherwise the pregame shape,
  // with only the marker moving
  const snapPmf = hoverPt.tpmf?.length ? { total_pmf: hoverPt.tpmf } : null;
  const tp = snapPmf ?? c.pmf;
  // the "already scored" pin belongs only to a snap's own distribution —
  // on the pregame shape it would sit where real mass still lives
  const onBoard = snapPmf && hoverPt.hp != null
    ? hoverPt.hp + hoverPt.ap : null;
  return marginCard(gh, jh, c.hAb, c.aAb, lab)
       + totalCard(gh, jh, tp, lab, onBoard)
       + wpCard(gh, jh, c.hAb, c.aAb, lab);
}

function paintBoard(hoverPt) {
  const el = $("board");
  if (el) el.innerHTML = boardCards(hoverPt);
}

/* margin / total gauge cards */
function gauge(markers, zeroLabel) {
  // markers: [{key:'model'|'vegas'|'final', label, value}]
  const vals = markers.map(m => m.value);
  const lo = Math.min(...vals, 0), hi = Math.max(...vals, 0);
  const pad = Math.max((hi - lo) * 0.14, 3);
  const a = lo - pad, b = hi + pad;
  const X = v => ((v - a) / (b - a) * 100).toFixed(2);
  // the label yields when the market pin hangs where it would sit
  const vegasNearZero = markers.some(m =>
    m.key === "vegas" && Math.abs(m.value) / (b - a) < 0.07);
  return `<div class="gauge">
    <div class="gauge-axis"></div>
    <div class="gauge-zero" style="left:${X(0)}%"></div>
    ${zeroLabel && !vegasNearZero
      ? `<div class="gauge-zero-label" style="left:${X(0)}%">${zeroLabel}</div>` : ""}
    ${markers.map(m => `
      <div class="gauge-marker ${m.key}" style="left:${X(m.value)}%" title="${esc(m.label)}">
        <span class="pin">${m.key[0].toUpperCase()}</span>
        <span class="stem"></span>
      </div>`).join("")}
  </div>`;
}

function gaugeRows(rows) {
  return `<div class="gauge-rows">${rows.map(r => `
    <div class="gauge-row">
      <span class="gauge-key ${r.key}">${r.key[0].toUpperCase()}</span>
      <span class="who">${esc(r.who)}</span>
      <span class="val">${r.val}</span>
      <span class="err">${r.err ?? ""}</span>
    </div>`).join("")}</div>`;
}

function marginCard(g, j, hAb, aAb, hoverLabel, live) {
  const p = g.pred, l = g.line;
  // the model's number ALWAYS renders; the market is an overlay when posted
  let inner;
  {
    const markers = [{ key: "model", label: "Model", value: p.pred_margin }];
    const rows = [{
      key: "model", who: "Model",
      val: fmtFav(p.pred_margin, hAb, aAb),
      err: j.played ? `off by ${j.modelMarginErr.toFixed(1)}` : null,
    }];
    if (l?.spread != null) {
      markers.push({ key: "vegas", label: "Vegas", value: j.mm });
      rows.push({
        key: "vegas", who: `Vegas (${l.provider})`,
        val: fmtFav(j.mm, hAb, aAb),
        err: j.played ? `off by ${j.vegasMarginErr.toFixed(1)}` : null,
      });
    }
    if (j.played) {
      markers.push({ key: "final", label: "Final", value: j.margin });
      rows.push({ key: "final", who: "Final margin",
        val: `${j.margin > 0 ? hAb : aAb} by ${Math.abs(j.margin)}`, err: "" });
    }
    let verdict = "";
    if (hoverLabel) {
      verdict = `<div class="card-verdict"><span class="chip brand">Model at ${esc(hoverLabel)}</span></div>`;
    } else if (j.played && l?.spread != null) {
      const v = closerVerdict(j.modelMarginErr, j.vegasMarginErr);
      const pick = j.modelPickHome ? hAb : aAb;
      const ats = j.atsHit == null
        ? `<span class="chip push">Push at ${esc(fmtSpread(l.spread, hAb, aAb))}</span>`
        : `<span class="chip ${j.atsHit ? "hit" : "miss"}">Model took ${esc(pick)} ATS — ${j.atsHit ? "covered" : "did not cover"}</span>`;
      verdict = `<div class="card-verdict"><span class="chip ${v.cls}">${v.label} on margin</span>${ats}</div>`;
    } else if (live) {
      // In progress: the posted number is the PREGAME close, so "the model
      // takes X against it" is not a bet anyone can make — a team down 21 in
      // Q4 would show a huge fake edge. State what this is instead.
      verdict = `<div class="card-verdict"><span class="chip brand">Model’s read right now</span>${
        l?.spread != null
          ? `<span class="chip ghost">${esc(fmtSpread(l.spread, hAb, aAb))} was the pregame close — shown for scale</span>`
          : ""}</div>`;
    } else if (l?.spread != null) {
      // upcoming: the model's standing side (the ticket below prices it)
      const pick = j.modelPickHome ? hAb : aAb;
      const gap = g.rosterGap?.length > 0;
      const cls = gap ? "warn" : j.absEdge >= 7 ? "edge-hot" : j.absEdge >= 4 ? "edge-warm" : "ghost";
      verdict = `<div class="card-verdict"><span class="chip ${cls}"
        ${gap ? `title="${esc(g.rosterGap.join(" and "))} had no published roster when the model ran — this number is suspect"` : ""}
        >Model takes ${esc(pick)} against ${esc(fmtSpread(l.spread, hAb, aAb))} · edge ${j.absEdge.toFixed(1)} pts${gap ? " ⚠" : ""}</span></div>`;
    } else if (!j.played) {
      verdict = `<div class="card-verdict"><span class="chip ghost">No line posted — model stands alone</span></div>`;
    }
    inner = gauge(markers, "even") + gaugeRows(rows) + verdict;
  }
  return `<div class="card"><div class="card-head">
      <span class="card-title">Margin — minus marks the favorite</span></div>${inner}</div>`;
}

/* the model's full distribution over the final total, drawn to scale —
   x is actual points, the curve is the pregame PMF, pins mark the numbers */
function totalCurve(probs, marks) {
  const W = 560, L = 16, R = 16, B = 30, PLOT_H = 120;
  const cdf = []; let c = 0;
  for (const p of probs) cdf.push(c += p);
  let lo = TOTAL_CENTRES[Math.max(cdf.findIndex(v => v >= 0.005), 1)];
  let hi = TOTAL_CENTRES[Math.min(
    cdf.findIndex(v => v >= 0.995) + 1, TOTAL_CENTRES.length - 2)];
  for (const m of marks) {
    lo = Math.min(lo, m.value - 4); hi = Math.max(hi, m.value + 4);
  }
  lo = Math.floor(lo / 5) * 5; hi = Math.ceil(hi / 5) * 5;
  const X = v => L + (v - lo) / (hi - lo) * (W - L - R);
  // Marker labels stack onto as many rows as it takes for none to collide,
  // tracking each row's right edge. Two fixed rows were not enough: a live
  // curve routinely puts points-on-the-board, the model and the O/U within
  // a couple of points of each other, and the third label landed on top of
  // the second. The header grows to fit however many rows are needed.
  const placed = [];
  const rowEnd = [];
  for (const m of [...marks].sort((a, b) => a.value - b.value)) {
    const x = X(m.value);
    let row = 0;
    while (row < rowEnd.length && x - rowEnd[row] < 58) row++;
    rowEnd[row] = x;
    placed.push({ ...m, x, row });
  }
  const T = 24 + Math.max(rowEnd.length, 1) * 16;
  const H = T + PLOT_H + B;
  // display smoothing only; every number quoted elsewhere is exact
  const sm = probs.map((p, i) =>
    (probs[i - 2] ?? 0) * .1 + (probs[i - 1] ?? 0) * .2 + p * .4
    + (probs[i + 1] ?? 0) * .2 + (probs[i + 2] ?? 0) * .1);
  const pts = TOTAL_CENTRES.map((cx, i) => [cx, sm[i]])
    .filter(([cx]) => cx >= lo && cx <= hi);
  const ymax = Math.max(...pts.map(([, p]) => p)) * 1.12 || 1;
  const Y = p => T + (1 - p / ymax) * PLOT_H;
  const line = pts.map(([cx, p], i) =>
    `${i ? "L" : "M"}${X(cx).toFixed(1)},${Y(p).toFixed(1)}`).join("");
  const area = `${line}L${X(pts.at(-1)[0]).toFixed(1)},${H - B}` +
    `L${X(pts[0][0]).toFixed(1)},${H - B}Z`;
  const step = (hi - lo) <= 45 ? 5 : (hi - lo) <= 90 ? 10 : 20;
  let ticks = "";
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step)
    ticks += `<line class="tc-grid" x1="${X(v).toFixed(1)}" y1="${T}"
        x2="${X(v).toFixed(1)}" y2="${H - B}"/>
      <text class="tc-tick" x="${X(v).toFixed(1)}" y="${H - B + 16}"
        text-anchor="middle">${v}</text>`;
  // every pin drops from just above the plot, so a tall pin can never run
  // through a label sitting on a lower row
  let pins = "";
  for (const m of placed) {
    const y = 14 + m.row * 16;
    const tx = Math.min(Math.max(m.x, L + 24), W - R - 24);
    pins += `<line class="tc-mark ${m.key}" x1="${m.x.toFixed(1)}" y1="${T - 8}"
        x2="${m.x.toFixed(1)}" y2="${H - B}"/>
      <text class="tc-lab ${m.key}" x="${tx.toFixed(1)}" y="${y}"
        text-anchor="middle">${esc(m.label)} ${m.value % 1 ? m.value.toFixed(1) : m.value}</text>`;
  }
  return `<svg class="tc" viewBox="0 0 ${W} ${H}" role="img"
      aria-label="Model distribution over total points">
    <line class="tc-base" x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}"/>
    ${ticks}
    <path class="tc-area" d="${area}"/>
    <path class="tc-line" d="${line}"/>
    ${pins}
  </svg>`;
}

function totalCard(g, j, pmf, hoverLabel, onBoard) {
  const p = g.pred, l = g.line;
  const anchor = l?.over_under ?? p.pred_total;
  const markers = [{ key: "model", label: "Model", value: p.pred_total - anchor }];
  const marks = [{ key: "model", label: "M", value: p.pred_total }];
  // points already scored: the distribution is hard-zero below this, so the
  // curve's left edge is a fact about the game, not a clipped chart
  if (onBoard != null)
    marks.push({ key: "now", label: "NOW", value: onBoard });
  const rows = [{
    key: "model", who: "Model", val: p.pred_total.toFixed(1),
    err: j.played ? `off by ${j.modelTotalErr.toFixed(1)}` : null,
  }];
  if (l?.over_under != null) {
    markers.push({ key: "vegas", label: "Vegas", value: 0 });
    marks.push({ key: "vegas", label: "O/U", value: l.over_under });
    rows.push({
      key: "vegas", who: `Vegas O/U`, val: l.over_under.toFixed(1),
      err: j.played ? `off by ${j.vegasTotalErr.toFixed(1)}` : null,
    });
  }
  if (j.played) {
    markers.push({ key: "final", label: "Final", value: j.total - anchor });
    marks.push({ key: "final", label: "F", value: j.total });
    rows.push({ key: "final", who: "Final total", val: String(j.total), err: "" });
  }
  let verdict = "";
  if (hoverLabel) {
    verdict = `<div class="card-verdict"><span class="chip brand">Model at ${esc(hoverLabel)}</span></div>`;
  } else if (j.played && l?.over_under != null) {
    const v = closerVerdict(j.modelTotalErr, j.vegasTotalErr);
    const ou = j.ouHit == null
      ? `<span class="chip push">${j.ouResult === "push" ? "Push on the total" : "Model sat out"}</span>`
      : `<span class="chip ${j.ouHit ? "hit" : "miss"}">Model said ${j.ouPick} — ${j.ouHit ? "right" : "wrong"}</span>`;
    verdict = `<div class="card-verdict"><span class="chip ${v.cls}">${v.label} on total</span>${ou}</div>`;
  } else if (!j.played && l?.over_under != null && j.ouPick && onBoard == null) {
    const d = Math.abs(j.totalEdge);
    verdict = `<div class="card-verdict"><span class="chip ${d >= 7 ? "edge-warm" : "ghost"}">Model leans ${j.ouPick} ${l.over_under.toFixed(1)} by ${d.toFixed(1)}</span></div>`;
  }
  if (!verdict && !j.played && l?.over_under == null && onBoard == null)
    verdict = `<div class="card-verdict"><span class="chip ghost">No total posted — model stands alone</span></div>`;
  // live: a projection that shrinks toward the current score would "lean
  // under" the pregame total in every close-out, which is not a real lean
  if (onBoard != null && !hoverLabel)
    verdict = `<div class="card-verdict"><span class="chip brand">Model’s read right now · ${onBoard} on the board</span></div>`;
  const zeroLabel = l?.over_under != null ? `O/U ${l.over_under.toFixed(1)}` : "";
  // the live season carries the full pregame distribution — draw it to
  // scale; otherwise (older seasons, live view) the gauge, model always on it
  const hasCurve = pmf?.total_pmf?.length === TOTAL_CENTRES.length;
  const inner = hasCurve
    ? totalCurve(pmf.total_pmf, marks) + gaugeRows(rows) + verdict
    : gauge(markers, zeroLabel) + gaugeRows(rows) + verdict;
  return `<div class="card"><div class="card-head">
      <span class="card-title">Total points</span></div>${inner}</div>`;
}

function wpCard(g, j, hAb, aAb, hoverLabel, live) {
  const p = g.pred;
  const vwp = vegasWP(g.line);
  const bar = (who, cls, v) => `
    <div class="wp-item">
      <div class="wp-top"><span class="wp-who">${who}</span>
        <span class="wp-val">${pct(v)}</span></div>
      <div class="wp-bar"><span class="wp-fill ${cls}" style="width:${(v * 100).toFixed(1)}%"></span>
        <span class="wp-mid"></span></div>
    </div>`;
  let caption = "";
  if (hoverLabel) {
    const lead = p.home_win_prob > 0.5;
    caption = `<div class="wp-caption">At ${esc(hoverLabel)} the model gave
      ${esc(lead ? g.home_team : g.away_team)}
      ${pct(lead ? p.home_win_prob : 1 - p.home_win_prob)}.</div>`;
  } else if (j.played) {
    const winner = j.margin > 0 ? g.home_team : g.away_team;
    caption = `<div class="wp-caption">${esc(winner)} won it —
      the model priced that outcome at ${pct(j.margin > 0 ? p.home_win_prob : 1 - p.home_win_prob)}${
      vwp != null ? `, the moneyline at ${pct(j.margin > 0 ? vwp : 1 - vwp)}` : ""}.</div>`;
  } else if (vwp != null && live) {
    // the ticket is suppressed during live play, so don't point at it
    const pickHome = p.home_win_prob > 0.5;
    caption = `<div class="wp-caption">From the current state the model gives
      ${esc(pickHome ? g.home_team : g.away_team)}
      ${pct(pickHome ? p.home_win_prob : 1 - p.home_win_prob)}. The pregame moneyline,
      de-vigged, implied ${pct(pickHome ? vwp : 1 - vwp)} before kickoff.</div>`;
  } else if (vwp != null) {
    const pickHome = p.home_win_prob > 0.5;
    caption = `<div class="wp-caption">The model gives ${esc(pickHome ? g.home_team : g.away_team)}
      ${pct(pickHome ? p.home_win_prob : 1 - p.home_win_prob)}; the de-vigged moneyline implies
      ${pct(pickHome ? vwp : 1 - vwp)} — the ticket below prices that gap.</div>`;
  }
  return `<div class="card"><div class="card-head">
      <span class="card-title">${esc(g.home_team)} win probability</span></div>
    <div class="wp-pair">
      ${bar("Model", "model", p.home_win_prob)}
      ${vwp != null ? bar("Vegas moneyline, de-vigged", "vegas", vwp)
                    : `<div class="no-line" style="padding:6px 2px 0">No moneyline recorded — spread and total carry the market view here.</div>`}
    </div>${caption}</div>`;
}

/* -------------------------------------------------- the model's ticket
   One meter per bet: the fill is the MODEL's win probability, the gold tick
   is the break-even probability the posted price demands. A fill past its
   tick is positive expectation; the row's number is expected winnings on a
   flat $100. Upcoming games only — played games are graded above. */
function ticketHtml(g, j, hAb, aAb, pmf) {
  if (j.played || !g.line) return "";
  const l = g.line, p = g.pred;
  const rows = [];

  if (l.spread != null && pmf?.margin_pmf) {
    const t = tailProb(MARGIN_CENTRES, pmf.margin_pmf, j.mm);
    const pWin = j.modelPickHome ? t.over : 1 - t.over - t.push;
    const side = j.modelPickHome ? l.spread : -l.spread;   // pick's handicap
    rows.push({
      kind: "Spread",
      pick: `${j.modelPickHome ? hAb : aAb} ${side > 0 ? "+" : "−"}${Math.abs(side).toFixed(1).replace(/\.0$/, "")}`,
      prob: pWin, need: BREAK_EVEN_110, ev: evAt110(pWin, t.push),
      cap: `model ${pct(pWin)} to cover · −110 needs ${pct(BREAK_EVEN_110)}`,
    });
  }
  if (l.over_under != null && j.ouPick && pmf?.total_pmf) {
    const t = tailProb(TOTAL_CENTRES, pmf.total_pmf, l.over_under);
    const pWin = j.ouPick === "over" ? t.over : 1 - t.over - t.push;
    rows.push({
      kind: "Total",
      pick: `${j.ouPick === "over" ? "Over" : "Under"} ${l.over_under.toFixed(1)}`,
      prob: pWin, need: BREAK_EVEN_110, ev: evAt110(pWin, t.push),
      cap: `model ${pct(pWin)} to hit · −110 needs ${pct(BREAK_EVEN_110)}`,
    });
  }
  if (l.home_moneyline != null && l.away_moneyline != null) {
    // the model's moneyline play is whichever side ITS probabilities pay on
    const evOf = (prob, ml) =>
      prob * (ml < 0 ? 10000 / (-ml) : ml) - (1 - prob) * 100;
    const sides = [
      { home: true, prob: p.home_win_prob, ml: l.home_moneyline },
      { home: false, prob: 1 - p.home_win_prob, ml: l.away_moneyline },
    ].map(s => ({ ...s, ev: evOf(s.prob, s.ml),
                  need: s.ml < 0 ? -s.ml / (-s.ml + 100) : 100 / (s.ml + 100) }));
    const s = sides[0].ev >= sides[1].ev ? sides[0] : sides[1];
    rows.push({
      kind: "Moneyline",
      pick: `${s.home ? hAb : aAb} ${s.ml > 0 ? "+" : ""}${s.ml}`,
      prob: s.prob, need: s.need, ev: s.ev,
      cap: `model ${pct(s.prob)} to win outright · ${s.ml > 0 ? "+" : ""}${s.ml} needs ${pct(s.need)}`,
    });
  }
  if (!rows.length) return "";

  const evd = v => `<span class="tk-ev ${v >= 0 ? "pos" : "neg"}">${v >= 0 ? "+" : "−"}$${Math.abs(v).toFixed(0)}</span>`;
  const body = rows.map(r => `
    <div class="tk-row">
      <div class="tk-head">
        <span class="tk-kind">${r.kind}</span>
        <span class="tk-pick">${esc(r.pick)}</span>
        ${evd(r.ev)}
      </div>
      <div class="tk-meter" title="${esc(r.cap)}">
        <span class="tk-fill${r.prob > r.need ? "" : " short"}" style="width:${(r.prob * 100).toFixed(1)}%"></span>
        <span class="tk-need" style="left:${(r.need * 100).toFixed(1)}%"></span>
      </div>
      <div class="tk-cap">${esc(r.cap)}</div>
    </div>`).join("");

  const gapNote = g.rosterGap?.length
    ? `<div class="tk-gap">⚠ ${esc(g.rosterGap.join(" and "))} had no published roster when the
       model ran; it saw league-average roster values instead. These prices are suspect.</div>`
    : "";
  return `
    <div class="sect"><h3>Expected winnings</h3>
      <span class="sect-note">the model's ticket · $100 a bet, priced at the posted numbers</span></div>
    <div class="card tk-card">
      ${gapNote}${body}
    </div>`;
}

async function loadVenue(g) {
  if (!g.venue_id) return;
  try {
    const [v] = await api(`venues?select=name,city,state&venue_id=eq.${g.venue_id}`);
    if (v && state.sel?.game_id === g.game_id)
      $("venue").textContent = [v.name, [v.city, v.state].filter(Boolean).join(", ")]
        .filter(Boolean).join(" · ");
  } catch (e) { /* venue is decoration */ }
}

/* ------------------------------------------------------------- wp chart */
const W = 1000, H = 380, L = 46, R = 14, T = 18, B = 34;
const PW = W - L - R, PH = H - T - B;
const X = t => L + ((3600 - t) / 3600) * PW;
const Y = v => T + (1 - v) * PH;

const clockStr = t => {
  const q = Math.min(Math.max(1 + Math.floor((3600 - t) / 900), 1), 4);
  const rem = Math.max(t - (4 - q) * 900, 0);
  return `Q${q} ${String(Math.floor(rem / 60)).padStart(2, "0")}:${String(Math.floor(rem % 60)).padStart(2, "0")}`;
};

async function loadCurve(g, ht, at, hc, ac) {
  const wrap = $("chart-wrap");
  let preds = [], raws = [];
  try {
    [preds, raws] = await Promise.all([
      api(`play_predictions?select=play_index,game_seconds_left,home_lead,home_win_prob,pred_margin,pred_total&game_id=eq.${g.game_id}&order=play_index.asc&limit=1000`),
      api(`play_raw?select=play_index,down,distance,home_has_ball,yds,is_scrim,is_pass,is_sack,is_to,is_pen,total_pts_before,home_lead&game_id=eq.${g.game_id}&order=play_index.asc&limit=1000`),
    ]);
  } catch (e) { /* falls through to the empty panel */ }
  if (state.sel?.game_id !== g.game_id) return;
  if (!preds.length) {
    // No published play curve: fall back to the live reads. This covers
    // games in progress AND finished ones whose official play-by-play
    // hasn't been ingested yet — the official score arriving must not make
    // the curve we drew all game disappear.
    let lc = [];
    try {
      lc = await api(`live_curve?select=play_index,t_left,period,clock,down,distance,situation,play_text,home_points,away_points,home_win_prob,pred_margin,pred_total,total_pmf&game_id=eq.${g.game_id}&order=play_index.asc&limit=2000`);
    } catch (e) { /* falls through */ }
    if (state.sel?.game_id !== g.game_id) return;
    if (lc.length >= 2) {
      const pts = lc.map(r => ({
        play_index: r.play_index, game_seconds_left: r.t_left,
        home_lead: r.home_points - r.away_points,
        hp: r.home_points, ap: r.away_points,
        sit: r.situation || `Q${r.period} ${r.clock}`,
        desc: r.play_text || "",
        home_win_prob: r.home_win_prob, pred_margin: r.pred_margin,
        pred_total: r.pred_total, tpmf: r.total_pmf || null,
      }));
      // a handful of points is a coarse poll, not a snap-by-snap curve —
      // say so rather than letting a smooth line imply detail we lack
      const sparse = pts.length < 10;
      $("chart-note").textContent = g.live
        ? `LIVE · ${pts.length} snap${pts.length === 1 ? "" : "s"} so far · refreshes every 30s`
        : sparse
          ? `${pts.length} coarse reads · no play-by-play was published for this game`
          : `final · ${pts.length} snaps from the live feed`;
      wrap.innerHTML = chartSvg(pts, hc, ac) + `<div id="tip"></div>` + `
        <div class="chart-foot">
          <span>Pregame ${pct(g.pred.home_win_prob)} home</span>
          <span>Q1 · Q2 · Q3 · Q4 →</span>
        </div>`;
      // hand the board the model's CURRENT distribution over the final
      // total. The last snap is the same state live_predictions headlines,
      // so the curve and the numbers beside it always agree.
      const now = pts[pts.length - 1];
      if (state.boardCtx?.useLive && now?.tpmf?.length) {
        state.livePmf = { gid: g.game_id, total_pmf: now.tpmf,
                          onBoard: now.hp + now.ap };
        state.boardCtx.pmf = { total_pmf: now.tpmf };
        state.boardCtx.onBoard = now.hp + now.ap;
        paintBoard(null);
      }
      wireHover(g, pts);
      return;
    }
    // Below FCS there is no play-by-play to draw from: ESPN's feed carries
    // scores for those games but no plays, and cfbfastR does not cover them
    // either. Blaming "recent seasons only" there is simply wrong — as of
    // 2026 week 1 that is 52 of the 125 completed games, every one of them
    // Division II or lower.
    // Play-by-play exists only when BOTH sides are FBS or FCS. A null
    // classification means a school outside those divisions entirely (e.g.
    // St. Francis (IL), an NAIA program) — treating null as "covered" left
    // those games blaming the season range instead of the real cause.
    const pbpCovered = c => c === "fbs" || c === "fcs";
    const noPbpLevel = !(pbpCovered(g.home_classification)
                      && pbpCovered(g.away_classification));
    const why = g.live
      ? "Game is live — the curve starts drawing after a couple of reads."
      : g.home_points == null && !g.livePost
        ? "This game hasn’t kicked off — the snap-by-snap curve appears once it’s played."
        : noPbpLevel
          ? "No play-by-play is published at this level of college football, so there is no snap-by-snap curve to draw — every pregame number above is still the model’s."
          : "No snap-by-snap curve is published for this game — curves cover recent seasons (2024+); every pregame number above is still the model’s.";
    wrap.innerHTML = `<div class="rail-empty" style="padding:56px 20px">${why}</div>`;
    $("chart-note").textContent = "";
    return;
  }
  const rawBy = new Map(raws.map(r => [r.play_index, r]));
  const pts = preds.map(p => ({ ...p, raw: rawBy.get(p.play_index) }));
  $("chart-note").textContent = `${pts.length} snaps · hover for the model’s live read`;
  wrap.innerHTML = chartSvg(pts, hc, ac) + `<div id="tip"></div>` + `
    <div class="chart-foot">
      <span>Pregame ${pct(g.pred.home_win_prob)} home</span>
      <span>Q1 · Q2 · Q3 · Q4 →</span>
    </div>`;
  wireHover(g, pts);
}

function chartSvg(pts, hc, ac) {
  const line = pts.map(p => `${X(p.game_seconds_left).toFixed(1)},${Y(p.home_win_prob).toFixed(1)}`).join(" ");
  const first = X(pts[0].game_seconds_left), last = X(pts[pts.length - 1].game_seconds_left);
  const area = `${first},${Y(.5)} ${line} ${last},${Y(.5)}`;
  const grid = [0, .25, .5, .75, 1].map(v => `
    <line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)" stroke-width="${v === .5 ? 0 : 1}"/>
    <text x="${L - 8}" y="${Y(v) + 3.5}" text-anchor="end" fill="var(--ink-3)"
      font-family="Spline Sans Mono" font-size="9.5">${v * 100}</text>`).join("");
  const quarters = [1, 2, 3, 4].map(q => `
    <line x1="${L + q / 4 * PW}" x2="${L + q / 4 * PW}" y1="${T}" y2="${T + PH}"
      stroke="var(--line)" stroke-dasharray="1 5"/>
    <text x="${L + (q - .5) / 4 * PW}" y="${T + PH + 22}" text-anchor="middle"
      fill="var(--ink-3)" font-family="Archivo" font-weight="700" font-size="10.5"
      letter-spacing="2.5">Q${q}</text>`).join("");
  return `<svg id="wpsvg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Win probability by play">
    <defs>
      <clipPath id="clip-above"><rect x="0" y="0" width="${W}" height="${Y(.5)}"/></clipPath>
      <clipPath id="clip-below"><rect x="0" y="${Y(.5)}" width="${W}" height="${H - Y(.5)}"/></clipPath>
      <linearGradient id="grad-h" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="${hc}" stop-opacity=".30"/>
        <stop offset="1" stop-color="${hc}" stop-opacity=".02"/></linearGradient>
      <linearGradient id="grad-a" x1="0" y1="1" x2="0" y2="0">
        <stop offset="0" stop-color="${ac}" stop-opacity=".28"/>
        <stop offset="1" stop-color="${ac}" stop-opacity=".02"/></linearGradient>
    </defs>
    ${grid}${quarters}
    <polygon points="${area}" fill="url(#grad-h)" clip-path="url(#clip-above)"/>
    <polygon points="${area}" fill="url(#grad-a)" clip-path="url(#clip-below)"/>
    <line x1="${L}" x2="${W - R}" y1="${Y(.5)}" y2="${Y(.5)}" stroke="var(--ink-3)" stroke-width="1.2"/>
    <polyline class="wp-path" points="${line}" fill="none" stroke="var(--ink)"
      stroke-width="1.8" stroke-linejoin="round"/>
    <line id="xhair" y1="${T}" y2="${T + PH}" stroke="var(--brand)" stroke-width="1" opacity="0"/>
    <circle id="xdot" r="4" fill="var(--brand)" stroke="var(--bg-raised)" stroke-width="2" opacity="0"/>
  </svg>`;
}

function playDesc(r) {
  if (!r) return null;
  const dn = ["", "1st", "2nd", "3rd", "4th"][r.down] || "";
  const sit = r.down > 0 ? `${dn} & ${r.distance}` : "Special teams";
  const off = r.home_has_ball ? "home ball" : "away ball";
  let res;
  if (r.is_to) res = "Turnover";
  else if (r.is_sack) res = `Sack for ${r.yds <= 0 ? r.yds : -r.yds}`;
  else if (r.is_pen) res = "Penalty";
  else if (r.is_pass && r.is_scrim) res = `Pass ${r.yds >= 0 ? "+" : ""}${r.yds}`;
  else if (r.is_scrim) res = `Rush ${r.yds >= 0 ? "+" : ""}${r.yds}`;
  else res = "No scrimmage play";
  return { sit, off, res };
}
const projScore = (m, t) => {
  let mm = Math.round(m), tt = Math.round(t);
  if ((Math.abs(mm) % 2) !== (tt % 2)) tt += 1;
  return [(tt + mm) / 2, (tt - mm) / 2];
};

function wireHover(g, pts) {
  const svg = $("wpsvg"), tip = $("tip"), wrap = $("chart-wrap");
  const xh = $("xhair"), dot = $("xdot");
  const xs = pts.map(p => X(p.game_seconds_left));
  let painted = -1;          // repaint the board only when the snap changes
  const nearest = mx => {
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] < mx) lo = m; else hi = m; }
    return (mx - xs[lo] < xs[hi] - mx) ? lo : hi;
  };
  const move = ev => {
    const r = svg.getBoundingClientRect();
    const mx = (ev.clientX - r.left) / r.width * W;
    if (mx < L || mx > W - R) return leave();
    const i = nearest(mx), p = pts[i], raw = p.raw;
    const x = xs[i], y = Y(p.home_win_prob);
    xh.setAttribute("x1", x); xh.setAttribute("x2", x); xh.setAttribute("opacity", .55);
    dot.setAttribute("cx", x); dot.setAttribute("cy", y); dot.setAttribute("opacity", 1);
    const hs = p.hp ?? (raw ? (raw.total_pts_before + raw.home_lead) / 2 : null);
    const as = p.ap ?? (raw ? (raw.total_pts_before - raw.home_lead) / 2 : null);
    // live points carry ESPN's own play call; historical points derive one
    const d = p.desc || p.sit
      ? { sit: p.sit || "", off: "", res: p.desc || "" }
      : playDesc(raw);
    const [prH, prA] = projScore(p.pred_margin, p.pred_total);
    tip.innerHTML = `
      <div class="t-clock">${clockStr(p.game_seconds_left)} · snap ${p.play_index + 1}</div>
      <div class="t-score">${as ?? "?"} <small>${esc(g.away_team)}</small>
        &nbsp;·&nbsp; ${hs ?? "?"} <small>${esc(g.home_team)}</small></div>
      <div class="t-play">${d ? [d.sit, d.off, d.res].filter(Boolean).join(" · ") : "situation unavailable"}</div>
      <table>
        <tr><td>Home win probability</td><td>${pct(p.home_win_prob)}</td></tr>
        <tr><td>Projected final</td><td>${prA}–${prH}</td></tr>
        <tr><td>Projected margin</td><td>${signed(p.pred_margin)}</td></tr>
        <tr><td>Projected total</td><td>${p.pred_total.toFixed(1)}</td></tr>
      </table>`;
    tip.style.display = "block";
    // sit beside the crosshair, never on top of it: prefer the right, flip
    // to the left when the card would run past the edge (clamping instead
    // would slide the card back over the cursor late in the game)
    const wr = wrap.getBoundingClientRect();
    const px = (x / W) * wr.width;
    const tw = tip.offsetWidth || 270;
    const GAP = 18;
    let tl = px + GAP;
    if (tl + tw > wr.width - 8) tl = px - GAP - tw;
    tip.style.left = `${Math.max(8, Math.min(tl, wr.width - tw - 8))}px`;
    tip.style.top = "16px";
    if (i !== painted) { painted = i; paintBoard(p); }
    $("sb-h").textContent = hs ?? "–";
    $("sb-a").textContent = as ?? "–";
    $("sb-status").textContent = "Model rewind";
    $("sb-clock").textContent = raw && d ? `${clockStr(p.game_seconds_left)} · ${d.sit}` : clockStr(p.game_seconds_left);
  };
  const leave = () => {
    tip.style.display = "none";
    xh.setAttribute("opacity", 0); dot.setAttribute("opacity", 0);
    if (painted !== -1) { painted = -1; paintBoard(null); }
    // restore the scorebug to its real state — live and pre-ingest finals
    // included, not just the official games the rewind was written for
    const lv = g.live, fin = g.livePost;
    $("sb-h").textContent =
      (lv ? lv.home_points : fin ? fin.home_points : g.home_points) ?? "–";
    $("sb-a").textContent =
      (lv ? lv.away_points : fin ? fin.away_points : g.away_points) ?? "–";
    $("sb-status").textContent =
      g.home_points != null || fin ? "Final" : lv ? "● LIVE"
        : liveGap(g) ? "No live data" : "Scheduled";
    $("sb-clock").textContent = lv ? `Q${lv.period} · ${lv.clock}` : "";
  };
  svg.addEventListener("mousemove", move);
  svg.addEventListener("mouseleave", leave);
  leave();
}

/* ------------------------------------------------------------- results page */
const SPLITS = {
  held:  { keep: ["test", "val"],      label: "held-out" },
  train: { keep: ["train", "pre2014"], label: "in-sample" },
  all:   { keep: ["test", "val", "train", "pre2014"], label: "all" },
};
let aggRows = null;         // raw view rows, fetched once
let aggSplit = "held";
let aggSeason = "all";      // "all" | "2026" (the live season alone)
let aggDiv = "all";         // "all" | "fbs" (FBS-involved) | "fcs"
const LIVE_SEASON = 2026;
const isRetrospectiveRow = r => r.reconstruction === true || r.backfilled === true
  || (r.season < LIVE_SEASON && String(r.model_version || "").includes("full-context"))
  || ["retrospective", "reconstructed", "backfilled", "historical"].includes(r.split);

function route() {
  const page = location.hash === "#results" ? "results"
             : location.hash === "#model" ? "model" : "slate";
  $("results-page").hidden = page !== "results";
  $("model-page").hidden = page !== "model";
  $("layout").hidden = page !== "slate";
  $("slate-controls").style.visibility = page === "slate" ? "visible" : "hidden";
  for (const t of ["slate", "results", "model"])
    $(`tab-${t}`).setAttribute("aria-current", page === t ? "page" : "false");
  if (page === "results") loadResults();
  if (page === "model") loadModel();
}
window.addEventListener("hashchange", route);

document.querySelectorAll(".seg-btn[data-split]").forEach(b => {
  b.addEventListener("click", () => {
    aggSplit = b.dataset.split;
    renderResults();
  });
});
document.querySelectorAll(".seg-btn[data-season]").forEach(b => {
  b.addEventListener("click", () => {
    aggSeason = b.dataset.season;
    renderResults();
  });
});
document.querySelectorAll(".seg-btn[data-rdiv]").forEach(b => {
  b.addEventListener("click", () => {
    aggDiv = b.dataset.rdiv;
    renderResults();
  });
});

async function loadResults() {
  // show what we have, then always re-fetch: the ledger grows as finals are
  // ingested, and a cached copy would look frozen for the whole session
  const first = !aggRows;
  if (!first) renderResults();
  try {
    const fresh = await api("v_model_vs_vegas?select=*&order=season.asc&limit=500");
    const changed = JSON.stringify(fresh) !== JSON.stringify(aggRows);
    aggRows = fresh;
    // open on the live season when it has graded games, since that's the
    // one still being decided; otherwise the full history
    if (first && aggRows.some(r => r.season === LIVE_SEASON))
      aggSeason = String(LIVE_SEASON);
    if (first || changed) renderResults();
  } catch (e) {
    if (first) $("results-body").innerHTML =
      `<div class="rail-empty" style="padding:60px">Could not load results — ${esc(e.message)}</div>`;
  }
}

/* weighted merge of view rows (per season-split) into one accumulator */
function foldRows(rows) {
  const z = {
    n: 0, su: 0, suV: 0, ats: 0, atsN: 0, closerM: 0,
    mMaeW: 0, vMaeW: 0, mSqW: 0, vSqW: 0,
    nT: 0, mTotW: 0, vTotW: 0, closerT: 0,
    nMl: 0, mBrW: 0, vBrW: 0,
    edges: { e0: { w: 0, n: 0, mlN: 0, mlPl: 0 }, e2: { w: 0, n: 0, mlN: 0, mlPl: 0 },
             e4: { w: 0, n: 0, mlN: 0, mlPl: 0 }, e7: { w: 0, n: 0, mlN: 0, mlPl: 0 } },
  };
  for (const r of rows) {
    for (const k of ["e0", "e2", "e4", "e7"]) {
      z.edges[k].w += r[`ats_${k}_w`] ?? 0;
      z.edges[k].n += r[`ats_${k}_n`] ?? 0;
      z.edges[k].mlN += r[`ml_${k}_n`] ?? 0;
      z.edges[k].mlPl += r[`ml_${k}_pl`] ?? 0;
    }
    z.n += r.n_games; z.su += r.su_wins; z.suV += r.su_wins_vegas ?? 0;
    z.ats += r.ats_wins; z.atsN += r.ats_graded; z.closerM += r.model_closer_margin;
    z.mMaeW += r.model_margin_mae * r.n_games;
    z.vMaeW += r.vegas_margin_mae * r.n_games;
    z.mSqW  += r.model_margin_rmse ** 2 * r.n_games;
    z.vSqW  += r.vegas_margin_rmse ** 2 * r.n_games;
    z.nT += r.n_totals; z.closerT += r.model_closer_total;
    if (r.n_totals) {
      z.mTotW += r.model_total_mae * r.n_totals;
      z.vTotW += r.vegas_total_mae * r.n_totals;
    }
    z.nMl += r.n_ml;
    if (r.n_ml) { z.mBrW += r.model_brier * r.n_ml; z.vBrW += r.vegas_brier * r.n_ml; }
  }
  if (!z.n) return null;
  return {
    n: z.n, su: z.su, suV: z.suV, ats: z.ats, atsN: z.atsN, closerM: z.closerM,
    mMae: z.mMaeW / z.n, vMae: z.vMaeW / z.n,
    mRmse: Math.sqrt(z.mSqW / z.n), vRmse: Math.sqrt(z.vSqW / z.n),
    nT: z.nT, closerT: z.closerT,
    mTot: z.nT ? z.mTotW / z.nT : null, vTot: z.nT ? z.vTotW / z.nT : null,
    nMl: z.nMl, mBr: z.nMl ? z.mBrW / z.nMl : null, vBr: z.nMl ? z.vBrW / z.nMl : null,
    edges: z.edges,
  };
}

function renderResults() {
  document.querySelectorAll(".seg-btn[data-split]").forEach(b =>
    b.setAttribute("aria-selected", b.dataset.split === aggSplit));
  document.querySelectorAll(".seg-btn[data-season]").forEach(b =>
    b.setAttribute("aria-selected", b.dataset.season === aggSeason));
  document.querySelectorAll(".seg-btn[data-rdiv]").forEach(b =>
    b.setAttribute("aria-selected", b.dataset.rdiv === aggDiv));
  const keep = new Set(SPLITS[aggSplit].keep);
  const rows = aggRows.filter(r => !isRetrospectiveRow(r) && keep.has(r.split)
    && (aggSeason === "all" || r.season === Number(aggSeason))
    && (aggDiv === "all" || (r.div_group ?? "fbs") === aggDiv));
  const all = foldRows(rows);
  if (!all) {
    // say which filter emptied it — a blank ledger otherwise reads as broken
    const scope = [
      aggSeason === "all" ? null : `${aggSeason}`,
      aggDiv === "all" ? null : aggDiv.toUpperCase(),
      SPLITS[aggSplit].label,
    ].filter(Boolean).join(" · ");
    const live = aggSeason === String(LIVE_SEASON)
      ? ` The ${LIVE_SEASON} ledger fills in as finals are ingested — live games are graded on the Slate.`
      : "";
    $("results-body").innerHTML =
      `<div class="rail-empty" style="padding:60px">No graded games yet for <b>${esc(scope)}</b>.${live}</div>`;
    return;
  }
  const seasons = [...new Set(rows.map(r => r.season))].sort();
  const bySeason = seasons.map(s => ({ season: s, ...foldRows(rows.filter(r => r.season === s)) }));
  const scopeNote = [
    aggDiv === "all" ? null : aggDiv === "fbs" ? "FBS games" : "FCS-only games",
    `${SPLITS[aggSplit].label} games`,
  ].filter(Boolean).join(" · ");

  const dTile = (label, mVal, vVal, unit, note, lowerIsBetter = true, dec = 2) => {
    const good = lowerIsBetter ? mVal < vVal : mVal > vVal;
    const d = mVal - vVal;
    return `<div class="tile">
      <div class="tile-label">${label}</div>
      <div class="tile-main"><span class="tile-value">${mVal.toFixed(dec)}</span>
        <span class="tile-unit">${unit}</span></div>
      <div class="tile-vs">Vegas <span class="num">${vVal.toFixed(dec)}</span>
        <span class="tile-delta ${good ? "good" : "bad"}">${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(dec)}</span></div>
      ${note ? `<div class="tile-note">${note}</div>` : ""}
    </div>`;
  };
  const pTile = (label, num, den, note, vNum) => {
    const p = num / den * 100;
    const v = vNum != null && den ? vNum / den * 100 : null;
    const d = v == null ? null : p - v;          // higher accuracy is better
    return `<div class="tile">
      <div class="tile-label">${label}</div>
      <div class="tile-main"><span class="tile-value">${p.toFixed(1)}%</span></div>
      ${v == null
        ? `<div class="tile-vs"><span class="num">${num.toLocaleString()}</span>&nbsp;of&nbsp;<span class="num">${den.toLocaleString()}</span></div>`
        : `<div class="tile-vs">Vegas <span class="num">${v.toFixed(1)}%</span>
             <span class="tile-delta ${d >= 0 ? "good" : "bad"}">${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(1)}</span></div>`}
      ${note ? `<div class="tile-note">${note}</div>` : ""}
    </div>`;
  };

  const tiles = [
    dTile("Margin · mean abs error", all.mMae, all.vMae, "pts",
          "how far each number landed from the final margin"),
    dTile("Margin · RMSE", all.mRmse, all.vRmse, "pts",
          "big misses punished harder"),
    pTile("Winners called", all.su, all.n,
          `straight-up accuracy · ${all.su.toLocaleString()} of ${all.n.toLocaleString()} · Vegas picks the spread favorite`,
          all.suV),
    pTile("Against the spread", all.ats, all.atsN,
          `model's side of the closing spread · pushes excluded`),
    pTile("Closer than Vegas", all.closerM, all.n, "game-by-game, on margin"),
  ];
  if (all.mTot != null)
    tiles.push(dTile("Total · mean abs error", all.mTot, all.vTot, "pts",
                     `${all.nT.toLocaleString()} games with a posted O/U`));
  if (all.mBr != null)
    tiles.push(dTile("Win prob · Brier score", all.mBr, all.vBr, "",
                     `${all.nMl.toLocaleString()} games with moneylines, de-vigged`, true, 4));

  const tableRows = bySeason.map(r => {
    const adv = r.mMae < r.vMae;
    return `<tr>
      <td>${r.season}</td>
      <td>${r.n.toLocaleString()}</td>
      <td>${(r.su / r.n * 100).toFixed(1)}%</td>
      <td>${r.ats}–${r.atsN - r.ats} <span style="color:var(--ink-3)">(${(r.ats / r.atsN * 100).toFixed(1)}%)</span></td>
      <td class="${adv ? "adv-model" : ""}">${r.mMae.toFixed(2)}</td>
      <td class="${adv ? "" : "adv-vegas"}">${r.vMae.toFixed(2)}</td>
      <td>${signed(r.mMae - r.vMae, 2)}</td>
      <td>${(r.closerM / r.n * 100).toFixed(0)}%</td>
    </tr>`;
  }).join("");

  const EDGE_LABELS = { e0: "under 2 pts", e2: "2\u20134 pts", e4: "4\u20137 pts", e7: "7+ pts" };
  const money = v => `${v >= 0 ? "+" : "\u2212"}$${Math.abs(Math.round(v)).toLocaleString()}`;
  const roi = (pl, n) => n ? ` <span style="color:var(--ink-3)">(${pl >= 0 ? "+" : "\u2212"}${Math.abs(pl / n).toFixed(1)}%)</span>` : "";
  const atsPl = e => e.w * (100 / 1.1) - (e.n - e.w) * 100;   // flat $100 at \u2212110
  const edgeRows = Object.entries(all.edges).map(([k, e]) => {
    if (!e.n) return "";
    const pctv = e.w / e.n * 100;
    const strong = k === "e7" && pctv > 52.4;
    const apl = atsPl(e);
    return `<tr>
      <td>${EDGE_LABELS[k]}</td>
      <td>${e.w.toLocaleString()}\u2013${(e.n - e.w).toLocaleString()}</td>
      <td class="${strong ? "adv-model" : ""}">${pctv.toFixed(1)}%</td>
      <td class="${apl > 0 ? "adv-model" : ""}">${money(apl)}${roi(apl, e.n)}</td>
      <td class="${e.mlPl > 0 ? "adv-model" : ""}">${e.mlN ? money(e.mlPl) + roi(e.mlPl, e.mlN) : "\u2014"}</td>
      <td><span class="share-bar"><span class="share-fill" style="width:${Math.min((pctv - 40) / 25 * 100, 100).toFixed(0)}%;${pctv < 52.4 ? "background:var(--ink-3)" : ""}"></span></span></td>
    </tr>`;
  }).join("");
  // the disagreement wallet: every 4+ pt departure, $100 a game
  const big = { w: all.edges.e4.w + all.edges.e7.w, n: all.edges.e4.n + all.edges.e7.n,
                mlN: all.edges.e4.mlN + all.edges.e7.mlN, mlPl: all.edges.e4.mlPl + all.edges.e7.mlPl };
  const bigLine = big.n ? `$100 on every 4+ pt disagreement (${big.n.toLocaleString()} bets):
      <b class="${atsPl(big) > 0 ? "adv-model" : ""}">${money(atsPl(big))}</b> against the spread at \u2212110${
      big.mlN ? `, <b class="${big.mlPl > 0 ? "adv-model" : ""}">${money(big.mlPl)}</b> on the model's winner at the posted moneyline (${big.mlN.toLocaleString()} games with prices)` : ""}.` : "";

  $("results-body").innerHTML = `<div class="detail-in">
    <div class="tile-grid">${tiles.join("")}</div>

    <div class="sect"><h3>When the model disagrees with the line</h3>
      <span class="sect-note">record against the spread, by size of disagreement \u00b7 break-even is 52.4%</span></div>
    <div class="card" style="padding:0">
      <div class="results-table-scroll"><table class="layers edges">
        <thead><tr><th>Disagreement</th><th>ATS record</th><th>Cover rate</th>
          <th>$100/game ATS</th><th>$100/game moneyline</th><th></th></tr></thead>
        <tbody>${edgeRows}</tbody>
      </table></div>
      ${bigLine ? `<div class="results-foot">${bigLine}</div>` : ""}
    </div>

    ${seasons.length > 1 ? `
    <div class="sect"><h3>Margin error by season</h3>
      <span class="sect-note">mean absolute error, ${esc(scopeNote)}</span></div>
    <div class="card chart-card">
      <div class="chart-legend">
        <span class="legend-item"><span class="legend-swatch" style="background:var(--brand)"></span>Model</span>
        <span class="legend-item"><span class="legend-swatch" style="background:var(--vegas)"></span>Vegas closing spread</span>
      </div>
      ${maeChart(bySeason)}
    </div>` : ""}

    <div class="sect"><h3>${seasons.length > 1 ? "Season by season" : `${seasons[0]} detail`}</h3>
      <span class="sect-note">${esc(scopeNote)}</span></div>
    <div class="card results-table-card">
      <div class="results-table-scroll">
      <table class="results">
        <thead><tr><th>Season</th><th>Games</th><th>Winners</th><th>Model ATS</th>
          <th>Model MAE</th><th>Vegas MAE</th><th>Δ MAE</th><th>Closer</th></tr></thead>
        <tbody>${tableRows}</tbody>
      </table>
      </div>
      <div class="results-foot">Lines cover ${seasons[0]}–${seasons[seasons.length - 1]} —
        CFBD's closing-spread record starts in 2013. A season's games count only when a closing
        spread exists; the split filter decides whether the model had seen them in training.
        Green MAE marks the side that was closer on average.</div>
    </div>
  </div>`;
}

/* small two-series line chart: margin MAE by season */
function maeChart(bySeason) {
  const CW = 1000, CH = 300, CL = 46, CR = 18, CT = 18, CB = 34;
  const pw = CW - CL - CR, ph = CH - CT - CB;
  const xs = bySeason.map(r => r.season);
  const vals = bySeason.flatMap(r => [r.mMae, r.vMae]);
  const lo = Math.floor(Math.min(...vals) - 0.6), hi = Math.ceil(Math.max(...vals) + 0.6);
  const X = s => xs.length === 1 ? CL + pw / 2
    : CL + (s - xs[0]) / (xs[xs.length - 1] - xs[0]) * pw;
  const Yc = v => CT + (1 - (v - lo) / (hi - lo)) * ph;
  const ticks = [];
  for (let v = lo; v <= hi; v += (hi - lo > 6 ? 2 : 1)) ticks.push(v);
  const grid = ticks.map(v => `
    <line x1="${CL}" x2="${CW - CR}" y1="${Yc(v)}" y2="${Yc(v)}" stroke="var(--line)"/>
    <text x="${CL - 8}" y="${Yc(v) + 3.5}" text-anchor="end" fill="var(--ink-3)"
      font-family="Spline Sans Mono" font-size="9.5">${v}</text>`).join("");
  const xLabels = xs.filter((s, i) => xs.length <= 8 || i % 2 === 0 || i === xs.length - 1)
    .map(s => `<text x="${X(s)}" y="${CT + ph + 22}" text-anchor="middle"
      fill="var(--ink-3)" font-family="Spline Sans Mono" font-size="10">${String(s).slice(2)}</text>`).join("");
  const series = (key, color) => {
    const pts = bySeason.map(r => `${X(r.season).toFixed(1)},${Yc(r[key]).toFixed(1)}`).join(" ");
    const dots = bySeason.map(r => `<circle cx="${X(r.season).toFixed(1)}" cy="${Yc(r[key]).toFixed(1)}"
      r="3.4" fill="${color}" stroke="var(--bg-raised)" stroke-width="1.5"><title>${r.season}: ${r[key].toFixed(2)}</title></circle>`).join("");
    return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2"
      stroke-linejoin="round"/>${dots}`;
  };
  return `<svg viewBox="0 0 ${CW} ${CH}" role="img" aria-label="Margin MAE by season">
    ${grid}${xLabels}
    ${series("vMae", "var(--vegas)")}
    ${series("mMae", "var(--brand)")}
  </svg>`;
}

/* --------------------------------------------------------------- model page */
let modelCard = null;

async function loadModel() {
  if (modelCard) return;
  try {
    modelCard = await fetch("./model_card.json").then(r => {
      if (!r.ok) throw new Error(`model card missing (${r.status})`);
      return r.json();
    });
  } catch (e) {
    $("model-body").innerHTML = `<div class="rail-empty" style="padding:60px">
      ${esc(e.message)} — run <span class="num">model_card.py</span> to generate it.</div>`;
    return;
  }
  let exGame = null;
  try {
    [exGame] = await api(`games?select=home_team,away_team,home_points,away_points,home_team_id,away_team_id&game_id=eq.${modelCard.example.game_id}`);
  } catch (e) { /* the PMF renders without names */ }
  renderModel(modelCard, exGame);
}

const fmtK = n => n >= 1e6 ? `${(n / 1e6).toFixed(2)}M`
  : n >= 1e3 ? `${(n / 1e3).toFixed(n >= 100e3 ? 0 : 1)}k` : String(n);

function renderModel(c, exGame) {
  const facts = [
    ["parameters", fmtK(c.n_params)],
    ["raw inputs / prediction", `~${c.n_hist_dims + c.n_scalars + 64 + 15}`],
    ["scalar features", c.n_scalars],
    ["history slots", `2 × ${c.n_hist_games}`],
    ["live state features", c.n_state],
    ["ensemble seeds", c.seeds],
    ["margin bins", c.n_margin_bins],
  ].map(([k, v]) => `<span class="fact">${k} <b>${v}</b></span>`).join("");

  $("model-body").innerHTML = `<div class="detail-in">
    <div class="model-head">
      <h2 class="model-title">Inside the engine</h2>
      <p class="model-sub">One network, trained ${c.train_span[0]}–${c.train_span[1]}, judged on
      ${c.test_seasons.join(", ")}. It never sees a rating, a ranking, or a betting line —
      team strength is learned from raw results and counting stats, three ways at once.
      Every number it emits is a readout of a single predicted distribution over the final margin.</p>
      <div class="fact-row">${facts}</div>
    </div>

    <div class="sect"><h3>How a prediction is made</h3>
      <span class="sect-note">every box below is real — drawn from the checkpoints, not a sketch</span></div>
    <div class="card diagram-card">${architectureSvg(c)}
      <div class="diagram-cap">The pregame tower is trained first and then frozen; the in-game
      state tower sees only two noised scalars of it (expected margin and total) so it can never
      memorise a game. The mixing weight α starts near 1 at kickoff and decays to 0 as the clock
      dies — pregame belief hands over to what is actually happening on the field. For the
      ${c.seeds}-seed ensemble, all of this runs ${c.seeds} times and the distributions are averaged.</div>
    </div>

    <div class="sect"><h3>The inputs</h3>
      <span class="sect-note">${c.n_scalars} scalar features in ${c.inventory.length} groups, plus history and identity</span></div>
    <div class="inv-grid">
      ${c.inventory.map(g => `<div class="inv-card">
        <div class="inv-top"><span class="inv-name">${esc(g.group)}</span>
          <span class="inv-n">${g.n}</span></div>
        <div class="inv-desc">${esc(g.desc)}</div>
        ${g.s2d_gated ? `<span class="inv-gated">shrunk early in the season</span>` : ""}
      </div>`).join("")}
    </div>

    <div class="sect"><h3>One distribution, three readouts</h3>
      <span class="sect-note">${exGame ? `${esc(exGame.away_team)} at ${esc(exGame.home_team)}, ${c.example.season} — pregame, held out` : "a held-out game"}</span></div>
    <div class="card chart-card">
      <div class="pmf-legend">
        <span class="legend-item"><span class="legend-swatch" style="background:var(--brand)"></span>
          ${exGame ? esc(exGame.home_team) : "Home"} wins — ${pct(c.example.wp)} of the mass</span>
        <span class="legend-item"><span class="legend-swatch" style="background:var(--ink-3)"></span>
          ${exGame ? esc(exGame.away_team) : "Away"} wins — ${pct(1 - c.example.wp)}</span>
      </div>
      ${pmfSvg(c.example)}
      <div class="chart-foot">
        <span>Win probability = the shaded mass above zero — never a separate model</span>
        <span>${exGame && exGame.home_points != null ? `actual final margin ${signed(exGame.home_points - exGame.away_points, 0)}` : ""}</span>
      </div>
    </div>

    <div class="sect"><h3>What the network learned</h3>
      <span class="sect-note">gates and parameters, seed-averaged from the live checkpoints</span></div>
    <div class="layer-grid">
      <div class="card" style="padding:0">
        <div class="results-table-scroll"><table class="layers">
          <thead><tr><th>Component</th><th>Tensors</th><th>Parameters</th><th>Share</th></tr></thead>
          <tbody>${c.layers.map(l => `<tr>
            <td>${esc(l.component)}</td><td>${l.tensors}</td>
            <td>${l.n_params.toLocaleString()}</td>
            <td><span class="share-bar"><span class="share-fill"
              style="width:${(l.n_params / c.n_params * 100).toFixed(1)}%"></span></span></td>
          </tr>`).join("")}</tbody>
        </table></div>
      </div>
      <div style="display:flex; flex-direction:column; gap:14px">
        <div class="card chart-card">${gateAlphaSvg(c.gates)}
          <div class="curve-cap"><b>The live handover.</b> α weights pregame belief against the
          in-game read. Bigger leads pull trust toward the field sooner; by the final minute the
          pregame prior is nearly mute.</div>
        </div>
        <div class="card chart-card">${gateS2dSvg(c.gates)}
          <div class="curve-cap"><b>The early-season gate.</b> Season-to-date stats are multiplied
          by this learned curve — a 2-game yards-per-play is shrunk toward the league mean,
          a 10-game one is trusted almost fully.</div>
        </div>
      </div>
    </div>

    <div class="sect"><h3>Training & honesty</h3></div>
    <div class="tile-grid">
      <div class="tile"><div class="tile-label">Held-out log loss</div>
        <div class="tile-main"><span class="tile-value">${c.metrics.test.logloss.toFixed(4)}</span></div>
        <div class="tile-vs">cfbfastR published <span class="num">0.4384</span></div>
        <div class="tile-note">win probability, all snaps, FBS vs FBS</div></div>
      <div class="tile"><div class="tile-label">Margin RMSE vs market</div>
        <div class="tile-main"><span class="tile-value">${c.eval.pregame ? c.eval.pregame.model_rmse.toFixed(2) : "—"}</span></div>
        <div class="tile-vs">closing spread <span class="num">${c.eval.pregame ? c.eval.pregame.market_rmse.toFixed(2) : "—"}</span></div>
        <div class="tile-note">behind the market by design — beating it trips the leak alarm</div></div>
      <div class="tile"><div class="tile-label">Splits</div>
        <div class="tile-main"><span class="tile-value" style="font-size:16px; line-height:1.4">
          train ${c.train_span[0]}–${c.train_span[1]}<br>test ${c.test_seasons.join(", ")}</span></div>
        <div class="tile-note">validation is a game-id hash inside the training years — never seen by the optimiser, used only to stop it</div></div>
      <div class="tile"><div class="tile-label">Ensemble</div>
        <div class="tile-main"><span class="tile-value">${c.seeds}</span>
          <span class="tile-unit">seeds</span></div>
        <div class="tile-note">independently initialised copies, their margin distributions averaged — a mixture keeps every consistency property</div></div>
      <div class="tile"><div class="tile-label">Anti-memorisation</div>
        <div class="tile-main"><span class="tile-value" style="font-size:16px; line-height:1.4">identity dropout 30%<br>frozen pregame tower</span></div>
        <div class="tile-note">embeddings would happily become an outcome lookup table; these measures make that impossible</div></div>
    </div>
  </div>`;
}

/* the architecture, drawn to scale of the real checkpoint */
function architectureSvg(c) {
  const block = (x, y, w, h, title, lines = [], tag = null, accent = null) => `
    <rect class="dg-block" x="${x}" y="${y}" width="${w}" height="${h}" rx="9"
      ${accent ? `style="stroke:${accent}"` : ""}/>
    ${tag ? `<text class="dg-tag" x="${x + 12}" y="${y + 17}">${tag}</text>` : ""}
    <text class="dg-label" x="${x + 12}" y="${y + (tag ? 34 : 21)}">${title}</text>
    ${lines.map((l, i) => `<text class="dg-note" x="${x + 12}"
        y="${y + (tag ? 50 : 37) + i * 13}">${l}</text>`).join("")}`;
  const arrow = (x1, y1, x2, y2, cls = "dg-flow") =>
    `<path class="${cls}" d="M${x1} ${y1} C ${x1 + (x2 - x1) * .5} ${y1}, ${x1 + (x2 - x1) * .5} ${y2}, ${x2 - 7} ${y2}" marker-end="url(#dg-arr)"/>`;

  const nIn = c.n_hist_dims + c.n_scalars + 64 + 15;
  return `<svg viewBox="0 0 1120 640" role="img" aria-label="Model architecture">
    <defs><marker id="dg-arr" viewBox="0 0 8 8" refX="7" refY="4"
      markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0 0 L8 4 L0 8 z" fill="var(--ink-3)"/></marker></defs>

    <text class="dg-tag" x="16" y="16">INPUTS · ~${nIn} RAW NUMBERS</text>
    ${block(16, 28, 280, 62, "Identity", [
      `${c.n_teams} team + ${c.n_confs} conference embeddings`,
      "24-d / 8-d · identity dropout 0.30"])}
    ${block(16, 106, 280, 76, "Prior results, each side", [
      `${c.n_hist_games} games × score, venue, recency,`,
      "opponent's embedding — adjustment is learned"])}
    ${block(16, 198, 280, 92, `Team sheet — ${c.n_scalars} raw scalars`, [
      "pbp rates · box-score lines · roster · coach ·",
      "recruiting & talent · returning production",
      "season-to-date block × early-season gate"])}
    ${block(16, 470, 280, 76, `Live game state — ${c.n_state} features`, [
      "clock, lead, down & distance, possession,",
      "timeouts, momentum, points so far"], "DURING THE GAME")}

    <text class="dg-tag" x="400" y="16">ENCODE</text>
    ${block(400, 106, 230, 62, "History encoder", [
      "shared slot MLP → mean + max pool"])}
    ${block(400, 198, 230, 104, "Pregame tower", [
      `Linear ${c.pre_hidden ?? 160} → LayerNorm → GELU × 2`,
      "dropout 0.30 · early-stopped, then frozen"], null, "var(--brand)")}
    ${block(400, 452, 230, 94, "State tower", [
      "Linear 128 × 3 · pregame reaches it",
      "only as E[margin], E[total]"])}

    <text class="dg-tag" x="704" y="16">DISTRIBUTIONS</text>
    ${block(704, 178, 190, 66, "Pregame PMFs", [
      `margin · ${c.n_margin_bins} bins`, "total · 103 bins"])}
    ${block(704, 452, 190, 66, "Live PMFs", [
      `margin · ${c.n_margin_bins} bins`, "total · 103 bins"])}
    ${block(704, 306, 190, 80, "α(clock, |lead|)", [
      "α·pregame + (1−α)·live",
      "monotone gate: 1 at kickoff → 0 late"], "MIX")}

    <text class="dg-tag" x="948" y="16">READOUTS</text>
    ${block(948, 178, 156, 208, "One distribution", [
      "win probability =", "  mass above zero", "",
      "projected score =", "  median margin ∘", "  median total", "",
      "they cannot disagree"], null, "var(--brand)")}

    ${arrow(296, 59, 400, 229)}
    ${arrow(296, 144, 400, 137)}
    ${arrow(630, 137, 704, 211, "dg-flow")}
    ${arrow(296, 244, 400, 262)}
    ${arrow(630, 250, 704, 200, "dg-flow brand")}
    ${arrow(515, 302, 515, 452, "dg-flow") /* cond scalars */}
    <text class="dg-note" x="524" y="380">E[m], E[t] only</text>
    ${arrow(296, 508, 400, 499)}
    ${arrow(630, 499, 704, 485)}
    ${arrow(799, 244, 799, 306, "dg-flow brand")}
    ${arrow(799, 452, 799, 386, "dg-flow")}
    ${arrow(894, 346, 948, 282, "dg-flow brand")}
  </svg>`;
}

function pmfSvg(ex) {
  const W2 = 1000, H2 = 300, L2 = 40, R2 = 16, T2 = 24, B2 = 36;
  const pw = W2 - L2 - R2, ph = H2 - T2 - B2;
  const xs = ex.margin_centres, ps = ex.margin_pmf;
  const lo = -45, hi = 45;
  const pk = Math.max(...ps);
  const X2 = v => L2 + (v - lo) / (hi - lo) * pw;
  const Y2 = p => T2 + (1 - p / (pk * 1.1)) * ph;
  const bw = pw / (hi - lo) * 0.86;
  let bars = "";
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] < lo || xs[i] > hi) continue;
    const col = xs[i] > 0 ? "var(--brand)" : "var(--ink-3)";
    const op = xs[i] > 0 ? .9 : .45;
    bars += `<rect x="${(X2(xs[i]) - bw / 2).toFixed(1)}" y="${Y2(ps[i]).toFixed(1)}"
      width="${bw.toFixed(2)}" height="${(T2 + ph - Y2(ps[i])).toFixed(1)}"
      fill="${col}" opacity="${op}" rx="1"><title>margin ${xs[i] > 0 ? "+" : ""}${xs[i]}: ${(ps[i] * 100).toFixed(2)}%</title></rect>`;
  }
  const ticks = [-42, -28, -14, 0, 14, 28, 42].map(v => `
    <text x="${X2(v)}" y="${T2 + ph + 20}" text-anchor="middle" fill="var(--ink-3)"
      font-family="Spline Sans Mono" font-size="10">${v > 0 ? "+" + v : v}</text>`).join("");
  return `<svg viewBox="0 0 ${W2} ${H2}" role="img" aria-label="Predicted margin distribution">
    ${bars}
    <line x1="${X2(0)}" x2="${X2(0)}" y1="${T2 - 6}" y2="${T2 + ph}" stroke="var(--ink)" stroke-width="1.4"/>
    <text x="${X2(0)}" y="${T2 - 10}" text-anchor="middle" fill="var(--ink-2)"
      font-family="Archivo" font-weight="700" font-size="10" letter-spacing="1.5">EVEN</text>
    <line x1="${X2(ex.em)}" x2="${X2(ex.em)}" y1="${T2 + 14}" y2="${T2 + ph}"
      stroke="var(--vegas)" stroke-width="1.4" stroke-dasharray="4 3"/>
    <text x="${X2(ex.em) + 5}" y="${T2 + 26}" fill="var(--vegas)"
      font-family="Spline Sans Mono" font-size="10">E[margin] ${ex.em >= 0 ? "+" : ""}${ex.em.toFixed(1)}</text>
    <line x1="${L2}" x2="${W2 - R2}" y1="${T2 + ph}" y2="${T2 + ph}" stroke="var(--line-strong)"/>
    ${ticks}
    <text x="${W2 - R2}" y="${T2 + ph + 20}" text-anchor="end" fill="var(--ink-3)"
      font-family="Archivo" font-weight="600" font-size="9.5" letter-spacing="1">HOME MARGIN →</text>
  </svg>`;
}

function curveFrame(W3, H3, L3, R3, T3, B3, yTicks, xLabels) {
  const pw = W3 - L3 - R3, ph = H3 - T3 - B3;
  const Y3 = v => T3 + (1 - v) * ph;
  let s = yTicks.map(v => `
    <line x1="${L3}" x2="${W3 - R3}" y1="${Y3(v)}" y2="${Y3(v)}" stroke="var(--line)"/>
    <text x="${L3 - 7}" y="${Y3(v) + 3.5}" text-anchor="end" fill="var(--ink-3)"
      font-family="Spline Sans Mono" font-size="9.5">${v}</text>`).join("");
  s += xLabels.map(([fx, lab]) => `<text x="${L3 + fx * pw}" y="${H3 - 8}"
      text-anchor="middle" fill="var(--ink-3)" font-family="Spline Sans Mono"
      font-size="9.5">${lab}</text>`).join("");
  return { frame: s, X: f => L3 + f * pw, Y: Y3 };
}

function gateAlphaSvg(g) {
  const W3 = 500, H3 = 210, L3 = 34, R3 = 14, T3 = 16, B3 = 30;
  const { frame, X, Y } = curveFrame(W3, H3, L3, R3, T3, B3, [0, 0.5, 1],
    [[0, "kick"], [0.25, "Q2"], [0.5, "half"], [0.75, "Q4"], [1, "0:00"]]);
  const shades = { 0: 1, 7: .7, 14: .45, 21: .25 };
  const lines = Object.entries(g.alpha_by_lead).map(([lead, ys]) => {
    const pts = ys.map((y, i) => {
      const elapsed = 1 - g.frac_remaining[i];
      return `${X(elapsed).toFixed(1)},${Y(y).toFixed(1)}`;
    });
    pts.reverse();
    return `<polyline points="${pts.join(" ")}" fill="none" stroke="var(--brand)"
      stroke-width="1.8" opacity="${shades[lead] ?? .5}"/>
      <text x="${W3 - R3 - 4}" y="${Y(ys[0]) + (lead === "0" ? -5 : 11)}" text-anchor="end"
        fill="var(--brand)" opacity="${Math.max(shades[lead], .5)}"
        font-family="Spline Sans Mono" font-size="9">${lead === "0" ? "tie game" : `±${lead}`}</text>`;
  }).join("");
  return `<svg viewBox="0 0 ${W3} ${H3}" role="img" aria-label="Learned pregame weight by clock and lead">
    <text x="${L3}" y="11" fill="var(--ink-2)" font-family="Archivo" font-weight="800"
      font-size="9.5" letter-spacing="1.8">α — WEIGHT ON PREGAME BELIEF</text>
    ${frame}${lines}</svg>`;
}

function gateS2dSvg(g) {
  const W3 = 500, H3 = 190, L3 = 34, R3 = 14, T3 = 16, B3 = 30;
  const { frame, X, Y } = curveFrame(W3, H3, L3, R3, T3, B3, [0, 0.5, 1],
    g.gp.filter((_, i) => i % 2 === 0).map(v => [v / 12, String(v)]));
  const pts = g.s2d_gate.map((y, i) =>
    `${X(g.gp[i] / 12).toFixed(1)},${Y(y).toFixed(1)}`).join(" ");
  return `<svg viewBox="0 0 ${W3} ${H3}" role="img" aria-label="Learned trust in season-to-date stats by games played">
    <text x="${L3}" y="11" fill="var(--ink-2)" font-family="Archivo" font-weight="800"
      font-size="9.5" letter-spacing="1.8">TRUST IN SEASON-TO-DATE STATS · BY GAMES PLAYED</text>
    ${frame}
    <polyline points="${pts}" fill="none" stroke="var(--vegas)" stroke-width="2"/>
    ${g.s2d_gate.map((y, i) => `<circle cx="${X(g.gp[i] / 12).toFixed(1)}"
      cy="${Y(y).toFixed(1)}" r="2.8" fill="var(--vegas)"/>`).join("")}
  </svg>`;
}

boot();
route();
