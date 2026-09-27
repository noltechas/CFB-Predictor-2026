import { scoringWork } from "./scorer_work.ts";
const state = (game_id: number, rest = {}) => ({ game_id, state: "post", home_points: 21, away_points: 10, ...rest });
Deno.test("only unchanged completed curves are skipped", () => {
  const states = [state(1), state(2), state(3), state(4), state(5, { state: "in" })];
  const saved = [
    { ...state(1), curve_done: true },
    { ...state(2), curve_done: false },
    { ...state(3, { home_points: 14 }), curve_done: true },
    { ...state(5), curve_done: true },
  ];
  const ids = scoringWork(states, saved).map(x => x.game_id);
  if (JSON.stringify(ids) !== "[2,3,4,5]") throw new Error(`wrong work set: ${ids}`);
});
Deno.test("missing state never silently suppresses a final", () => {
  if (scoringWork([state(1)], []).length !== 1) throw new Error("missing final suppressed");
});

Deno.test("final backlog is bounded and drains without dropping live games", () => {
  const finals = Array.from({length:40}, (_,i) => state(i+1));
  const live = state(100, {state:"in"});
  const first = scoringWork([...finals, live], []);
  if (first.length !== 17 || first.at(-1)?.game_id !== 100) throw new Error("live game lost or backlog unbounded");
  const saved = first.filter(s=>s.state==="post").map(s=>({...s,curve_done:true}));
  const next = scoringWork([...finals, live], saved);
  if (next[0].game_id !== 17 || next.length !== 17) throw new Error("backlog did not advance");
});
Deno.test('unsupported games cannot consume the final backlog slots',()=>{
 const games=Array.from({length:40},(_,i)=>state(i+1));
 const selected=scoringWork(games,[],new Set([39,40]));
 if(JSON.stringify(selected.map(s=>s.game_id))!=='[39,40]')throw Error('unsupported games starved supported games');
});
Deno.test('missing feeds rotate behind games not yet attempted',()=>{
 const games=Array.from({length:40},(_,i)=>state(i+1));
 const saved=games.slice(0,16).map(s=>({...s,curve_done:false,updated_at:'2026-09-14T04:00:00Z'}));
 if(scoringWork(games,saved)[0].game_id!==17)throw Error('missing feeds monopolized work slots');
});
