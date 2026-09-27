/** Offline full-history scoring with the exact deployed state network.
 * Produces files only; publish_week_curves.py validates and commits each game. */
import { fetchPlayRows, stateFromSummary, parsePlayRows } from './functions/_shared/espn.ts';
import { deriveState } from './functions/_shared/features.ts';
import { prepareCurve, finalHistoryComplete } from './functions/_shared/curve_history.ts';
import { Net, readouts } from './functions/_shared/net.ts';
const [input, output] = Deno.args;
const games = JSON.parse(await Deno.readTextFile(input));
const weights = await Deno.readFile(new URL('./model/weights.bin', import.meta.url));
const net = new Net(weights.buffer, JSON.parse(await Deno.readTextFile(new URL('./model/weights.json', import.meta.url))));
for (const game of games) {
  const gid = game.game_id;
  try {
    const state = await stateFromSummary(gid);
    if (!state || state.state !== 'post') throw Error('No final summary');
    if (state.home_points !== game.home_points || state.away_points !== game.away_points) throw Error('Final score disagrees with database');
    let raw = await fetchPlayRows(state);
    // An archived raw CFBD feed can recover a game ESPN no longer serves.
    // The adapter contains only observed plays, scores and clock state.
    if (!raw.length && game.fallback_summary) raw = parsePlayRows(game.fallback_summary, state);
    const rows = prepareCurve(raw, state);
    if (!finalHistoryComplete(state, rows)) throw Error(`Missing or incomplete play feed: ${raw.length} plays`);
    const feats = deriveState(rows);
    const m = game.n_members;
    const pm = Array.from({length:m},(_,k)=>Float32Array.from(game.margin_logits.slice(k*115,(k+1)*115)));
    const pt = Array.from({length:m},(_,k)=>Float32Array.from(game.total_logits.slice(k*103,(k+1)*103)));
    const stamp = new Date().toISOString();
    const curve = rows.map((p,i)=>{
      const o = net.run(feats[i],p.t_left/3600,Math.abs(p.home_lead)/21,p.home_lead,p.total_pts_before,pm,pt);
      const ro = readouts(o.margin,o.total);
      return {game_id:gid,play_index:i,updated_at:stamp,t_left:p.t_left,period:p.period,clock:p.clock,
        down:p.down,distance:p.distance,situation:p.sit??'',play_text:p.play_text??'',home_points:p.hp,away_points:p.ap,
        home_win_prob:ro.wp,pred_margin:ro.em,pred_total:ro.et,total_pmf:Array.from(o.total,v=>Math.round(v*1e6)/1e6)};
    });
    await Deno.writeTextFile(`${output}/${gid}.json`,JSON.stringify({state,curve}));
    console.log(JSON.stringify({game_id:gid,rows:curve.length,status:'ready'}));
  } catch(e) { console.log(JSON.stringify({game_id:gid,status:'unavailable',error:String(e)})); }
}
