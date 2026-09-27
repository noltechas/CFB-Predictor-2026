import { prepareCurve, finalHistoryComplete, resumeCurve } from './curve_history.ts';
import { synthNowRow, LiveState } from './espn.ts';
const state: LiveState = {game_id:1,state:'in',period:3,clock:'2:00',home_points:21,away_points:10,home_id:'1',away_id:'2',home_abbr:'A',away_abbr:'B',down:null,distance:null,possession:null,possession_text:'',home_to:null,away_to:null};
const assert=(ok:boolean,message:string)=>{if(!ok)throw Error(message);};
Deno.test('catching up during the third quarter cannot mark a game complete',()=>{
 const rows=prepareCurve([synthNowRow(state,1,0.5,0)],state);
 assert(!finalHistoryComplete(state,rows),'live history marked complete');
 const final={...state,state:'post',period:4,clock:'0:00'};
 assert(!finalHistoryComplete(final,prepareCurve([synthNowRow(state,1,0.5,0)],final)),'truncated third-quarter feed marked complete');
});
Deno.test('final history includes the observed score after the last play',()=>{
 const final={...state,state:'post',period:4,clock:'0:00',home_points:28};
 const rows=prepareCurve([synthNowRow({...state,period:4,clock:'0:01'},1,0.5,0)],final);
 assert(rows.length===2 && rows[1].hp===28 && rows[1].t_left===0,'missing final score');
 assert(finalHistoryComplete(final,rows),'final history rejected');
 assert(!finalHistoryComplete(final,[]),'empty history completed');
});
Deno.test('snap budgets resume long games and replace temporary live points',()=>{
 const next=resumeCurve({max:119,n:120},220,false,false);
 assert(!next.full && next.from===120,'long game restarted');
 const live=resumeCurve({max:119,n:120},160,false,true);
 assert(!live.full && live.from===119,'temporary point not replaced');
 assert(resumeCurve({max:219,n:120},220,false,false).full,'holes mistaken for full curve');
});
Deno.test('final summary cannot erase overtime recorded in the play feed',()=>{
 const final={...state,state:'post',period:4,clock:'0:00'};
 const play={...synthNowRow({...state,period:5,clock:'0:00'},30,0.5,0),period:6};
 const rows=prepareCurve([play],final);
 assert(rows.at(-1)!.period===6,'overtime period lost');
});
