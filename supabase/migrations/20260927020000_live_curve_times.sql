-- Per-point timing of stored live curves, so the scorer can re-price points a
-- later parse has re-timed (a bad ESPN clock stamp used to freeze a point in
-- the wrong quarter for good). One row per game keeps it under PostgREST's
-- 1000-row cap; arrays are in play_index order.
create or replace function public.live_curve_times(ids bigint[])
 returns table(game_id bigint, t_left real[], period smallint[],
               home_points smallint[], away_points smallint[])
 language sql
 stable
 set search_path to ''
as $function$
  select c.game_id,
         array_agg(c.t_left order by c.play_index),
         array_agg(c.period order by c.play_index),
         array_agg(c.home_points order by c.play_index),
         array_agg(c.away_points order by c.play_index)
    from public.live_curve c where c.game_id = any(ids) group by c.game_id
$function$;
revoke all on function public.live_curve_times(bigint[]) from public, anon, authenticated;
grant execute on function public.live_curve_times(bigint[]) to service_role;
notify pgrst, 'reload schema';
