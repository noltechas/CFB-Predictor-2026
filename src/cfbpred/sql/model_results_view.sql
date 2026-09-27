create or replace view public.v_model_vs_vegas as
        with per_game as (
          select p.season, p.split, p.model_version,
                 -- same split the slate rail filters on: FBS-involved vs
                 -- the lower-division-only slate
                 case when g.home_classification = 'fbs'
                        or g.away_classification = 'fbs'
                      then 'fbs' else 'fcs' end     as div_group,
                 g.margin::real                     as act_margin,
                 g.total::real                      as act_total,
                 p.pred_margin, p.pred_total, p.home_win_prob,
                 -l.spread                          as mkt_margin,
                 l.over_under,
                 l.home_moneyline                   as h_ml,
                 l.away_moneyline                   as a_ml,
                 case when l.home_moneyline is not null
                       and l.away_moneyline is not null then
                   (case when l.home_moneyline < 0
                         then -l.home_moneyline / (-l.home_moneyline + 100.0)
                         else 100.0 / (l.home_moneyline + 100.0) end)
                   /
                   ((case when l.home_moneyline < 0
                          then -l.home_moneyline / (-l.home_moneyline + 100.0)
                          else 100.0 / (l.home_moneyline + 100.0) end)
                    +
                    (case when l.away_moneyline < 0
                          then -l.away_moneyline / (-l.away_moneyline + 100.0)
                          else 100.0 / (l.away_moneyline + 100.0) end))
                 end                                as mkt_wp
          from public.game_predictions p
          join public.games g using (game_id)
          join public.game_lines l using (game_id)
          where g.home_points is not null and g.away_points is not null
            and l.spread is not null
        ),
        graded as (
          select *,
                 abs(pred_margin - mkt_margin)                as edge,
                 (act_margin <> mkt_margin)                   as is_graded,
                 ((pred_margin > mkt_margin)
                   = (act_margin > mkt_margin))               as ats_hit,
                 -- $100 on the model's predicted WINNER at the posted
                 -- moneyline: actual American odds, no vig assumption needed
                 case
                   when h_ml is null or a_ml is null then null
                   when home_win_prob > 0.5 then
                     case when act_margin > 0
                          then case when h_ml < 0 then 10000.0 / (-h_ml)
                                    else h_ml::real end
                          else -100.0 end
                   else
                     case when act_margin < 0
                          then case when a_ml < 0 then 10000.0 / (-a_ml)
                                    else a_ml::real end
                          else -100.0 end
                 end                                          as ml_pl
          from per_game
        )
        select season, split, div_group,
          count(*)::int                                        as n_games,
          -- ATS by size of model-vs-line disagreement ("edge")
          count(*) filter (where is_graded and edge < 2)::int             as ats_e0_n,
          count(*) filter (where is_graded and edge < 2 and ats_hit)::int as ats_e0_w,
          count(*) filter (where is_graded and edge >= 2 and edge < 4)::int             as ats_e2_n,
          count(*) filter (where is_graded and edge >= 2 and edge < 4 and ats_hit)::int as ats_e2_w,
          count(*) filter (where is_graded and edge >= 4 and edge < 7)::int             as ats_e4_n,
          count(*) filter (where is_graded and edge >= 4 and edge < 7 and ats_hit)::int as ats_e4_w,
          count(*) filter (where is_graded and edge >= 7)::int             as ats_e7_n,
          count(*) filter (where is_graded and edge >= 7 and ats_hit)::int as ats_e7_w,
          -- moneyline P/L on the model's winner, bucketed by the same
          -- spread disagreement ("how much do the loud games pay?")
          count(ml_pl) filter (where edge < 2)::int                as ml_e0_n,
          coalesce(sum(ml_pl) filter (where edge < 2), 0)::real    as ml_e0_pl,
          count(ml_pl) filter (where edge >= 2 and edge < 4)::int  as ml_e2_n,
          coalesce(sum(ml_pl) filter (where edge >= 2 and edge < 4), 0)::real as ml_e2_pl,
          count(ml_pl) filter (where edge >= 4 and edge < 7)::int  as ml_e4_n,
          coalesce(sum(ml_pl) filter (where edge >= 4 and edge < 7), 0)::real as ml_e4_pl,
          count(ml_pl) filter (where edge >= 7)::int               as ml_e7_n,
          coalesce(sum(ml_pl) filter (where edge >= 7), 0)::real   as ml_e7_pl,
          avg(abs(pred_margin - act_margin))                   as model_margin_mae,
          avg(abs(mkt_margin - act_margin))                    as vegas_margin_mae,
          sqrt(avg((pred_margin - act_margin) ^ 2))            as model_margin_rmse,
          sqrt(avg((mkt_margin - act_margin) ^ 2))             as vegas_margin_rmse,
          count(*) filter (where abs(pred_margin - act_margin)
                               < abs(mkt_margin - act_margin))::int as model_closer_margin,
          count(*) filter (where (home_win_prob > 0.5) = (act_margin > 0))::int as su_wins,
          -- the market's outright call on the same games: its favorite is
          -- whichever side the spread favours, judged the same way
          count(*) filter (where (mkt_margin > 0) = (act_margin > 0))::int as su_wins_vegas,
          count(*) filter (where act_margin <> mkt_margin
                             and (pred_margin > mkt_margin)
                               = (act_margin > mkt_margin))::int    as ats_wins,
          count(*) filter (where act_margin <> mkt_margin)::int     as ats_graded,
          count(over_under)::int                               as n_totals,
          avg(abs(pred_total - act_total))
            filter (where over_under is not null)              as model_total_mae,
          avg(abs(over_under - act_total))                     as vegas_total_mae,
          count(*) filter (where over_under is not null
                             and abs(pred_total - act_total)
                               < abs(over_under - act_total))::int  as model_closer_total,
          count(mkt_wp)::int                                   as n_ml,
          avg((home_win_prob - (act_margin > 0)::int) ^ 2)
            filter (where mkt_wp is not null)                  as model_brier,
          avg((mkt_wp - (act_margin > 0)::int) ^ 2)            as vegas_brier,
          model_version
        from graded
        group by season, split, model_version, div_group
