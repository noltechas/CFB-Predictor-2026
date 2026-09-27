# Fieldcast rankings

A neutral-field readout of the active production pregame ensemble. It produces one comparable team rating without retraining or feeding derived strength values back into the prediction model. The current implementation uses the three-checkpoint `player-sequence-tabm16-v2` ensemble; each checkpoint averages 16 internal TabM members.

## Rating

Every eligible Division I team faces the same current-season FBS reference pool. Each matchup uses a neutral site, seven days' rest and 250 miles of travel for both teams, zero altitude and body-clock change, and no conference-game flag. The week context is the latest completed season week plus one (capped at 22). Method `neutral-fbs-pool-v2` explicitly normalizes body-clock context as well as the original rest/travel/site controls.

For model member `k`, let `m_k(i,j)` be its expected point margin when team `i` occupies the home input and team `j` occupies the away input. Compute the margin from the model's softmax margin distribution, then remove input-order dependence:

```
neutral_k(i,j) = (m_k(i,j) - m_k(j,i)) / 2
neutral_k(i,i) = 0
rating(i) = mean_k(mean_j_in_FBS(neutral_k(i,j)))
```

Both input orders are evaluated for every required pair. FBS self-comparisons contribute zero, keeping the exact same denominator and opponent pool for every team. The mean FBS rating is zero by construction. A rating of +10 means an average projected ten-point margin against this pool under these conditions. A difference between two ratings is a summary of common-opponent performance, not a replacement for the model's direct matchup prediction. The active method is `neutral-fbs-pool-player-tabm-v1`.

All/FBS/FCS filters retain identical point values. All shows overall positions; FBS and FCS show division positions. Selecting a conference switches the # column to positions within that conference. Conference positions are computed before search or followed-team filters, so hiding other teams does not renumber the remaining rows. Unrated teams remain unranked. Clearing the conference restores overall or division positions. Exact rating ties use the school name for stable ordering; displayed values round to one decimal.

**+/− is the change in rating points, not rank position:** `current.rating − previous.rating`, computed at full precision and then rounded to one decimal. Week 1 compares with preseason (week 0); Week 2 compares with Week 1, and so on. Every filter preserves the same change. An unchanged rating displays `0.0`; a missing, unranked, or incompatible baseline displays a dash. A team's rating can increase while its position falls.

## Inputs and coverage

- Current-season schedule classifications determine FBS/FCS membership, including teams transitioning divisions. Provider team IDs are required and must be unique.
- The latest isolated production refresh supplies the same corrected scalar state used for current game predictions, including season box totals, participation, career inputs, quarterback priors, play-by-play rates, and recent form. Opponent-dependent recent-form columns are rebuilt for each synthetic matchup rather than copied from a scheduled opponent.
- Results and the last 14 games of team history are rebuilt at one cutoff. Every ranked team receives its complete current player-roster encoding, including dated scouting, career history, transfers, and recent individual-game sequences.
- Current-season results require kickoff before the cutoff, two scores, and the provider's explicit completed flag. Partial live boxes and future results are excluded. Each checkpoint supplies its original normalization, scalar ordering, embeddings, and parameters.
- Known team identities, available season priors, and at least four historical games are required. Teams below that minimum remain visible with no rating. Missing stat blocks use the model's existing missingness flags; no invented FCS stats are inserted.
- The entire FBS pool must have coverage or generation fails. FCS may have thinner box-score, roster, and play-by-play coverage. Team details expose history count, current-season box count, and roster availability.
- The Week 1 snapshot contains 138 FBS and 128 FCS teams: 265 rated and one team unranked for insufficient history. All 138 FBS teams have complete current-roster snapshots.

## Snapshots and publication

```
python3 src/cfbpred/rankings.py --season 2026
python3 src/cfbpred/rankings.py --season 2026 --publish
python3 src/cfbpred/rankings.py --season 2026 --preseason --publish
python3 src/cfbpred/rankings.py --publish-file data/parquet/rankings/latest.json
python3 src/cfbpred/rankings.py --publish-file data/parquet/rankings/2026-preseason.json --replace-preseason
```

Publishing uses the repository's existing database environment variables. The CLI writes `latest.json` and a `{season}-week-{ranking_week}.json` archive in `data/parquet/rankings/`. The explicit preseason command writes `{season}-preseason.json` and week 0 without replacing `latest.json`. The loaded `com.cfbpred.refresh` launchd job runs `refresh_week.sh` daily at 4:30 AM local time. Rankings run after the source, feature, roster, prediction, line, and pregame-logit refreshes under the pipeline's shared lock. Before any games, normal generation publishes week 0; after games begin, it publishes the latest completed source football week.

The migration `supabase/migrations/20260906204118_fieldcast_rankings.sql` creates `public.fieldcast_rankings`. Migration `20260906211750_rankings_weekly_rating_change.sql` keys it by `(season, ranking_week)` and adds `snapshot_kind`. One row contains the complete team array and metadata; a transaction atomically replaces the current football week. Older jobs cannot overwrite a newer cutoff. Anonymous and authenticated clients can only select; RLS is enabled. Rankings are separate from the training and prediction tables.

`ranking_week` is the source continuous **completed-results** football week, with 0 reserved for preseason. Refreshes replace the current result week's snapshot; later weeks get new rows even when they share a Sunday-based `period_start`. That date remains informational. Comparisons require the immediately preceding result week, matching season, checkpoint fingerprint, methodology version, and FBS opponent-pool hash. Missing weeks cannot silently become multi-week changes.

The client loads the latest two snapshots by default and lists saved week metadata in the week dropdown. Historical requests fetch the exact selected result week and its immediately preceding week. Each selection is validated and cached separately for 15 minutes. Pull-to-refresh bypasses that cache. A failed request retains a previously loaded snapshot for that same selection with a visible error. A missing selected snapshot cannot fall back to an older edition. Global score badges and slate sorting continue using the latest snapshot.

The September 8 refresh failed at step 6 because the direct script entry point lacked the package import path. That entry point is fixed, with a regression check that invokes `--help` without `PYTHONPATH`. A manual recovery then ran out of local disk space before it could publish; its incomplete isolated output was removed and the previous published snapshot was retained. The player refresh now checks for at least 2 GiB free before allocating a new snapshot. Free that space and rerun `./refresh_week.sh` to refresh inputs and publish current rankings through the normal shared-lock pipeline.

General historical backfills are disabled: today's feature store is not a point-in-time archive. The explicit `--preseason` path allows a labeled reconstruction: it uses the first Division I kickoff as the historical cutoff, excludes all current-season team and player results, and removes current-season box, participation, play-by-play, quarterback, and recent-form state. Team and individual history come only from before the global cutoff. It retains today's saved season-level roster, recruiting, returning-production inputs and frozen model; those source revisions are not guaranteed to match what was available before kickoff. A reconstruction is therefore not an archived preseason forecast and must not be used as a point-in-time backtest. `as_of` records the reconstructed cutoff, `generated_at` the actual generation time, and `snapshot_kind='reconstructed_preseason'` plus `diagnostics.backfilled=true` identify it. The methodology panel discloses this when it is the active baseline.

The 2026 preseason baseline was reconstructed this way at an August 27, 22:00 UTC cutoff, using no 2026-season results. Published reconstructions are frozen by default. The publisher will not silently replace an archived or reconstructed preseason baseline; `--replace-preseason` is an explicit administrative override and only accepts a Week 0 document. Genuine preseason snapshots can refresh before kickoff, then remain the season's Week 1 reference.

Training checkpoints are fingerprinted from their bytes. Vocabulary dimensions and scalar schema must match the frozen model; checkpoint dimensions alone cannot detect a same-sized reordered vocabulary, so production refreshes must retain the trained ID mapping.

## Validation and limits

The automated checks cover common-pool centering, invariance to team ordering, cancellation of symmetric input-slot bias, completed-game cutoff boundaries, raw neutral context, current scalar-side extraction, reconstruction of opponent-dependent recent form, latest box-score inclusion for idle teams, preseason result/stat exclusion, division/filter semantics, Week 1 versus preseason, consecutive football-week comparisons, missing weeks, rank-versus-rating distinctions, malformed snapshots, and signed/zero/missing formatting.

These checks establish calculation and application behavior. They do **not** establish future predictive accuracy for this derived ranking. There is no additional fit or calibration for hypothetical neutral matchups.

The active architecture enforces team-order consistency. The Week 1 production snapshot's mean absolute opposite-order error is approximately 0.0000013 points (95th percentile 0.0000057), which is floating-point noise. The stored `member_sd` measures agreement among the three outer checkpoints, not a validated uncertainty interval, and is not presented as confidence in the app.

Before accuracy claims or downstream season simulations: collect immutable input/prediction cutoffs, evaluate rating differences against future results and simple baselines, report FBS and FCS separately, assess early-season stability and neutral-site performance, and investigate the underlying input-order asymmetry. Keep the app's experimental designation until that work supports a stronger claim.
