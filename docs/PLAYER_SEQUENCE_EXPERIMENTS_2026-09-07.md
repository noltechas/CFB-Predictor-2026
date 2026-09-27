# Individual game context, recent contribution and joint player-stat training

This experiment tests the three follow-ups to the corrected-identity player model. Its isolated artifacts are in `models/experiments/player_sequences_20260907_v2`. Production models and feature snapshots are unchanged.

## What is tested

| Variant | Change from the experiment control |
| --- | --- |
| Control | Corrected identities, individual recruiting information, full-career aggregate windows, and the existing raw team/game inputs |
| Context | Adds eight recent observed individual game boxes, with the player's school, opponent, prior score, venue, time elapsed, same-season/program flags and recency order |
| Contribution | Adds learned position-specific pooling based on the most recent, recent three, and preceding three observed game workloads and their missingness |
| Auxiliary | Jointly trains the player representation to predict 27 observed individual box statistics as well as game outcomes |
| Combined | Adds all three changes in one network |

Every variant keeps every roster member and all existing raw scalar inputs. The eight-game sequence supplements the previous-season, two-seasons-ago, older-career and current-season cumulative summaries; it does not replace them. Players are encoded without player-ID embeddings. Previous programs and opponents receive embeddings learned jointly with the game network, using only identities present in the fold's training histories.

The contribution branch retains the existing mean and maximum player pools and adds a weighted pool. Its inputs include passing attempts, carries, receptions, tackles, field-goal attempts, punts and returns. These are observed box-score measures, not full snap counts, receiving targets or definitive participation labels. The weights are internal learned contributions, not separately fitted player ratings or calibrated playing-time forecasts.

The auxiliary head predicts signed `log1p(abs(stat))/5` values, with Smooth L1 loss and explicit observation masks. The loss gives equal weight to each observed statistic within each game. Unreported player/stat combinations receive no auxiliary gradient. Its inputs are player representations and upcoming matchup identities/games played. Actual upcoming player outcomes are loaded only by the training loss, never by the prediction forward pass. The auxiliary head is disabled during ordinary inference.

## Data and validation

The immutable extension contains 959,005 corrected player-game records and 4,249,917 roster references, of which 2,085,008 have prior individual game history. Lossless compression reduces the new artifacts to about 32 MB. No existing player store is overwritten.

The complete reference audit found zero histories at or after their target kickoff, zero 2026 histories entering pre-2026 targets, and zero 2026 auxiliary outcomes entering pre-2026 targets. The label API also explicitly refuses 2026 player outcomes. Transfers retain earlier-program observations through the existing canonical identities. All events at a target kickoff are excluded, including simultaneous games.

Historical public coverage remains incomplete for injuries, starters, full snaps, offensive-line blocking and some lower-division production. A missing box row never means the player did not play. Annual roster membership and recruiting revisions still lack complete archived publication timestamps. This experiment does not claim to reconstruct information that public sources never recorded.

The implementation tests cover cross-program timelines, simultaneous-game exclusion, lossless data construction, missing-stat masks and gradients, refusal of 2026 labels, roster permutation, 200-player rosters, empty rosters, packed/padded equivalence, shared auxiliary gradients and team-order consistency. Packed computation removes padding work, not players. The CPU/GPU diagnostic before the final CPU configuration agreed within 0.000001 in logits.

The original execution was replaced with faster equivalent batching after a numerical audit: every input tensor matched bit-for-bit over 1,024 sampled games spanning 2001–2026. The optimized path computes recent workload summaries once per distinct history reference and avoids constructing padded event histories. Six completed members were retained from v1 with checkpoint/cache hashes in `reused_members.json`; incomplete runs were restarted. The network, optimizer, masks, seeds and selection rules are unchanged. Both execution versions and their frozen source snapshots remain available.

## Frozen evaluation

- Development: forecast 2020, 2021 and 2022, with two seeds per variant.
- Assessment: forecast 2023, 2024 and 2025, with three seeds per variant.
- For forecast year Y, train before Y-1, select epochs using year Y-1, then refit all labeled years before Y for that epoch count.
- Selection uses previous-year FBS-involved winner log loss. Normalization and known-identity mappings are fitted on training rows only.
- Training uses the original game-distribution loss plus auxiliary weight 1 only in the two auxiliary variants. AdamW uses learning rate 0.002 and weight decay 0.03, effective batches of 2,048 and microbatches of 512, up to 200 epochs with patience 20. FCS-only training extras are sampled at 25% per epoch.
- The lowest pooled development FBS-vs-FBS margin MAE identifies the development choice, recorded before assessment. All five variants are assessed regardless of their development result.
- Every comparison within a stage uses the same seeds, target games and ensemble size. Overall margin/total errors, probability quality, early weeks, annual results and paired week-cluster uncertainty are retained. There is no minimum-point cutoff for retaining a result.
- The 2023–2025 assessments have been examined in earlier experiments and are reused historical benchmarks. No 2026 outcome is used to fit or select these models.

The plan hashes source files, immutable input snapshots and both player-store manifests. Each member retains its selected epochs, train-game IDs, normalizers, known vocabularies, pregame checkpoint and full margin/total probability distributions. Results belong in `report.json`; source coverage and leakage checks are in `data_audit.json`.

## Reproduction

```sh
PYTHONPATH=src python3 -m cfbpred.player_sequence_inputs \
  --base data/parquet/player_inputs_v3 \
  --inputs models/experiments/players_20260906_v3/inputs \
  --output data/parquet/player_sequences_v1

PYTHONPATH=src python3 -m cfbpred.player_sequence_fast_experiment \
  --threads 2 --workers 3 --micro-batch 512

PYTHONPATH=src python3 -m cfbpred.player_sequence_forecast train \
  --threads 2 --workers 3 --micro-batch 512
PYTHONPATH=src python3 -m cfbpred.player_sequence_forecast score
PYTHONPATH=src python3 -m cfbpred.player_sequence_current_audit
```

The completed data builder refuses to overwrite its output. The experiment resumes complete cached members only under the identical frozen plan. The current production forecast and live model are separate from this pregame experiment.

## Results

Development selected **Game context** using the frozen pooled margin-MAE criterion. All 75 members completed and passed the checkpoint audit. The final repository check passed 105 tests.

The assessment below covers 2,398 FBS-vs-FBS games in 2023–2025. Every row uses three model seeds. Negative margin-error changes favor the candidate.

| Model | Margin MAE | Change vs control | 95% week-cluster interval | Total MAE | Winner log loss |
| --- | ---: | ---: | --- | ---: | ---: |
| Control | 12.6589 | +0.0000 | — | 12.9805 | 0.55169 |
| Game context | 12.6136 | -0.0454 | [-0.108, +0.016] | 12.9397 | 0.55030 |
| Contribution weighting | 12.6818 | +0.0229 | [-0.029, +0.077] | 13.0629 | 0.55272 |
| Auxiliary statistics | 12.7163 | +0.0574 | [-0.007, +0.121] | 13.0369 | 0.55383 |
| All three combined | 12.7142 | +0.0553 | [-0.000, +0.106] | 12.9663 | 0.55358 |

The closing-line benchmark has margin MAE **12.0296** on 2,396 matched games. Model comparisons with that benchmark must use each model's `market_matched.model_mae`, because two assessment games lack a closing line.

The intervals describe paired game-error changes for these fitted ensembles, clustered by season/week. They do not eliminate selection uncertainty from repeatedly examining the historical assessment years. Individual-seed metrics, early-week results, year-by-year results and comparisons with earlier retained ensembles are saved alongside the main report.

Every experiment is retained. The context change is also retained as a deployable candidate because it improved margin MAE, total MAE, winner log loss, Brier score and margin CRPS against the equal-size control on the assessment games. Its 0.045-point margin-MAE gain is small and its paired 95% interval crosses zero, so this is evidence for an incremental gain rather than a conclusive effect. Contribution weighting, auxiliary player-stat training and their combination remain reproducible experiment artifacts, but are excluded from the candidate because they increased assessment margin error by 0.023, 0.057 and 0.055 points respectively. Combining changes did not make their effects additive.

## Retained 2026 candidate

The retained context candidate is in `models/experiments/player_sequences_20260907_v2/current_2026_context_v2`. Three members were selected on 2025 and refitted through 2025 for 9, 13 and 8 epochs. The checkpoints contain 26,873 training games apiece and no 2026 outcomes.

At the common 2026-09-07T14:16:28.951443Z cutoff, `future_predictions.parquet` contains full margin and total distributions for all 1,402 remaining scheduled games. The independent audit passed all lineage, probability, readout and temporal checks. It verified 2,804 future team-game sides, including 1,501 FBS sides with no empty roster snapshots, and found no player-history observation at or after its target kickoff. Maximum probability-mass error is 0.0000002384 and all displayed predictions reproduce exactly from their saved distributions.

The candidate was promoted to the production pregame path on September 7, 2026 after a new snapshot was rebuilt from the active feature store and passed the same coverage, lineage, probability and temporal audits. All 1,402 future predictions and their full PMFs were published to the active Supabase database. A database readback matched every game ID and scoreline; maximum numeric differences were 0.000050 for prediction fields and 0.000000492 for PMF entries, within PostgreSQL `real` storage precision.

The active pointer is `models/player_sequence_production/active.json`; weekly refreshes rebuild player inputs and sequences before replacing only games that have not kicked off. The prior production files are retained under `models/player_sequence_production/promotions` for rollback. The validated raw-DNN state tower remains responsible for in-game updates because the player sequence tower is a pregame-only checkpoint.

The assessment context model's market-matched margin MAE is **12.6211**, versus **12.0296** for the closing line, leaving a **0.5914-point** average gap. The new player histories improve the model, but they do not yet rival the information set embedded in the closing market.
