# Week 2 publishing repair — September 14, 2026 UTC

Chas's 30-team screenshot was published as editorial Week 3 in left-to-right order. Editorial Week 2 remains archived. Fieldcast uses completed-results weeks, so its new model snapshot is Week 2.

The nightly refresh previously stopped at the database outage. This recovery refreshed scores, raw plays, team features, player histories and future forecasts with the existing frozen three-checkpoint player TabM ensemble. The post–Week 2 ranking snapshot was then generated and published.

All 131 Week 2 games with published predictions now have full histories: 23,270 points with contiguous indices and an observed final-score endpoint. The replays used the pregame logits already locked before kickoff and the deployed frozen state network. ESPN supplied 130 histories. Alcorn State–Arkansas-Pine Bluff's 213 archived CFBD plays supplied the missing overtime history, plus its observed final state. No provider win probabilities were used. Each replacement was validated and committed transactionally; original graphs were retained locally for rollback.

The deployed scorer now:

- Marks a curve complete only for a finished game with a final-score endpoint.
- Replaces temporary live endpoints and resumes partially written histories without repeatedly erasing their prefix.
- Filters unsupported games before applying its work limit and rotates unfinished games by their last attempt, so unavailable feeds cannot starve the slate.
- Checks completion/write errors and detects gaps before resuming a history.

The current provider vocabulary removed Madonna University (Mich.), shifting later team indices. The legacy raw-model logits publisher now remaps refreshed team/conference IDs to its preserved pre-refresh vocabulary instead of changing embedding sizes or identities. The retained vocabulary is `models/raw_dnn_vocab.json`, copied from the September 6 feature-audit backup, SHA-256 `348e6c6903d6835703da38802ccd693fcfd1c30d3e5411c6a7baa21145598440`. The existing 16 checkpoints remain unchanged. Upcoming logits were successfully published; started-game logits remain locked. Unfinished histories retain their logits after falling outside the normal rolling window.

Verification artifacts are in `logs/score_repair/week2_final_verification.json`, `week2_curve_publish.log`, `week2_rankings_publish.log`, and `week2_logits_publish.log`. Tests cover premature live completion, truncated feeds, final endpoints, partial-budget resumes, gaps, queue starvation, and embedding identity after vocabulary changes. The deployed model-load and dry-run checks passed; scheduled cron runs succeeded.

For a future historical repair, export the game's locked inputs, use `supabase/replay_week.ts` to generate local files, and validate with `python3 src/cfbpred/publish_week_curves.py DIRECTORY`. Add `--publish` only for the intended replay files; `--game ID` selects one game. Coordinate with the scheduled scorer during bulk publication. These tools never retrain a model.
