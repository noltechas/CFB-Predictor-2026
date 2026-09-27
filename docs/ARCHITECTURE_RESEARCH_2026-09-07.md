# Pregame architecture research and TabM evaluation

This study asks which neural architecture best combines the project's full pregame information: 270 raw game and team scalars, fourteen chronological team-history slots, every roster member's recruiting and biographical profile, full-career aggregates, and eight strictly prior individual game records. It changes the learner, not the data definition. Betting lines, external ratings and 2026 outcomes remain excluded.

## Research conclusion

The best tested design is the existing complete player encoder followed by a **TabM-style, parameter-efficient ensemble MLP** at the game level. Each outer seed contains sixteen related learners. They share the main linear weights and use learned rank-one adapters plus independent output heads. Training applies the proper categorical loss to every internal learner; inference averages their margin and total probabilities. Production retains three independently initialized outer seeds and averages those probabilities again.

This choice follows the strongest recent evidence for medium-sized, engineered tabular data. [TabM (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html) reports that a parameter-efficient ensemble of MLPs outperformed prior tabular deep-learning models across its evaluation. The [official implementation](https://github.com/yandex-research/tabm) trains the internal predictions independently and averages them for inference; this implementation follows that rule and uses its conservative adapter initialization. [TabReD (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/571799482291411607c54984153190b0-Abstract-Conference.html) is particularly relevant because it evaluates realistic temporal splits and rich feature sets, under which simple MLP-like architectures and gradient-boosted trees were strongest. The project's pure-neural constraint makes TabM the better fit for that evidence.

The study also tested architectures motivated by other primary research:

| Candidate | Motivation | Result |
| --- | --- | --- |
| Residual MLP | [Revisiting Deep Learning Models for Tabular Data](https://papers.nips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html) found tuned ResNets and FT-Transformers to be strong baselines | Did not improve pooled development margin MAE |
| History attention | [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) and [Deep Sets](https://papers.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html) support permutation-aware or invariant set encoders | Small screen result; did not advance |
| Position-unit cross-attention | Attention between each offense/defense position group and its opponent | Worse screen margin MAE |
| TabM plus both attention branches | Tests whether TabM and structured attention combine | Improved development, but less than TabM alone |
| TabM-16 | [TabM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html), building on [BatchEnsemble](https://openreview.net/pdf?id=Sklf1yrYDr) | Selected and passed assessment |

[TabR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html) was considered but not selected for this round. Retrieval would require a changing historical index with additional temporal and identity leakage controls, while the present data already contains explicit chronological history. [TabPFN](https://doi.org/10.1038/s41586-024-08328-6) was also considered. Its externally pretrained prior and flat-table operating assumptions do not fit the project's in-house pure-neural policy or its ragged player histories as directly as TabM.

There is no defensible way to guarantee an absolute best architecture from a finite historical sample. The selected network is the best architecture tested under the project's current constraints, frozen chronology and complete feature set.

## Frozen protocol

- Screen every candidate on 2022 with seeds 0 and 1.
- Advance the three lowest margin-MAE candidates whose winner log loss is no more than 0.010 above control.
- Select on pooled 2020–2022 forecasts with seeds 0 and 1. For forecast year Y, fit before Y-1, choose epochs on Y-1, refit through Y-1 and predict Y.
- Require development winner log loss no more than 0.005 above control, Brier score no more than 0.002 above control, total MAE no more than 0.10 above control, and lower margin MAE.
- Save the choice before reading the reused 2023–2025 assessment.
- Assess only the selected architecture on 2023–2025 with seeds 0, 1 and 2.
- Fit normalization and known team/conference identities from each training fold only. Include every roster member and only histories strictly before kickoff. Preserve exact home/away swap symmetry.

The experiment plan hashes the input snapshot, both player manifests, control predictions and source files. Full checkpoints, PMFs, epoch logs and paired errors are retained in `models/experiments/architecture_research_20260907_v1`.

## Results

Development selection used 1,974 FBS-vs-FBS games from 2020–2022:

| Model | Margin MAE | Change vs control | 95% week-cluster interval | Winner log loss | Brier | Total MAE |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Existing context control | 13.0463 | — | — | 0.54707 | 0.18514 | 12.9753 |
| Residual MLP | 13.1420 | +0.0957 | [-0.049, +0.238] | 0.54935 | 0.18624 | 13.1257 |
| Full attention hybrid | 12.8884 | -0.1580 | [-0.266, -0.058] | 0.54179 | 0.18325 | 12.9916 |
| **TabM-16** | **12.8566** | **-0.1897** | **[-0.292, -0.089]** | **0.53941** | **0.18234** | 13.0175 |

The frozen selector chose TabM-16. On the 2,398-game 2023–2025 assessment:

| Metric | Context control | TabM-16 | Change |
| --- | ---: | ---: | ---: |
| Margin MAE | 12.6136 | **12.5005** | **-0.1130** |
| Margin RMSE | 15.9636 | **15.8127** | -0.1509 |
| Median margin MAE | 12.6076 | **12.4842** | -0.1234 |
| Total MAE | 12.9397 | **12.8873** | -0.0525 |
| Total RMSE | 16.1386 | **16.0714** | -0.0672 |
| Winner log loss | 0.55030 | **0.54149** | -0.00882 |
| Brier score | 0.18643 | **0.18345** | -0.00298 |
| Margin CRPS | 8.9831 | **8.8858** | -0.0972 |
| 10-bin ECE | 0.02668 | **0.01961** | -0.00708 |

The paired margin-MAE change is **-0.1130 points**, with a season/week-cluster bootstrap 95% interval of **[-0.2048, -0.0284]**. The interval excludes zero. TabM improved 2024 by 0.2881 points and 2025 by 0.0753; it was 0.0249 worse in 2023. Probability log loss improved in all three years.

Early weeks through week 3 improved more: margin MAE fell from 13.1559 to 12.9385, winner log loss from 0.49845 to 0.48027, Brier score from 0.16720 to 0.16074, and CRPS from 9.3264 to 9.1462 across 438 games. This matters because the architecture is extracting more value from the complete recruiting, transfer and prior-player history when current-season team observations are scarce.

On the 2,396 assessment games with closing lines, market margin MAE is 12.0296. The context model's matched MAE is 12.6211 and TabM's is 12.5079. The remaining gap therefore falls from 0.5914 to 0.4782 points, a **19.1% reduction**. The model still does not equal the closing market.

## Deployment candidate

The production configuration is three outer seeds, sixteen internal TabM members per seed, two 192-wide ReLU blocks, dropout 0.10, AdamW learning rate 0.002 and weight decay 0.0003. Each seed chooses its epoch count on 2025 and is then refit on all 26,873 labeled games through 2025. The outputs remain separate categorical margin and total distributions, preserving calibrated probabilities and the existing score readouts.

The weekly pipeline rebuilds the complete player/game snapshot before scoring. The architecture consumes all existing scalar, team-history, roster, recruiting, career and individual-game sequence features for every future game. It remains a pregame model; the validated raw state network continues to handle after-kickoff updates.

The three refitted checkpoints selected 14, 14 and 27 epochs and each contains 26,873 training games through 2025. The candidate originally scored all 1,402 future 2026 games in the current snapshot. Its independent audit verified all 2,804 team sides, nonempty player inputs for all 1,501 FBS sides, valid probability distributions, exact pre-kickoff history boundaries and no 2026 fitting labels.

After explicit approval, `player-sequence-tabm16-v2` was activated and used to reconstruct all 1,610 modeled 2026 games, including completed games. The full-season audit verified 3,220 team sides, 1,649 FBS sides, complete coverage, valid distributions and strictly pre-kickoff player histories. All 1,610 prediction and PMF rows were published to Supabase. Database readback matched every game ID and scoreline, with maximum prediction-field error 0.000050 and maximum PMF error 0.000000499 from PostgreSQL `real` precision. The previous local and database rows are archived for rollback.

Across the 208 completed Week 1 games in the model population, margin MAE improved from 16.6025 to 16.5844, winner accuracy from 76.92% to 78.85%, winner log loss from 0.45398 to 0.44355, and Brier score from 0.14952 to 0.14600. Total MAE increased from 12.8056 to 13.1315. On the core 50 FBS-vs-FBS games, margin MAE improved from 16.1149 to 15.4379, winner accuracy from 80% to 84%, log loss from 0.42619 to 0.39703, and Brier score from 0.13046 to 0.11945; total MAE increased from 12.1355 to 12.5436.

These Week 1 numbers are a retrospective reconstruction rather than an original timestamped forecast. Model weights use labels only through 2025 and every dated team/player history is cut before kickoff, but annual roster membership and recruiting revisions lack complete archived publication timestamps.

## Reproduction

```sh
PYTHONPATH=src python3 -m cfbpred.architecture_research_experiment \
  --workers 3 --threads 2 --micro-batch 256

PYTHONPATH=src python3 -m cfbpred.architecture_research_forecast train \
  --workers 3 --threads 2 --micro-batch 256
PYTHONPATH=src python3 -m cfbpred.architecture_research_forecast score
PYTHONPATH=src python3 -m cfbpred.architecture_research_forecast audit
```
