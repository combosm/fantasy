# Fantasy Basketball Projections

An independent, auditable 2026–27 NBA projection model for ESPN head-to-head points leagues.
It projects games, minutes, usage, pace and a coherent box score, then produces **two boards**:
raw expected season points and eight-team points above waiver replacement. Historical preseason
roster snapshots are still missing; four dated ESPN preseason points lists are available as an
independent benchmark, not as model inputs.
The configured scoring matches [ESPN's default head-to-head points system](https://www.espn.com/fantasy/basketball/story/_/id/30296896/espn-fantasy-new-default-points-league-scoring-explained).

## Quick start

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

# Verify everything with deterministic fixture data (no network required)
fantasy all --fixture

# With current cached NBA data, produce both CSV draft boards
fantasy project
```

`fantasy project` writes `output/raw_points_board_2026-27.csv` and
`output/draft_value_board_2026-27.csv`; intermediate projections and validation reports
are written to `data/processed/`. The legacy Excel exporter is separate from these two boards.

## Commands

```bash
fantasy ingest             # NBA.com history and target-season rosters
fantasy ingest --fixture   # deterministic test/demo dataset
fantasy project            # projections and cached tabular outputs
fantasy validate           # constraints and walk-forward backtests
fantasy backtest           # end-to-end season forecast vs naive baselines
fantasy check-uncertainty  # interval coverage from saved backtest predictions
fantasy extend-history     # append locally downloaded public season archives
fantasy ingest-drafts       # cache official NBA draft history for rookie priors
fantasy export             # legacy one-board workbook (not the two-board deliverable)
fantasy all                # live-data pipeline plus legacy workbook
```

All commands accept `--config PATH` before the command name.

## Inputs

Editable CSVs live in `data/inputs/`:

- `external_rankings.csv`: current dated FantasyPros ESPN-points rankings, audit-only.
- `market_preseason_rankings.csv`: four archived ESPN points cheat sheets (2022–25), used as
  a dated historical market benchmark. These are one publisher, not a consensus or ADP.
- `player_context.csv`: optional documented MPG, usage and games-played overrides.
- `healthy_rotations.csv`: optional manual 240-minute rotation scenarios for a team;
  no positional slots are enforced.
- `team_context.csv`: optional team-pace override when reliable coaching evidence is available.
- `overrides.csv`: documented set/delta/multiply changes to primitive assumptions.
- `rookie_source_stats.csv`: per-game source-league records for players without NBA history.
- `preseason_rosters.csv`: optional dated, source-backed historical roster snapshots for the
  end-to-end backtest; dates must precede opening night. Without snapshots, the test explicitly
  carries forward the previous season's roster and does not claim to capture summer moves.
- `injury_context.csv`: documented manual expected-GP changes for known injuries. Each row
  requires status, review date, source and explanation. The current file includes provisional
  Jimmy Butler and Shaedon Sharpe scenarios; these are judgment calls, not medical forecasts.

Overrides must include a reason, source, review date and confidence. The model rejects targets that
are not primitive fields and prevents large rate/minutes changes outside configured safety bounds.

## Model outline

1. Weight up to 11 historical seasons by recency and sample size. The current recency weights
   are provisional, not yet selected by the end-to-end backtest.
2. Shrink per-minute statistics and shooting percentages toward league priors.
3. Separately forecast active-game MPG, usage, games played, and team pace. Availability and
   opportunity candidates are selected by mean-squared error, with opportunity selection
   evaluated among prior-season draft-relevant players. The forecasts target means rather
   than MAE-optimal medians.
   MPG and usage have current-teammate workload/playstyle features, without positional quotas.
   A separate candidate is selected for previously draft-relevant players coming off a
   low-GP season; it anchors role changes to a games-weighted three-year baseline. This
   subgroup route reduces historical MPG and usage error but is not a complete recovery model.
   For established starters with at least 30 MPG in each of the two preceding seasons and
   at least 40 games in each, a separate validation-selected MPG route avoids treating a
   disrupted latest season as a new permanent minutes limit. This does not override GP or
   usage and is not a blanket 30-MPG floor.
   A separate established-starter route also tests the prior active-game MPG against learned
   changes; it keeps the observed MPG when the learned model's historical advantage is within
   one season-block standard error. These role cohorts are small or context-sensitive, so
   their reported gains should be treated as development evidence, not final validation.
   Historical training currently has only carry-forward rosters, so teammate-arrival effects
   are not yet reliably learned or validated. Games played is forecast separately from MPG.
4. Compute on-court team possessions as MPG × team pace / 48. Forecast shot-ending plays from
   historical FGA + 0.44×FTA + turnovers per minute, adjusted by projected MPG, pace change,
   and usage change. This avoids equating usage's team-play denominator with possessions.
   Preserve the historical FGA/FTA/turnover mix; scale rebounds, assists, steals and blocks by
   possessions. Calculate makes from attempts
   and regressed percentages; reconcile points from FGM, 3PM and FTM.
5. Reconcile a team over 240 availability-weighted minutes per scheduled game by a logged,
   uniform MPG reduction. Do not fill under-capacity teams: unprojected players may take those
   minutes. This is a feasibility correction after MPG and GP have been forecast separately,
   but it still couples the final MPG to GP and can over-shrink crowded teams. It is a known
   limitation, not a validated healthy-rotation optimizer.
6. Apply configurable ESPN scoring and simulate 10,000 seasons. When a recent end-to-end
   backtest exists, sample paired FPPG/games errors from comparable historical players;
   otherwise label the legacy simulation as uncalibrated.
7. Produce both rankings. The raw board sorts `projected FPPG × projected games`. The draft
   board sorts `(projected FPPG − waiver FPPG) × projected games`. The waiver FPPG is the
   105th-highest projected FPPG among current candidates: eight teams × (10 active + 3 bench)
   creates 104 drafted spots, using the user's confirmed roster size and
   [ESPN's standard roster description](https://www.espn.com/fantasy/basketball/story/_/id/20800285/adjusting-settings).
   A manual `replacement_fppg` in config can override that estimate.
   This assumes missed games can be filled from waivers and ignores position eligibility.
8. Assign separate tiers and compare the raw board with a dated, audit-only external ranking.
   `sanity_review.csv` audits the full projected pool, not just the top 250, for low ranks,
   market gaps, role drops, and availability anomalies.

The workbook exposes factor forecasts, contextual changes, team audits, factor backtests,
unmatched names and the external-ranking audit queue.

`Team Rotations` lists every projected roster member, including players outside the top 250.
It shows minutes when active and availability-weighted minutes per scheduled team game.
`Healthy Rotations` contains only optional manually supplied team scenarios.

## Known data boundary

The live importer intentionally caches NBA.com responses because the public Stats endpoints can be
intermittent. Populate `rookie_source_stats.csv` with NCAA or international per-game statistics;
the pipeline translates them through conservative league factors and applies wider uncertainty.
Current drafted rookies without source-league statistics receive a generic, draft-slot-and-age
translation learned from historical NBA debut seasons; it is not position- or skill-specific and
its GP estimate excludes stashed/nonplaying picks. Remaining players without NBA history or source
statistics are explicitly flagged rather than silently receiving a veteran default.

Coach changes, injury diagnoses, rookie translation, and shot-quality effects are not reliably
learned from box scores alone. They are not automatically inferred;
documented pace/role overrides can be used when supported by better evidence. The `Baseline Backtest`
tab is the historical box-score baseline; `Factor Backtest` evaluates MPG, usage, GP and
pace models separately. The workbook does not yet contain the newer end-to-end backtest.
`sanity_review.csv` surfaces players whose projections remain far from a dated external ranking
or their own prior production. The established-starter MPG route addresses an identified
underprojection pattern, but GP, new-roster effects and the team-minute reconciliation remain
material sources of uncertainty; review the flagged players before drafting.

`fantasy backtest` runs the factor-to-fantasy pipeline season by season, with no target-season
box scores or eventual rosters used as prediction inputs. It compares fantasy points/game,
games, season totals, rank correlation, and top-12/top-30/top-50/top-60 overlap against last
season unchanged and a three-year weighted average. Top-K overlap is the fraction of the
actual top K players also in the predicted top K, measured against every player in the actual
season (including rookies or others absent from the projection pool). Raw season points and
draft value over replacement have separate scorecards. The last four seasons are summarized
in `data/processed/topk_target_summary.csv` against configurable goals of 80–85% for top 30
and 70–75% for top 60. These are targets, not claims of achieved accuracy. The report also
shows the candidate-pool ceiling and how many more correct selections are needed to reach
each minimum. A separate probability-ranking experiment learns from earlier held-out seasons
only, using the factor model and two baseline forecasts as inputs. It evaluates top 30 and
top 60 separately for both boards, and its results are included in the same target summary.
Its per-season results and player scores are saved as `topk_probability_backtest.csv` and
`topk_probability_predictions.parquet`. Season outcomes are held out from model fitting,
but this is still a development-set experiment rather than an untouched final validation.
These probability scores have not been shown to be calibrated and do not replace the current draft boards. Row-level forecasts,
conditional FPPG/GP oracle experiments, eight-normal-season top-K summaries, NDCG, top-100
rank correlation, value regret and season-block bootstrap intervals are saved in `data/processed`.
The ESPN market benchmark covers only the four seasons with dated sheets.
The report also contains two trajectory-aware alternatives: a shallow boosted classifier and
a regularized logistic classifier. They add only pre-cutoff prior-season age, games, minutes,
usage, fantasy production, per-minute rates, and year-over-year changes. Neither uses game-log
trends or roster changes; both remain experiments and are not used in the draft boards.
Until dated historical snapshots are supplied, the reported roster source is
`previous_season_carryforward`, which is leak-free but misses summer transactions and rookies.
The `uncertainty_coverage.csv` report checks whether nominal 20th–80th intervals actually cover
60% of future season totals. They currently under-cover, particularly in 2025–26, and should
not be treated as calibrated probabilities.
Older seasons (2015–16 through 2020–21) come from the public
[`llimllib/nba_data` player-season archive](https://github.com/llimllib/nba_data/tree/main/data).
Place `players_2016.parquet` through `players_2021.parquet` under `data/raw/archive/` and run
`fantasy extend-history` after normal NBA ingestion. The archive's season totals are divided
by games played and checked for unique player-season IDs. Historical opportunity and pace
training currently carry the prior roster forward to avoid using eventual team assignments.
For availability fitting, the shortened 2019–20 and 2020–21 seasons are excluded as targets;
2020–21 GP is normalized from its 72-game schedule when used as a lag, while uneven 2019–20
GP is treated as unavailable without team game logs.

### Related approaches and what transferred

- [Fantasy_Basketball_ML](https://github.com/iocak28/Fantasy_Basketball_ML) forecasts
  *daily* DraftKings/FanDuel results. Its useful transferable ideas are lagged player and
  team features, learned rather than hand-picked moving-average weights, and time-ordered
  validation. Daily salary, betting lines, opponents, and its reported nightly prediction
  results do not directly validate a season-long ESPN top-30 ranking.
- [KengoA/fantasy-basketball](https://github.com/KengoA/fantasy-basketball) compares simple
  and weighted recent-game baselines with larger models on a held-out month. This supports
  testing recent-game form against a simple baseline, but its daily salary-cap lineup task
  is different from ours.
- [Papageorgiou et al. (2024)](https://link.springer.com/article/10.1007/s10115-024-02092-9)
  find that forecasting the component stats and then constructing fantasy points improved
  their fantasy-point error. Our factor-to-box-score model already follows that structure;
  their filtered 90-player, next-game evaluation does not establish season top-K accuracy.

As a test of the first project's learned-lag idea, `learned_lag.py` fits an expanding-window
ridge model on the previous three seasons' fantasy points, games, minutes, usage, and age.
On the last four seasons, it hit 78/120 draft-value top-30 players (65.0%), versus 87/120
(72.5%) for the existing probability ranker. It is retained as a documented comparison,
not used to set the draft boards. The newer trajectory classifiers likewise did not beat
the existing ranker on draft-value top 30. This evidence points toward collecting
point-in-time roster, role, injury, and late-season game-log data before adding more model
complexity to the same season averages.
An optional game-log experiment now reads the public
[`data/espn/player_box.parquet`](https://github.com/llimllib/nba_data/tree/main/data/espn)
archive (save it as `data/raw/archive/espn_player_box.parquet`). It uses only the final 20
played regular-season games of the *prior* season, with starter share, minutes, fantasy
points, shots, assists, rebounds, assisted-shot share, and how long before season end the
player last appeared in league-game sequence (not elapsed days). Seasons with fewer than 900 distinct regular-season games are rejected
as incomplete. The results are written to `game_form_backtest.csv` when this optional file
exists. In the last four folds, this reached 82/120 (68.3%) draft-value top-30 hits, below
the existing ranker's 87/120, so it is not used for the draft boards. Game logs alone did
not capture the offseason roster and injury changes behind many breakout seasons.
When the archive is available, `fantasy project` also writes
`data/processed/game_form_review.csv` for the current draft. This is an audit-only view
of those pre-cutoff signals beside each player’s raw and draft ranks; it does not alter
projected points, games, or rank.
