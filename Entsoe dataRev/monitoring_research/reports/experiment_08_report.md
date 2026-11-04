# Experiment 08 — Indicator Correlation and Redundancy Analysis

## Purpose
Examine whether realized forecast-quality indicators carry overlapping information before selecting components of the LEAR Health Index. This experiment does **not** choose final weights or optimize the index.

## Input and periods
- Input: `market_regime_results/health_baseline/daily_health_baseline.csv`
- Calibration/reference: 2025-07-01 through 2025-12-31 (184 days).
- Evaluation: 2026-01-01 through 2026-10-13 (286 days).
- Daily Pearson and Spearman correlations; also trailing 30-day averages with a minimum of 15 days.
- Rolling windows use current and earlier dates only; early evaluation rolling values include late calibration days.

## Indicators
- CRPS, MAE, RMSE, 80% interval score: realized forecast performance.
- Absolute 80% coverage deviation: `abs(observed coverage - 0.80)`, penalizing both over- and undercoverage.
- Absolute PIT-mean deviation: `abs(mean PIT - 0.50)`; does **not** measure complete PIT uniformity.
- Absolute PIT-variance deviation: `abs(PIT variance - 1/12)`; another incomplete PIT diagnostic.
- 80% interval width is correlated with the health metrics separately, because width is available at forecast time and is **not** itself a realized-error metric.

## High correlations in raw daily Pearson analysis (descriptive |r| >= 0.85)
| Period | Indicators | Pearson r |
|---|---|---:|
| evaluation | CRPS / MAE | +0.989 |
| calibration | CRPS / MAE | +0.986 |
| calibration | CRPS / RMSE | +0.980 |
| evaluation | MAE / RMSE | +0.979 |
| evaluation | CRPS / RMSE | +0.979 |
| calibration | MAE / RMSE | +0.973 |
| evaluation | CRPS / Interval score 80% | +0.966 |
| calibration | CRPS / Interval score 80% | +0.951 |
| evaluation | RMSE / Interval score 80% | +0.934 |
| calibration | RMSE / Interval score 80% | +0.926 |
| evaluation | MAE / Interval score 80% | +0.923 |
| calibration | MAE / Interval score 80% | +0.894 |

## Interpretation rules and limitations
1. Pearson measures linear association; Spearman measures rank-monotonic association. Neither establishes causality or that two metrics are interchangeable.
2. High correlations can arise from common market difficulty, outliers, seasonal changes, or overlapping rolling windows. Check raw daily results before interpreting rolling correlations.
3. The 2025 reference period lacks spring and is only six months long; cross-period changes are confounded with seasonal and market changes.
4. Daily absolute coverage deviations are noisy because there are usually only 24 hourly observations per day; conditional hourly coverage needs longer aggregation.
5. PIT-mean and PIT-variance deviations are incomplete calibration diagnostics. Experiment 05's distributional PIT findings remain important.
6. Width is available at forecast issue time; realized CRPS, MAE, RMSE, interval score, PIT and coverage are not. Keep Health and Early-Warning signals separate.
7. |r| >= 0.85 is a **screening convention**, not an automatic deletion rule or a statistical significance threshold. Serial dependence invalidates naive independent-day significance claims.

## Outputs
- `experiment_08_*_pearson.csv` and `experiment_08_*_spearman.csv`: matrices by period and granularity.
- `experiment_08_indicator_pairs.csv`: pairwise correlations, including width comparisons.
- `experiment_08_high_correlation_pairs.csv`: screening list.
- `experiment_08_period_comparison.csv`: changes in daily Pearson correlations.
- `experiment_08_sample_sizes.csv`: available observations.
- Matching PNG and PDF heatmaps.

## Decision before Experiment 09
Review the empirical matrices and select complementary signals. Do not treat this exploratory analysis as validated index construction or as a detector evaluation.
