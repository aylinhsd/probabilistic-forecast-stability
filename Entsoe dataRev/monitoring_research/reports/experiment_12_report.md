# Experiment 12 — Market volatility versus probabilistic LEAR v3 performance

## Research question
How does observed within-day electricity price volatility relate to the accuracy, sharpness, and calibration of native probabilistic LEAR v3 forecasts?

## Data and procedure
- Source: `ENTSOE_Probabilistic_LEAR_v3_predictions.csv`; `Split == test` only.
- 11,280 test hours across 470 Italian calendar dates (2025-07-01 to 2026-10-13).
- Volatility: sample standard deviation of realized hourly electricity prices within each Italian calendar date.
- Approximate CRPS: twice the trapezoidal integral of pinball loss across available quantiles.
- MAE: absolute error of the predicted median, averaged by date.
- PI80 width: Q0.900 minus Q0.100, averaged by date.
- PI80 coverage: share of hourly actual prices within [Q0.100, Q0.900] per date.
- Days assigned to equally sized low/medium/high groups using realized volatility ranks.

## Results — daily correlations
| Metric | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---:|---:|---:|---:|
| CRPS_approx | +0.7925 | 1.44e-102 | +0.7873 | 2.36e-100 |
| MAE | +0.8016 | 1.26e-106 | +0.7639 | 4.77e-91 |
| PI80_width | +0.6636 | 5.54e-61 | +0.6973 | 1.09e-69 |
| PI80_coverage | -0.4025 | 9.85e-20 | -0.4029 | 9.1e-20 |

## Results — volatility groups
| Group | Days | Mean volatility | Mean CRPS | Mean MAE | Mean PI80 width | PI80 coverage |
|---|---:|---:|---:|---:|---:|---:|
| Low | 157 | 11.96 | 4.78 | 6.43 | 28.08 | 89.46% |
| Medium | 156 | 19.83 | 7.16 | 9.75 | 35.65 | 85.18% |
| High | 157 | 37.93 | 12.36 | 17.09 | 46.68 | 77.24% |

## Interpretation and limitations
- Associations are descriptive, not evidence of causation or model failure.
- Same-day realized volatility is known only after the target day; it cannot serve as a same-day forecast-issue early-warning feature.
- Daily observations may be serially dependent, so conventional correlation p-values should not be treated as definitive inference.
- Approximate CRPS integrates only over the supplied quantile range, not the entire distribution tails.
- A daily average of coverage is noisy; the nominal 80% rate is not guaranteed on each individual day.
- Group boundaries are defined using the full test sample; they are descriptive rather than prospectively calibrated thresholds.
- This experiment does not establish whether deterioration is model-specific; benchmark-relative performance is the next question.

## Saved outputs
- `figures/experiment_12_volatility_correlations.png` (all four metrics, with Pearson and Spearman correlations)
- `figures/experiment_12_volatility_correlations.pdf`
- `tables/experiment_12_daily_metrics.csv`
- `tables/experiment_12_volatility_group_summary.csv`
- `figures/experiment_12_volatility_crps.png`
- `figures/experiment_12_volatility_width.png`
- `figures/experiment_12_volatility_coverage.png`
- `figures/experiment_12_volatility_groups.png`
- `reports/experiment_12_report.md` (this file)
