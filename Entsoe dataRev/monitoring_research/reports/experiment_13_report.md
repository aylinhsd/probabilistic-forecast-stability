# Experiment 13 — Absolute versus benchmark-relative performance

## Research question
Does native Probabilistic LEAR v3 maintain its advantage over existing benchmarks during difficult market conditions?

## Inputs and alignment
- Native probabilistic LEAR v3: test split, 11,280 matched hourly timestamps, 470 Italian calendar dates.
- Seasonal naive: reused its previously generated point forecasts; joined on `Target_UTC`, **not** on its original split labels.
- Residual-based LEAR v3: reconstructed from original LEAR v3 point forecasts plus **fixed empirical quantiles of its validation residuals** (no test residuals used).
- Evaluation window: 2025-07-01 to 2026-10-13.
- Validation residual count: 4343.
- All compared observations share the same realized price, checked numerically.

## Methods
- Native point forecast: median `Q_0.500`; seasonal-naive and original point LEAR use their saved point predictions.
- Approximate CRPS: twice the trapezoidal integral of quantile/pinball loss over the common saved quantile grid (0.010–0.990); tails outside this grid are omitted for both models.
- Daily rMAE = daily native MAE / daily naive MAE. Daily CRPSS = 1 − daily native CRPS / daily residual CRPS.
- **Overall** and group ratios use ratios of pooled mean errors, not averages of daily ratios.
- PI80 coverage uses inclusive 10th/90th percentile bounds; width is upper minus lower.
- Volatility = sample SD of realized hourly prices per Italian calendar date. Groups are equal-frequency tertiles of daily volatility.

## Results — pooled test period
| Indicator | Result |
|---|---:|
| Native MAE | 11.0899 €/MWh |
| Original point LEAR MAE | 11.7219 €/MWh |
| Seasonal naive MAE | 16.4261 €/MWh |
| Native / naive rMAE | 0.6751 |
| Native approximate CRPS | 8.1002 €/MWh |
| Fixed-residual approximate CRPS | 8.9275 €/MWh |
| Native / residual CRPSS | 0.0927 |
| Native PI80 coverage | 83.96% |
| Residual PI80 coverage | 77.56% |
| Native PI80 width | 36.8049 €/MWh |
| Residual PI80 width | 31.9105 €/MWh |

## Interpretation guidance
- rMAE below 1 means lower absolute error than seasonal naive; above 1 means worse.
- CRPSS above 0 means lower approximate CRPS than fixed-residual LEAR; below 0 means worse.
- Relative advantage is not proof of model health; absolute errors and coverage must also be monitored.
- Examine daily and volatility-group outputs before making claims about stability or deterioration.

## Numerical graphs
- [Relative performance by volatility group](../figures/experiment_13_relative_performance_by_volatility_group.png): labeled pooled rMAE and CRPSS, with equal-performance reference lines.
- [MAE, CRPS and coverage comparison](../figures/experiment_13_numerical_metric_comparison.png): labeled overall and volatility-group results for native forecasts and the corresponding benchmarks.
- [Updated scatterplots](../figures/experiment_13_volatility_relative_performance.png): daily rMAE and CRPSS against volatility, with straight least-squares trend lines, equations, Pearson r and Spearman rho.
- Each graph is also saved as a PDF with the same filename stem.
- Trend lines and correlations are descriptive; daily ratios are not pooled group skill scores.

## Limitations
- Residual calibration uses the original point model's validation period, which was also used during native model selection; this is an existing-method comparison, not a fully independent prospective benchmark study.
- The fixed-residual comparator is reconstructed from saved validation residuals; verify its width against the previously reported ~31.91 €/MWh. Differences may indicate different calibration conventions or files.
- The seasonal naive uses historical predictions from a differently partitioned experiment. Timestamp and target checks are essential.
- Volatility uses realized same-day prices, so it is **retrospective**, not available at forecast issue time.
- Daily ratios can be unstable when denominator errors are small; examine pooled ratios for main conclusions.
- CRPS is a finite-quantile approximation; do not compare directly with results calculated using a different integration convention.
- Correlations and volatility strata do not establish causation or detect confirmed model failure.
