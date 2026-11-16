# Experiment 15 — Multi-Dimensional Degradation Detection and Detector Comparison

**Data:** EXPERIMENT 14/14B DAILY INPUT: experiment_14_daily_monitoring.csv (not independently verified)  
**Period:** 2025-07-01 to 2026-10-13  
**Reference:** 2025-07-01 to 2026-02-28 (243 days)  
**Evaluation:** 2026-03-01 to 2026-10-13 (227 days)

## Objective
Compare fixed P95, CUSUM, Page-Hinkley, EWMA and (if installed) ADWIN across complementary probabilistic monitoring signals. Assess both below- and above-target PI80 coverage, its temporal variation, CRPS, interval width, benchmark-relative performance and realized-volatility-adjusted performance.

## Methods
- Signals evaluated: absolute_crps, coverage_signed, coverage_undercoverage, coverage_overcoverage, coverage_absolute_deviation, rolling_coverage_deviation_14d, rolling_coverage_instability_14d, interval_width_change, relative_crps_skill_loss, relative_rmae, volatility_adjusted_crps, volatility_adjusted_width.
- Detectors evaluated: Fixed P95, CUSUM, Page-Hinkley, EWMA, ADWIN.
- Thresholds are chosen from the reference period only using a heuristic raw alarm-day rate target, not a validated event-level false-alarm constraint.
- Synthetic signal-level stress tests: 20 trials per scenario, method and signal.
- Coverage deviation is two-sided: sustained undercoverage **and** sustained overcoverage may be problematic.
- The input file might omit PI80 width; this metric is then **skipped** rather than fabricated.

## Observed historical alerts (unvalidated)
Counts below are *not* false-positive rates: there are no independently verified real-world deterioration labels.

| Detector | Evaluation alarm rows (across all signals) |
|---|---:|
| Fixed P95 | 572 |
| CUSUM | 464 |
| Page-Hinkley | 436 |
| EWMA | 487 |
| ADWIN | 14 |

Counts sum across separate signal alarms and must not be interpreted as distinct events or as a ranking.

## Synthetic controlled-change comparison (illustrative)
These trials perturb daily monitoring signals, **not the underlying 101 predictive quantiles**. Results only characterize detector mechanics.

| Signal | Detector | Scenario | Detection rate | Median detection delay (days) | Pre-event false alarms / 100 days |
|---|---|---|---:|---:|---:|
| absolute_crps | ADWIN | shift_up | 0.25 | 11.0 | 0.1 |
| absolute_crps | CUSUM | shift_up | 1.00 | 2.0 | 4.3 |
| absolute_crps | EWMA | shift_up | 1.00 | 2.0 | 3.5 |
| absolute_crps | Fixed P95 | shift_up | 0.95 | 4.0 | 5.3 |
| absolute_crps | Page-Hinkley | shift_up | 1.00 | 3.0 | 4.4 |
| coverage_absolute_deviation | ADWIN | shift_up | 1.00 | 11.0 | 0.0 |
| coverage_absolute_deviation | CUSUM | shift_up | 1.00 | 1.0 | 2.0 |
| coverage_absolute_deviation | EWMA | shift_up | 1.00 | 1.0 | 0.7 |
| coverage_absolute_deviation | Fixed P95 | shift_up | 1.00 | 0.0 | 3.5 |
| coverage_absolute_deviation | Page-Hinkley | shift_up | 1.00 | 0.0 | 4.6 |
| coverage_overcoverage | ADWIN | shift_up | 0.70 | 11.0 | 0.0 |
| coverage_overcoverage | CUSUM | shift_up | 1.00 | 1.0 | 0.7 |
| coverage_overcoverage | EWMA | shift_up | 1.00 | 4.0 | 0.0 |
| coverage_overcoverage | Fixed P95 | shift_up | 1.00 | 0.0 | 0.0 |
| coverage_overcoverage | Page-Hinkley | shift_up | 1.00 | 0.0 | 4.2 |
| coverage_signed | ADWIN | shift_up | 0.65 | 11.0 | 0.0 |
| coverage_signed | CUSUM | shift_up | 1.00 | 1.0 | 0.6 |
| coverage_signed | EWMA | shift_up | 1.00 | 4.5 | 0.2 |
| coverage_signed | Fixed P95 | shift_up | 1.00 | 0.0 | 0.0 |
| coverage_signed | Page-Hinkley | shift_up | 1.00 | 1.0 | 3.2 |
| coverage_undercoverage | ADWIN | shift_up | 0.70 | 11.0 | 0.0 |
| coverage_undercoverage | CUSUM | shift_up | 1.00 | 1.0 | 5.4 |
| coverage_undercoverage | EWMA | shift_up | 1.00 | 2.0 | 4.4 |
| coverage_undercoverage | Fixed P95 | shift_up | 1.00 | 2.0 | 4.4 |
| coverage_undercoverage | Page-Hinkley | shift_up | 1.00 | 1.0 | 4.2 |
| interval_width_change | ADWIN | shift_up | 0.70 | 11.0 | 0.7 |
| interval_width_change | CUSUM | shift_up | 1.00 | 2.0 | 3.4 |
| interval_width_change | EWMA | shift_up | 1.00 | 2.0 | 4.1 |
| interval_width_change | Fixed P95 | shift_up | 1.00 | 0.5 | 3.9 |
| interval_width_change | Page-Hinkley | shift_up | 1.00 | 2.0 | 5.2 |
| relative_crps_skill_loss | ADWIN | shift_up | 0.85 | 11.0 | 0.0 |
| relative_crps_skill_loss | CUSUM | shift_up | 1.00 | 0.0 | 4.8 |
| relative_crps_skill_loss | EWMA | shift_up | 1.00 | 1.0 | 3.9 |
| relative_crps_skill_loss | Fixed P95 | shift_up | 1.00 | 0.5 | 6.2 |
| relative_crps_skill_loss | Page-Hinkley | shift_up | 1.00 | 1.0 | 4.5 |
| relative_rmae | ADWIN | shift_up | 0.65 | 11.0 | 0.0 |
| relative_rmae | CUSUM | shift_up | 1.00 | 1.5 | 5.6 |
| relative_rmae | EWMA | shift_up | 1.00 | 1.0 | 6.0 |
| relative_rmae | Fixed P95 | shift_up | 1.00 | 3.5 | 5.4 |
| relative_rmae | Page-Hinkley | shift_up | 1.00 | 1.0 | 4.9 |
| rolling_coverage_deviation_14d | ADWIN | shift_up | 0.70 | 11.0 | 0.2 |
| rolling_coverage_deviation_14d | CUSUM | shift_up | 1.00 | 2.5 | 3.8 |
| rolling_coverage_deviation_14d | EWMA | shift_up | 1.00 | 2.0 | 4.5 |
| rolling_coverage_deviation_14d | Fixed P95 | shift_up | 1.00 | 4.0 | 7.0 |
| rolling_coverage_deviation_14d | Page-Hinkley | shift_up | 1.00 | 2.5 | 4.8 |
| rolling_coverage_instability_14d | ADWIN | shift_up | 0.70 | 11.0 | 0.3 |
| rolling_coverage_instability_14d | CUSUM | shift_up | 1.00 | 2.0 | 4.2 |
| rolling_coverage_instability_14d | EWMA | shift_up | 1.00 | 2.0 | 3.7 |
| rolling_coverage_instability_14d | Fixed P95 | shift_up | 0.85 | 5.0 | 5.2 |
| rolling_coverage_instability_14d | Page-Hinkley | shift_up | 1.00 | 2.0 | 4.0 |
| volatility_adjusted_crps | ADWIN | shift_up | 0.40 | 11.0 | 0.0 |
| volatility_adjusted_crps | CUSUM | shift_up | 1.00 | 2.0 | 4.9 |
| volatility_adjusted_crps | EWMA | shift_up | 1.00 | 3.0 | 4.5 |
| volatility_adjusted_crps | Fixed P95 | shift_up | 1.00 | 2.0 | 4.6 |
| volatility_adjusted_crps | Page-Hinkley | shift_up | 1.00 | 2.0 | 4.2 |
| volatility_adjusted_width | ADWIN | shift_up | 0.85 | 11.0 | 0.1 |
| volatility_adjusted_width | CUSUM | shift_up | 1.00 | 2.0 | 4.4 |
| volatility_adjusted_width | EWMA | shift_up | 1.00 | 2.0 | 3.5 |
| volatility_adjusted_width | Fixed P95 | shift_up | 1.00 | 0.0 | 5.1 |
| volatility_adjusted_width | Page-Hinkley | shift_up | 1.00 | 1.5 | 4.3 |

## Interpretation and limitations
- A high detection rate with many pre-event false alarms is not necessarily preferable to a slower detector.
- Upward and downward deviations from 80% PI80 coverage both matter; daily coverage from 24 hours is noisy, so use rolling windows and dependence-aware assessment.
- Increasing interval width can reflect greater anticipated market uncertainty; interpret width and coverage jointly.
- PI80 interval width is omitted if not present in the Experiment 14 daily CSV. Add a verified matched series to enable it.
- ADWIN is bidirectional; CUSUM, Page-Hinkley and EWMA are upward-only. Dedicated undercoverage and overcoverage signals allow explicit directional testing, but raw rankings are not fair without matched detection targets.
- A natural historical alarm is not proof of model failure. A single retrospective March 2026 screening episode is insufficient for reliable method ranking.
- Realized volatility, CRPS and coverage require observed prices and are not valid as earlier forecast-time warning inputs.
- Future research should introduce coherent perturbations to hourly forecast quantiles and recompute all metrics jointly, and compare event-level false alarms under matched budgets.

## Output files
- `figures/experiment_experiment_15_monitoring_signals.png`
- `figures/experiment_experiment_15_synthetic_detection_rates.png`
- `tables/experiment_15_input_daily_checked.csv`
- `tables/experiment_15_reference_settings.csv`
- `tables/experiment_15_historical_alerts_unvalidated.csv`
- `tables/experiment_15_synthetic_trials.csv`
- `tables/experiment_15_synthetic_comparison.csv`
- `reports/experiment_15_run_metadata.json`
- `reports/experiment_15_report.md`

**Status:** Research prototype, not independently validated degradation-detection performance.

## Expanded quantitative visual comparisons

The following plots were generated automatically from experiment outputs. Historical plots use observed monitoring data but lack independent failure labels. Simulation plots use controlled **signal-level**, not forecast-level, perturbations.

| Figure | Meaning |
|---|---|
| `figures/experiment_15_coverage_two_sided.png` | Numerical visualization |
| `figures/experiment_15_coverage_stability_comparison.png` | Numerical visualization |
| `figures/experiment_15_historical_alert_count_heatmap.png` | Numerical visualization |
| `figures/experiment_15_distinct_alert_days.png` | Numerical visualization |
| `figures/experiment_15_crps_detector_agreement.png` | Numerical visualization |
| `figures/experiment_15_simulated_detection_heatmap.png` | Numerical visualization |
| `figures/experiment_15_crps_detection_delay_falsealarms.png` | Numerical visualization |
| `figures/experiment_15_under_vs_overcoverage_detection.png` | Numerical visualization |
| `figures/experiment_15_detection_tradeoff_scatter.png` | Numerical visualization |

### Observed PI80 coverage summary

| period     |   days |   mean_coverage_pct |   mean_absolute_deviation_pp |   under_80_days |   over_80_days |   exactly_80_days |
|:-----------|-------:|--------------------:|-----------------------------:|----------------:|---------------:|------------------:|
| reference  |    243 |              87.229 |                       13.477 |              66 |            176 |                 1 |
| evaluation |    227 |              80.46  |                       12.888 |             100 |            127 |                 0 |

### Selected synthetic controlled upward-shift comparison

See `tables/experiment_15_numeric_detector_comparison.csv` for the full numerical comparison. High simulated detection alone does not establish a superior real-world model-health detector.
