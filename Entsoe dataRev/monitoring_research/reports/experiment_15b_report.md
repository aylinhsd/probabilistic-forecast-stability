# Experiment 15b — Forecast-level detector selection

## Aim in simple terms
Deliberately change saved hourly price forecasts, then ask: **which detector notices the problem promptly without sending too many unnecessary alerts?** We test incorrect forecast levels, slow drift, short disruptions and intervals that become too narrow or too wide.

## Main finding
CRPS methods meeting the null-notification budget in all three test folds: **ADWIN upward-confirmed: 23.3% mean bias-event detection**. In the final fold, validation selected **CUSUM**: test detection 42.3%, baseline notifications 0.66/100 days (budget met). Selection varies with period and signal; these results do not establish a universal winner.

## What was run
- Original native Test forecasts only, with all 101 saved quantiles and complete Italian calendar days, including DST days. Actual prices and the benchmark are never changed.
- Three expanding chronological folds: fitting, threshold calibration, method selection (validation), then held-out testing. Test windows do not overlap. Later folds may use earlier dates as past information.
- 300 paired trials per scenario, fold and validation/test stage; 200 separate calibration trials. Moving blocks contain 7 complete days. The same sampled actuals and quantile curves are shared across methods and signals.
- Budget: **1 notification per 100 empirical-null days, per signal**. Notifications are at least 7 days apart; continuing alarms produce another notification after that gap. State and cooldown continue across the known injection onset.
- Fixed P95 is an untuned comparator. Tuned quantile is its budget-calibrated counterpart. CUSUM uses k=0.5, Page-Hinkley delta=0.15 and EWMA alpha=0.2 on reference-standardized signals; only thresholds are calibrated. ADWIN is included when River is installed, with explicit upward direction confirmation and clock=1 so it checks daily (River's default clock=32 would check only every 32 days).
- Thresholds are selected on **no-injection calibration trials only**, taking the most sensitive grid value satisfying the notification budget. A method must also meet the independent validation-null budget to be eligible for selection.
- For each signal, validation chooses the method with highest average event detection, then lowest restricted delay, then fewer null notifications. That choice is frozen before test evaluation. All methods are still reported on the test set.

## Chronological splits
| fold | period | start | end | days |
| --- | --- | --- | --- | --- |
| 1 | fit | 2025-07-01 | 2025-10-28 | 120 |
| 1 | calibration | 2025-10-29 | 2025-12-27 | 60 |
| 1 | validation | 2025-12-28 | 2026-02-25 | 60 |
| 1 | test | 2026-02-26 | 2026-04-26 | 60 |
| 2 | fit | 2025-07-01 | 2025-12-27 | 180 |
| 2 | calibration | 2025-12-28 | 2026-02-25 | 60 |
| 2 | validation | 2026-02-26 | 2026-04-26 | 60 |
| 2 | test | 2026-04-27 | 2026-06-25 | 60 |
| 3 | fit | 2025-07-01 | 2026-02-25 | 240 |
| 3 | calibration | 2026-02-26 | 2026-04-26 | 60 |
| 3 | validation | 2026-04-27 | 2026-06-25 | 60 |
| 3 | test | 2026-06-26 | 2026-10-13 | 110 |

## Controlled changes
- Sudden forecast bias in both directions: 0.5, 1 and 2 times the fitting-period median PI80 half-width.
- Gradual bias in both directions: ramps to one half-width over 35 days.
- Temporary bias in both directions: one half-width for three days, then forecasts return exactly to their untouched counterparts.
- Interval contraction: width multipliers 0.8 and 0.5; expansion: 1.25 and 1.75, around the unchanged median.
- Bias or positive width scaling preserves quantile ordering. Approximate CRPS, coverage and width are recalculated from the transformed hourly curves **before** paired daily-block sampling. Gradual-ramp metrics use the appropriate daily ramp strength.
- Signals: absolute CRPS; CRPS / benchmark CRPS minus one; undercoverage max(0,0.8−coverage); overcoverage max(0,coverage−0.8); and CRPS residual after a fitting-only regression on log(1+realized daily volatility).
- The benchmark matches Experiment 13: point LEAR plus fixed validation-residual quantiles from before the native Test period. No held-out residuals are fitted.

## Main held-out CRPS comparison
Numbers below equally weight folds and the ten bias scenarios. Temporary-event detection uses a three-day window; persistent/gradual detection uses 30 days. Restricted delay assigns misses the corresponding window length; conditional median delays are in the detailed tables.

| method | detection_rate | detection7 | restricted_delay | null_notifications_per_100 | test_folds_meeting_budget |
| --- | --- | --- | --- | --- | --- |
| Fixed P95 | 0.901 | 0.828 | 2.877 | 11.482 | 0 |
| Tuned quantile | 0.643 | 0.496 | 9.833 | 5.899 | 1 |
| CUSUM | 0.728 | 0.483 | 8.938 | 6.556 | 1 |
| Page-Hinkley | 0.835 | 0.539 | 6.896 | 6.920 | 1 |
| EWMA | 0.744 | 0.587 | 7.483 | 6.687 | 1 |
| ADWIN upward-confirmed | 0.233 | 0.059 | 19.969 | 0.515 | 3 |

**Read the budget column before comparing detection rates.** High detection from an over-budget method does not make it eligible. The table is descriptive, not a test-set optimization rule.

## Choices made on validation, evaluated on test
| fold | signal | method | validation_detection | test_detection | test_null_notifications_per_100 | test_budget_met |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | absolute_crps | Page-Hinkley | 0.562 | 0.887 | 8.224 | False |
| 1 | relative_crps_loss | Tuned quantile | 0.990 | 0.940 | 3.531 | False |
| 1 | undercoverage | ADWIN upward-confirmed | 0.502 | 0.545 | 0.000 | True |
| 1 | overcoverage | CUSUM | 0.750 | 0.197 | 0.000 | True |
| 1 | volatility_adjusted_crps | Page-Hinkley | 0.781 | 0.892 | 7.511 | False |
| 2 | absolute_crps | ADWIN upward-confirmed | 0.212 | 0.243 | 0.544 | True |
| 2 | relative_crps_loss | ADWIN upward-confirmed | 0.589 | 0.506 | 0.004 | True |
| 2 | undercoverage | ADWIN upward-confirmed | 0.555 | 0.695 | 0.233 | True |
| 2 | overcoverage | Page-Hinkley | 0.568 | 0.327 | 0.000 | True |
| 2 | volatility_adjusted_crps | ADWIN upward-confirmed | 0.296 | 0.272 | 0.471 | True |
| 3 | absolute_crps | CUSUM | 0.759 | 0.423 | 0.660 | True |
| 3 | relative_crps_loss | ADWIN upward-confirmed | 0.506 | 0.518 | 0.002 | True |
| 3 | undercoverage | CUSUM | 0.977 | 0.638 | 0.007 | True |
| 3 | overcoverage | CUSUM | 0.752 | 0.998 | 2.620 | False |
| 3 | volatility_adjusted_crps | CUSUM | 0.704 | 0.363 | 0.322 | True |

Selection frequency (a shortlist, not a universal winner):

| signal | method | folds_selected |
| --- | --- | --- |
| absolute_crps | ADWIN upward-confirmed | 1 |
| absolute_crps | CUSUM | 1 |
| absolute_crps | Page-Hinkley | 1 |
| overcoverage | CUSUM | 2 |
| overcoverage | Page-Hinkley | 1 |
| relative_crps_loss | ADWIN upward-confirmed | 2 |
| relative_crps_loss | Tuned quantile | 1 |
| undercoverage | ADWIN upward-confirmed | 2 |
| undercoverage | CUSUM | 1 |
| volatility_adjusted_crps | ADWIN upward-confirmed | 1 |
| volatility_adjusted_crps | CUSUM | 1 |
| volatility_adjusted_crps | Page-Hinkley | 1 |

For CRPS-based signals, selection averages bias scenarios; undercoverage selects on interval-contraction scenarios and overcoverage on expansion scenarios. This declared equal weighting is a research choice, not an operational cost function. `NONE` means no method passed the budget; do not force a winner. A frozen selection that exceeds the test-null budget is not operationally validated.

## Interpretation and limitations
- No historical dates have independently verified stable/failure labels. Therefore these are **empirical-null alerts and controlled perturbation detections**, not real-world false-positive or failure-detection estimates. Unchanged baseline periods may contain real changes.
- Budget matching is an empirical constraint, not exact equality or a statistical guarantee. Calibration/validation/test achieved rates and intervals are exported. Fixed P95 can fail the budget and is then excluded from selection.
- The budget is **per signal**. Combining five monitors needs separate joint notification calibration; it cannot inherit this budget.
- All trial detections count notifications, including chance coincidences. A paired untouched trial is evaluated on the identical sampled days. `paired_excess_detection` quantifies detection above these coincident baseline alerts.
- Median delay is conditional on detection. Misses, detection within 7/14/30 days and restricted delay are reported separately. For three-day events, all horizon columns are capped at the event's duration.
- Widening may improve undercoverage while reducing sharpness; additive bias can partly correct existing model bias. Achieved CRPS/coverage/width changes and fractions with worse CRPS are exported. Scenario names describe interventions, not guaranteed overall model damage.
- The 95% Wilson detection intervals and trial-based rate/delay intervals describe conditional Monte Carlo variation. Resampled days repeat a limited historical pool; trials are not independent real market failure episodes. No claim of statistical superiority is made from overlapping intervals.
- Seven-day blocks preserve some within-block dependence, but joins disrupt longer dependence and calendar structure. Fold and volatility summaries probe robustness; they do not reproduce every seasonal regime. No source history is removed as an inconvenient null period.
- All scores depend on observed prices. This is **after-outcome health monitoring**, not a forecast-time early-warning experiment. The same-day volatility residual is explanatory adjustment, not a prospective input.
- Upward-confirmed ADWIN is a modified comparator: a drift alert is accepted only if the latest seven-day mean exceeds both the fitting mean and the preceding seven-day mean. This gate can miss late changes and does not prove causal direction.
- This experiment evaluates predeclared parameter grids, not every possible tuning of each algorithm. Each trial initializes detector state from the final 30 fitting days; it does not estimate performance for every possible deployment initialization.

## Files
- `tables/experiment_15b_splits.csv`, `_input_daily.csv`, `_calibration.csv`
- `tables/experiment_15b_null_trials.csv`, `_null_summary.csv`
- `tables/experiment_15b_event_trials.csv.gz` (compressed detailed trials), `_event_summary.csv`
- `tables/experiment_15b_scenario_effects.csv`, `_volatility_summary.csv`
- `tables/experiment_15b_method_selection.csv`, `_crps_method_comparison.csv`
- `tables/experiment_15b_historical_alerts_unvalidated.csv`, `_example_daily_paths.csv`
- `figures/experiment_15b_detection_heatmap.png`, `_false_alert_tradeoff.png`, `_calibration_detection.png`, `_coherent_forecast_examples.png`, `_volatility_robustness.png`
- `reports/experiment_15b_run_metadata.json`

Checks: UTC/local-date alignment, complete DST-aware days, no quantile crossings, benchmark actual alignment, identity transformation, width scaling, coverage nesting, notification cooldown, miss-aware delay, chronological splits. Runtime: 11.2 seconds.
