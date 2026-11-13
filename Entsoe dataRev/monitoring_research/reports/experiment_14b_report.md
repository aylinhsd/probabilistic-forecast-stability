# Experiment 14B — Alert severity and diagnostic evaluation

## Scope
Uses Experiment 14 daily metrics; no model retraining and no independent failure labels.
All forward-seven-day summaries are **retrospective diagnostics**, never available at alert time.
The current-day volatility column is also ex post and is not used for alert issuance.

## Definitions
- Fixed severity = max(0, daily CRPS − fixed threshold), in €/MWh.
- Market-aware severity = max(0, daily CRPS − market-aware threshold), in €/MWh.
- Relative rMAE severity = max(0, rMAE − historical upper threshold).
- Relative CRPSS severity = max(0, historical lower threshold − CRPSS).
- High realized volatility = at or above reference-period 75th percentile of daily realized price SD; **ex-post diagnostic only**.
- Seven-day follow-up excludes the alert day and requires all seven future days; last six evaluation dates have unavailable follow-up.
- These are different severity scales and should not be directly compared.
- PI80 nominal coverage is 80%; both systematic overcoverage and undercoverage matter.
- Rolling 7/30-day coverage is the unweighted mean of daily coverage rates; DST dates may have 23/25 hours.
- Rolling daily coverage variability is the sample SD of daily coverage percentages in the window.
- Reference P95 thresholds are estimated only from available reference rolling-window observations.
- High rolling variability and high absolute calibration deviation are different diagnostics.
- No true/false-positive classification is made without independent deterioration labels.

## Category summary

```
      category  days  median_CRPS  median_fixed_exceedance  median_rMAE  median_CRPSS  median_PI80_coverage  median_past7_volatility  high_realized_volatility_pct  relative_alert_days  median_rolling30_coverage  median_rolling30_daily_sd_pp  rolling30_deviation_alert_days  rolling30_variability_alert_days
    Both alert    73       15.619                    4.528        0.971         0.112                 0.667                   30.356                        94.521                   19                      0.794                        15.286                               0                                 8
    Fixed only     9       11.401                    0.310        0.688         0.127                 0.750                   34.917                        88.889                    3                      0.799                        15.426                               0                                 1
Neither alerts   145        7.741                    0.000        0.563         0.063                 0.917                   30.282                        66.897                   25                      0.801                        15.152                               0                                21
```

## Nine fixed-only days

Total: 9. These are **not** automatically false alarms.
See `tables/experiment_14b_nine_fixed_only_days.csv` for each date, CRPS severity, volatility, relative metrics, coverage, and follow-up.

## Figures
- `experiment_14b_crps_alert_severity.png`
- `experiment_14b_nine_removed_alerts.png`
- `experiment_14b_volatility_vs_crps_by_alert.png`
- `experiment_14b_followup_crps.png`
- `experiment_14b_coverage_rolling.png`
- `experiment_14b_coverage_stability.png`
- `experiment_14b_nine_removed_coverage.png`

## Coverage stability thresholds and counts

```
                             metric  reference_p95  evaluation_alert_days  evaluation_valid_days
 coverage_rolling7_deviation_abs_pp         16.429                      0                    227
              coverage_daily_sd7_pp         21.459                     19                    227
coverage_rolling30_deviation_abs_pp         12.271                      0                    227
             coverage_daily_sd30_pp         18.157                     30                    227
```

## Limitations

- Observed association is not causation; alert categories are partly defined by CRPS itself.
- Seven-day follow-up is not independent ground truth, and overlapping windows create dependence.
- Apparent persistence alone cannot establish model-specific deterioration.
- Future evaluation should use prespecified, preferably independent events and compare alert episodes, lead time, missed events, and false alarms.
