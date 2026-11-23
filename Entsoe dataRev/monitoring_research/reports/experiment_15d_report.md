# Experiment 15d — Historical Alert Investigation and Validation

## Goal and source

Analyse original chronological LEAR v3 monitoring results **before** interpreting synthetic detector rankings. Inputs: `ENTSOE_Probabilistic_LEAR_v3_predictions.csv`; separate point-forecast MAE reference: `ENTSOE_LEAR_v3_predictions.csv`.

**Time split:** Fit = 2025-07-01 to 2025-12-07 (160 days); remaining reference = 2025-12-08 to 2026-02-28 (83 days); historical evaluation = 2026-03-01 to 2026-10-13 (227 days).

Historical timestamps are NEVER shuffled, synthetic perturbations occur only in separate copied arrays.

## 1. Real alerts and grouping

```text
      method  evaluation_raw_alert_days  evaluation_notifications  evaluation_episodes  evaluation_days  configured_cooldown_days           historical_false_alarm_rate
   Fixed P95                        147                        26                   10              227                         7 NOT IDENTIFIABLE WITHOUT GROUND TRUTH
       CUSUM                        141                        26                    5              227                         7 NOT IDENTIFIABLE WITHOUT GROUND TRUTH
Page-Hinkley                        121                        26                    4              227                         7 NOT IDENTIFIABLE WITHOUT GROUND TRUTH
        EWMA                        157                        27                    4              227                         7 NOT IDENTIFIABLE WITHOUT GROUND TRUTH
       ADWIN                          2                         2                    2              227                         7 NOT IDENTIFIABLE WITHOUT GROUND TRUTH
```

'Raw alert days' are days where at least one monitored signal fired. Notifications use a seven-day cooldown; descriptive episodes combine raw alert days separated by up to 3 clear days. Episodes are NOT confirmed deterioration events.

## Combined incidents across all detectors

Total combined descriptive incidents: 3. `experiment_15d_cross_detector_incidents.csv` clusters alerts across different detector methods into one overall timeline. These remain UNLABELED.

## 2. Context around episodes

{
  "high_volatility_context_only": 19,
  "no_strong_contextual_evidence": 5,
  "performance_deviation_present_unverified_cause": 1
}

For each episode, `experiment_15d_episodes.csv` reports preceding 14-day means and the episode plus following 14 days: CRPS, PI80 coverage, realized volatility, a point-forecast MAE comparison and the originating monitoring signal(s). Evidence categories are heuristic, descriptive, and **not ground truth**. The separate point model is not a probabilistic CRPS benchmark.

## 3. Real-world true and false alarms

Cannot be established with current records alone. We have no independently adjudicated model-failure labels. High volatility is contextual evidence, not proof alerts are false; high CRPS is a monitoring signal, not independent verification. Assess against documented model failures, expert review, and prospective tests of model updates.

## 4. Controlled tests on original chronological data (not block-mixed)

For transparency, artificial signal-level sustained, gradual, and five-day temporary changes were added to COPIES of the unshuffled final 83 reference days. Tested injected and paired unmodified sequences with the same detector configurations; incremental detection is counted only if the paired unmodified series did not alert in the same window.

```text
      method  tests  detection_pct  paired_baseline_alert_pct  incremental_detection_pct  median_detected_delay_days
       ADWIN    108           72.2                       66.7                        5.6                        17.0
       CUSUM    108          100.0                      100.0                        0.0                         7.0
        EWMA    108          100.0                      100.0                        0.0                         7.0
   Fixed P95    108           34.3                        0.0                       34.3                        11.0
Page-Hinkley    108          100.0                      100.0                        0.0                         1.0
```

This is not a genuine no-change reference: chronological reference-tail days may already include changes. Thus paired unmodified alerts are NOT established false alarms. These tests have only 3 onset positions and are a sensitivity check, not an independent detector ranking.

## 5. Conclusions and follow-up

- Examine which real episodes coincide with larger CRPS, poor PI80 calibration, or benchmark skill loss; do not label them causal or confirmed model deterioration.
- Distinguish raw alert days, post-cooldown notifications, and grouped episodes.
- Analyse all held-out 15c synthetic-null and controlled tests to compare sensitivity under common budget.
- Add forecast-level perturbations (Experiment 15b) for coherent CRPS/coverage changes.
- Obtain independently documented degradation periods or expert labels before computing real false alarm rates.
- Do not automate retraining or claim a best detector from this analysis alone.
- The detector named `Fixed P95` reuses the Experiment 15c family label, but its selected `q=0.99` setting corresponds to a **99th-percentile** threshold. Do not describe its current 15d cutoff as P95.
- ADWIN is omitted unless independently calibrated; the original 15c run had no `river` ADWIN results.

## Output files

- `tables/experiment_15d_daily.csv`: actual chronological monitoring indicators.
- `tables/experiment_15d_alerts.csv`: individual alert days and signals.
- `tables/experiment_15d_episodes.csv`: grouped events and pre/post evidence.
- `tables/experiment_15d_method_summary.csv`: operational burden.
- `tables/experiment_15d_monthly.csv`: monthly conditions and notifications.
- `tables/experiment_15d_chronological_tests.csv`: controlled perturbations on original order.
- `tables/experiment_15d_chronological_summary.csv`: paired test outcomes.
- `figures/experiment_15d_*.png`: five diagnostic figures.
