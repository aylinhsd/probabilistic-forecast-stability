# Experiment 10 — Detector Evaluation and Early-Warning Assessment

## Objective
Compare the Experiment 09 detectors under a common **notification policy**, and retrospectively describe whether forecast-width notifications precede elevated future daily CRPS. No LEAR retraining or new detector fitting.

## Inputs and periods
- Daily baseline: `market_regime_results/health_baseline/daily_health_baseline.csv`.
- Existing Experiment 09 daily detector signals and settings.
- Reference: July 1–December 31, 2025 (184 days); evaluation: January 1–October 13, 2026 (286 days).
- Daily CRPS reference p95 = **10.9461**; 30-day rolling CRPS reference p95 = **6.4858**.

## Methods
1. Retain Experiment 09 alarm events exactly as saved, including their exploratory sensitivity settings. **Do not optimize settings on evaluation outcomes.**
2. Count raw alarm days and contiguous runs; note that event-based detectors and persistent fixed-threshold alarms are structurally different.
3. Apply a **common 7-calendar-day notification cooldown** to each detector/metric/configuration: keep first alarm, then suppress subsequent notifications until at least seven days after the previous notification. This is an illustrative operational rule, **not a calibrated optimal policy**. The underlying detector still runs continuously and is never reset by this notification policy.
4. Identify entries into the 30-day rolling CRPS above its reference p95. These are **screening onsets, not independent ground-truth degradation events**. Do not interpret their timing as a validated detector delay.
5. For **width80** notifications, examine whether *any* daily realized CRPS exceeds the reference **daily** p95 in the next 7, 14, or 30 calendar days, **excluding the notification date**. Require the entire future horizon; otherwise censor the observation. Compare with the same future-outcome prevalence among **all eligible evaluation days**.

## Critical limitations
- Alarm days, detector events, and notifications are different quantities. Seven-day cooldown reduces repeated notifications but does not establish statistical false-alarm control.
- **Future CRPS exceedance is an exploratory proxy, not independently labeled degradation**. It is also a relatively common event during 2026; high event-following fractions may arise from background prevalence.
- Notifications can cluster; their future windows overlap and observations are not independent. This analysis does **not** estimate calibrated predictive precision, recall, significance, or a causal warning lead time.
- Width80 can reflect legitimate market uncertainty; CRPS also responds to changes in market difficulty. Six months of reference cannot remove seasonal confounding.
- Daily width is a daily aggregate; this analysis does not establish hour-level forecast issue-time lead times.
- Sensitivity labels from Experiment 09 are historical script labels: `low` is the most responsive configuration and `high` is the most conservative. They should not be read as verified statistical sensitivity levels.
- The Page-Hinkley and ADWIN implementations have different internal reset behavior from the custom one-sided CUSUM and fixed threshold; this comparison does not calibrate equal false-alarm rates.

## Saved outputs
- `experiment_10_notification_summary.csv`: alarm-day and cooldown notification counts.
- `experiment_10_notification_dates.csv`: notification dates and distance to next rolling-CRPS screening onset.
- `experiment_10_width_future_outcomes.csv`: individual width notifications and complete-horizon outcomes.
- `experiment_10_width_future_summary.csv`: grouped future-outcome rates, with all-day baseline.
- `experiment_10_future_base_rates.csv`: reference future-outcome prevalence among all eligible evaluation days.
- `experiment_10_alert_burden.png/.pdf`: raw alarm days versus common notification policy.
- `experiment_10_width_notifications_vs_crps.png/.pdf`: notification dates relative to realized CRPS.

## Results and interpretation
Review the generated CSV tables and figures before drawing conclusions or selecting a detector. In particular, compare each width detector's future-exceedance fraction with the **all-day baseline rate** and consider the number of eligible notifications. Do not treat the first warning before March as a proven lead time.
