# Experiment 09 — Comparative Change-Detection Methods

## Purpose
Compare fixed daily reference thresholds, River Page-Hinkley, a one-sided standardized CUSUM, and River ADWIN on LEAR daily CRPS and daily 80% interval width. This is monitoring research, not model retraining.

## Data and chronology
- Reference: 2025-07-01 through 2025-12-31 (184 days).
- Evaluation: 2026-01-01 through 2026-10-13 (286 days).
- Input: `market_regime_results/health_baseline/daily_health_baseline.csv`.
- All detectors process dates chronologically. River detectors warm up on reference observations before evaluation; the CUSUM mean and standard deviation and fixed p95 thresholds are estimated from reference observations only.
- River Page-Hinkley uses its own running-mean formulation; CUSUM is a distinct reference-standardized, upper-sided algorithm. ADWIN is a two-sided adaptive-window detector.

## Detector settings
- Page-Hinkley: `delta=0.005`, `alpha=0.999`, `min_instances=30`; thresholds 25, 50, 100.
- ADWIN: delta 0.01, 0.002, 0.0005.
- Upper CUSUM: standardized reference mean, allowance k=0.25/0.5/0.75 standard deviations and alarm h=3/5/8; reset after each alarm.
- Fixed threshold: reference **daily** 95th percentile. It is not the 30-day rolling threshold from Experiment 01.
- All configurations are exploratory sensitivity settings, **not optimized or calibrated to a specified false-alarm rate**. Page-Hinkley and ADWIN may signal decreases as well as increases depending on implementation; only upper CUSUM is explicitly one-sided.

## Outputs
- `experiment_09_daily_detector_signals.csv`: every day and configuration, including warm-up alarms.
- `experiment_09_detector_summary.csv`: evaluation alarm-day counts, contiguous alert episodes, and first/last alarms.
- `experiment_09_settings.csv`: detector parameters.
- `experiment_09_crps_detector_timeline.png/.pdf` and `experiment_09_width80_detector_timeline.png/.pdf`.

## Interpretation cautions
- Daily CRPS requires realized outcomes; daily width uses forecast information, but the daily aggregate is fully available only after all forecasts included in that delivery date have been issued. This analysis does **not** establish exact intraday lead time.
- Different detectors have different reset behavior. Alarm counts are not directly comparable measures of false-positive rates.
- There is no independently verified ground-truth degradation onset. The March 11, 2026 rolling-CRPS threshold crossing from Experiment 01 is a prior monitoring alarm, not ground truth.
- Seasonal differences and market regimes may produce alarms without model degradation. Six months of reference data cannot fully establish annual seasonality.
- Do not infer statistical significance, validated early warning, or superiority from one evaluation period.

## Results
See `experiment_09_detector_summary.csv` and the two timelines. Conclusions should be written after inspecting the actual outputs.
