# Experiment 11 — Sustained Deterioration and Early-Warning Lead-Time Analysis

## Purpose
Evaluate whether forecast-width notifications precede a sustained rolling-CRPS screening onset, rather than merely precede an isolated high-error day.

## Data and method
- Reference: dates through 2025-12-31; evaluation: 2026-01-01 to 2026-10-13.
- Daily CRPS and the previously calculated 30-day rolling CRPS are read from the saved health baseline.
- Width notification dates come from Experiment 10, including its common seven-day cooldown.
- Reference rolling-CRPS 95th percentile: **6.485803** (computed from nonmissing reference rolling observations).
- Primary persistence requirement: **3 consecutive calendar days** above the reference threshold.
- Screening onset is dated to the first above-threshold day of a qualifying run; this is known only when the persistence requirement is confirmed. Thus onset dating is retrospective, not a real-time alarm date.
- Sensitivity checks use persistence requirements of 1, 3, 5, and 7 days.
- Pre-onset windows: 7, 14, 30, and 60 days; notifications on the onset day are **not** advance warnings.
- Detector settings are inherited from Experiment 09; they are not tuned using evaluation outcomes.

## Results
- Primary qualifying screening runs: **1**.
- Onset 2026-03-11, confirmation 2026-03-13, end 2026-10-13, length 217 days.

### Medium-setting width notifications

| Detector | Notifications | Before first onset | On/after onset | Onsets warned within 7d | 14d | 30d |
|---|---:|---:|---:|---:|---:|---:|
| ADWIN | 3 | 1 | 2 | 0 | 0 | 1 |
| CUSUM | 30 | 2 | 28 | 1 | 1 | 1 |
| FixedP95 | 33 | 2 | 31 | 1 | 1 | 1 |
| PageHinkley | 9 | 2 | 7 | 0 | 0 | 1 |

## Interpretation and limitations
- This is a **screening definition**, not independently labeled model failure or concept drift.
- The reference period is only six months, and seasonality/market-regime changes may affect the threshold.
- A long run can be one episode; it does not supply multiple independent deterioration onsets.
- A warning before an onset is temporal association, not proof that width predicts deterioration.
- Notifications during an already ongoing deterioration are not counted as advance warnings.
- We cannot estimate meaningful detection sensitivity or false-positive rates from one onset.
- A full-day width aggregate is usable only once the forecasts contributing to that day have been issued; issue-time availability must be checked before operational deployment.
- The 7-day cooldown is an illustrative policy, not a calibrated optimal setting.

## Outputs
- Tables: experiment_11_rolling_onsets.csv, experiment_11_pre_onset_windows.csv, experiment_11_width_notification_classification.csv, experiment_11_detector_lead_summary.csv.
- Figures: experiment_11_rolling_onsets and experiment_11_width_pre_onset_timeline (PNG/PDF).
