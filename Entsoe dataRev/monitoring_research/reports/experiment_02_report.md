# Experiment 02 — Width Warning Episodes

## Objective

Evaluate whether episodes of unusually wide forecast
intervals precede new rolling-CRPS health-alarm onsets.

## Calibration

Calibration: 2025-07-01 to 2025-12-31
Evaluation: 2026-01-01 to 2026-10-13

Width threshold: 37.2360 EUR/MWh
CRPS threshold: 6.4858 EUR/MWh

## Event definition

A width episode is a consecutive sequence of days
with daily width above the calibration threshold.

A CRPS onset is a transition from below to above
the rolling-CRPS calibration threshold.

For each width episode, warning timing is measured
from its first day.

## Results

Width-alert episodes: 20
CRPS health onsets: 1

 horizon_days  width_episodes  episodes_followed_by_crps_onset  episode_hit_fraction  crps_onsets
            1              20                                0                  0.00            1
            3              20                                1                  0.05            1
            7              20                                2                  0.10            1
           14              20                                2                  0.10            1

## Limitations

- Only one CRPS onset may be present.
- Warning horizons are exploratory.
- Width episodes occurring during an already active
  health alarm are not excluded from this table.
- This is not an estimate of a prospective false-alarm rate.
- CRPS uses overlapping rolling windows.
- Same-day realized CRPS is unavailable at forecast issue.
- Forecast issue timestamps require separate verification.
