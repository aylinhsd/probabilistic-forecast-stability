# Experiment 01 — Width Early Warning

## Objective

Investigate whether the 80% prediction interval width
crosses a historical threshold before the first rolling
CRPS health alarm.

## Calibration

Period: 2025-07-01 to 2025-12-31

Width threshold (daily 95th percentile):
37.2360 EUR/MWh

CRPS threshold (rolling 95th percentile):
6.4858 EUR/MWh

## Results

First CRPS alarm:
2026-03-11

First earlier width alert:
2026-01-20

Last earlier width alert:
2026-03-10

Lead time from last earlier width alert:
1 days

Width alert days before CRPS alarm:
6

## Interpretation

This experiment examines temporal ordering only.
An earlier width threshold crossing is not sufficient
evidence of predictive early-warning performance.

The width threshold is calibrated on daily forecast
interval widths, while the CRPS threshold is calibrated
on rolling health values.

Forecast-time availability of width must be verified
using forecast issue timestamps for a strict
operational early-warning evaluation.

## Limitations

- Retrospective evaluation.
- Only one first CRPS alarm onset examined.
- Thresholds are exploratory historical quantiles.
- Daily width may reflect changing market regimes.
- Alarm precision and false-positive rates are not
  estimated by this experiment.
