# Experiment 03 — Fixed vs Adaptive Width Monitoring

Calibration: July-December 2025
Evaluation: January 1-October 13, 2026

Fixed width threshold: 37.2360 EUR/MWh
Adaptive threshold: 95th percentile of the preceding 90 days
CRPS health threshold: 6.4858 EUR/MWh
First CRPS health alarm: 2026-03-11

## Results

  method  total_alert_days  alert_episodes  alert_days_before_onset  alert_days_in_prior_7_days last_prior_alert  lead_days_from_last_prior_alert
   fixed               181              20                        6                           3       2026-03-10                                1
adaptive                43              24                        7                           4       2026-03-10                                1

## Methodological notes

- The adaptive threshold excludes the current delivery day.
- Both methods use the same evaluation period.
- CRPS health alarms require realized prices.
- Width is forecast-side information, but operational
  timing must be verified against forecast issue times.
- Fewer alerts do not automatically mean better detection.
- Only one CRPS onset is available for this comparison.
- The adaptive threshold may normalize persistent changes.
- This is a retrospective exploratory comparison.
