# Probabilistic LEAR v3 — Health and Early-Warning Monitoring

## Research objective

Evaluate whether forecast-time indicators can provide
early warning of deterioration in probabilistic electricity
price forecasting performance.

## Forecasting model

Model: Probabilistic LEAR v3
Market: Italy North (IT-North)
Forecast test period: 2025-07-01 to 2026-10-13
Hourly forecasts: 11,280
Test days: 470

The forecasting model is not retrained in this study.

## Monitoring design

Fixed calibration:
2025-07-01 to 2025-12-31 (184 days)

Evaluation:
2026-01-01 to 2026-10-13 (286 days)

Adaptive monitoring will be investigated separately.

## Fixed health thresholds

Rolling CRPS upper threshold: 6.4858
Rolling MAE upper threshold: 8.8430
Rolling RMSE upper threshold: 13.7927
Rolling 80% interval score upper threshold: 42.9138

Rolling 80% coverage historical lower threshold: 0.8193
Rolling 80% coverage historical upper threshold: 0.9128
Nominal 80% coverage: 0.8000

Thresholds are exploratory calibration-period quantiles.

## Detected health events

CRPS:
2026-03-11 to 2026-10-13
217 alarm days, 1 episode

Historical low coverage:
First onset 2026-03-26
158 alarm days, 4 episodes

Below nominal 80% coverage:
First onset 2026-03-29
113 alarm days, 7 episodes

## Important limitations

- Health indicators require realized prices.
- Rolling indicators use overlapping 30-day windows.
- Calibration covers only six months, not a full seasonal cycle.
- Higher realized volatility is associated with higher errors.
- Threshold exceedance does not prove model deterioration.
- Early-warning performance has not yet been validated.
- Realized same-day volatility is not available at forecast issue time.

## Next experiments

1. Forecast interval width before March 11, 2026.
2. Fixed versus adaptive monitoring thresholds.
3. Calendar effects: weekdays, weekends, Italian holidays.
4. Evaluate warning lead times and false alarms.
