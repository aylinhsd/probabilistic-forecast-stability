# Experiment 04 — Six-Indicator Health Monitoring

## Objective

Compare probabilistic LEAR v3 health indicators with
historical calibration-period reference distributions.

## Monitoring periods

Calibration: 2025-07-01 to 2025-12-31
Evaluation: 2026-01-01 to 2026-10-13

## Reference statistics

         indicator                            column  calibration_mean  calibration_p05  calibration_p95  evaluation_mean  evaluation_days
              CRPS               health_rolling_crps            5.6355           5.0350           6.4858           9.4458              286
               MAE                health_rolling_mae            7.6601           6.9028           8.8430          12.9904              286
              RMSE               health_rolling_rmse           11.5339          10.1953          13.7927          19.5090              286
      Coverage 80%                rolling_coverage80            0.8668           0.8193           0.9128           0.8231              286
Interval score 80%   health_rolling_interval_score80           38.0476          33.9919          42.9138          63.3190              286
PIT mean deviation health_rolling_pit_mean_deviation            0.1226           0.0931           0.1512           0.1241              286

## Interpretation

Thresholds are exploratory reference quantiles, not
validated statistical alarm limits.

Higher CRPS, MAE, RMSE and interval scores indicate
worse forecast performance.

Coverage is evaluated in both directions and relative
to its nominal 80% target.

The PIT panel is a preliminary diagnostic only.


PIT limitation:
The dashboard uses rolling PIT mean deviation as a
preliminary diagnostic, not a complete PIT drift test.

A mean near 0.5 does not guarantee uniform PIT values.
A proper PIT histogram and distributional calibration
test require the individual hourly PIT observations.


## Limitations

- Health indicators require realized electricity prices.
- Rolling windows overlap and are statistically dependent.
- Six months of calibration may not represent all seasons.
- Market volatility can affect forecast difficulty.
- Indicator exceedances are not independent events.
- PIT distributional calibration remains to be evaluated.
- A composite health index has not yet been calibrated.
