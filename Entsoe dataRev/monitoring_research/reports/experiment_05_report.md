# Experiment 05 — PIT Calibration

## Objective

Investigate distributional calibration of probabilistic
LEAR v3 forecasts using approximate hourly PIT values.

## Periods

Calibration: 2025-07-01 to 2025-12-31
Evaluation: 2026-01-01 to 2026-10-13

## Quantile reconstruction

Number of quantiles: 101
Minimum quantile level: 0.010
Maximum quantile level: 0.990

PIT values are reconstructed by linear interpolation
between predicted quantiles.

Observations outside the available quantile range are
assigned tail probability midpoints. Consequently, PIT
values are approximations and the tails are censored.

Duplicate forecast quantile values are handled using
midpoints of their probability ranges.

Non-monotonic quantile forecasts are excluded rather
than silently corrected.

## Reconstruction status

     period   pit_status  count
Calibration interpolated   4370
Calibration   lower_tail     40
Calibration   upper_tail      7
 Evaluation interpolated   6708
 Evaluation   lower_tail    102
 Evaluation   upper_tail     53

## Overall PIT statistics

     period      n  pit_mean  pit_variance  mean_abs_deviation_from_0_5  tail_fraction_10pct  center_fraction_20pct  ks_statistic  ks_pvalue_naive
Calibration 4417.0    0.4903        0.0669                       0.2197               0.1288                 0.2461        0.0558              0.0
 Evaluation 6863.0    0.4879        0.0787                       0.2423               0.1807                 0.2068        0.0281              0.0

## Day/night PIT statistics

     period day_night      n  pit_mean  pit_variance  mean_abs_deviation_from_0_5  tail_fraction_10pct  center_fraction_20pct  ks_statistic  ks_pvalue_naive
Calibration       Day 2208.0    0.4719        0.0744                       0.2340               0.1726                 0.2287        0.0639              0.0
Calibration     Night 2209.0    0.5087        0.0588                       0.2053               0.0851                 0.2635        0.0780              0.0
 Evaluation       Day 3432.0    0.4600        0.0860                       0.2559               0.2346                 0.1955        0.0620              0.0
 Evaluation     Night 3431.0    0.5159        0.0699                       0.2288               0.1268                 0.2180        0.0546              0.0

## Interpretation

For an ideal continuous calibrated predictive distribution,
PIT values should be uniformly distributed on [0, 1].

Expected PIT mean: 0.5
Expected PIT variance: 1/12 (approximately 0.0833)
Expected outer 10% tail fraction: 0.20
Expected central 20% fraction: 0.20

Excessive tail concentration may indicate underdispersion.
Excessive central concentration may indicate overdispersion.

Mean PIT alone cannot establish calibration.

## Important limitations

- Reconstructed PIT values are approximate.
- Tail probabilities outside the available quantile grid
  cannot be recovered exactly.
- Tail midpoint assignment creates artificial point masses.
- The reported KS p-values assume independent samples;
  they must NOT be used as formal rejection decisions
  for serially dependent hourly electricity forecasts.
- Quantile-based PIT reconstruction assumes a continuous
  interpolated forecast distribution.
- Day/night groups are defined as 06:00–17:59 and
  18:00–05:59, Italian local time.
- Local-hour groups contain daylight-saving transitions.
- PIT behavior can vary with season and market conditions.
- Calibration and evaluation periods differ in duration.
- Formal dependence-aware calibration testing is still
  required before incorporating PIT into a health index.

## Next experiment

Experiment 06: Kupiec and Christoffersen interval
coverage and violation-independence diagnostics,
including both undercoverage and overcoverage.
