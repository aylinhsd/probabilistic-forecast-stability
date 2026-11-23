# Experiment 06 — Interval Reliability Tests

## Objective

Evaluate probabilistic LEAR v3 prediction interval
calibration and temporal clustering of violations.

Both overcoverage and undercoverage are considered
undesirable deviations from nominal calibration.

## Periods

Reference: 2025-07-01 to 2025-12-31
Evaluation: 2026-01-01 to 2026-10-13

## Intervals

Central 80%, 85%, and 90% prediction intervals.

Intervals are obtained from the original forecast
quantiles, using interpolation where necessary.

## Statistical tests

Kupiec UC: unconditional coverage.
Christoffersen IND: first-order violation independence.
Christoffersen CC: joint conditional coverage.

Nominal violation probabilities are respectively
20%, 15%, and 10%.

## Overall results

group_type group     period  interval    n  violations  nominal_coverage  observed_coverage  signed_coverage_error_pp  absolute_coverage_error_pp    direction  n00  n01  n10  n11  p_violation_after_no_violation  p_violation_after_violation     LR_UC    p_UC     LR_IND  p_IND      LR_CC  p_CC  UC_reject_5pct IND_reject_5pct CC_reject_5pct
   Overall   All Evaluation        80 6863        1240              0.80            0.81932                   1.93210                     1.93210 Overcoverage 5096  526  526  714                         0.09356                      0.57581  16.42046 0.00005 1300.15488    0.0 1316.57534   0.0            True            True           True
   Overall   All Evaluation        85 6863         894              0.85            0.86974                   1.97363                     1.97363 Overcoverage 5572  396  396  498                         0.06635                      0.55705  21.78033 0.00000 1168.83491    0.0 1190.61524   0.0            True            True           True
   Overall   All Evaluation        90 6863         597              0.90            0.91301                   1.30118                     1.30118 Overcoverage 5987  278  278  319                         0.04437                      0.53434  13.44399 0.00025  955.75856    0.0  969.20255   0.0            True            True           True
   Overall   All  Reference        80 4417         569              0.80            0.87118                   7.11795                     7.11795 Overcoverage 3598  249  249  320                         0.06473                      0.56239 155.37542 0.00000  768.43623    0.0  923.81165   0.0            True            True           True
   Overall   All  Reference        85 4417         380              0.85            0.91397                   6.39688                     6.39688 Overcoverage 3861  175  175  205                         0.04336                      0.53947 163.34692 0.00000  625.36307    0.0  788.70999   0.0            True            True           True
   Overall   All  Reference        90 4417         230              0.90            0.94793                   4.79285                     4.79285 Overcoverage 4068  118  118  112                         0.02819                      0.48696 134.30514 0.00000  413.49222    0.0  547.79736   0.0            True            True           True

## Day/night results

group_type group     period  interval    n  violations  nominal_coverage  observed_coverage  signed_coverage_error_pp  absolute_coverage_error_pp     direction  n00  n01  n10  n11  p_violation_after_no_violation  p_violation_after_violation     LR_UC    p_UC    LR_IND  p_IND     LR_CC  p_CC  UC_reject_5pct IND_reject_5pct CC_reject_5pct
  DayNight   Day Evaluation        80 3432         805              0.80            0.76544                  -3.45571                     3.45571 Undercoverage 2309  317  317  488                         0.12072                      0.60621  24.60308 0.00000 724.52012    0.0 749.12319   0.0            True            True           True
  DayNight   Day Evaluation        85 3432         630              0.85            0.81643                  -3.35664                     3.35664 Undercoverage 2527  274  274  356                         0.09782                      0.56508  28.65643 0.00000 615.24295    0.0 643.89938   0.0            True            True           True
  DayNight   Day Evaluation        90 3432         445              0.90            0.87034                  -2.96620                     2.96620 Undercoverage 2794  192  192  253                         0.06430                      0.56854  30.97979 0.00000 613.80240    0.0 644.78220   0.0            True            True           True
  DayNight Night Evaluation        80 3431         435              0.80            0.87321                   7.32148                     7.32148  Overcoverage 2759  236  236  199                         0.07880                      0.45747 128.15246 0.00000 356.79180    0.0 484.94427   0.0            True            True           True
  DayNight Night Evaluation        85 3431         264              0.85            0.92305                   7.30545                     7.30545  Overcoverage 3022  144  144  120                         0.04548                      0.45455 169.79050 0.00000 325.93957    0.0 495.73007   0.0            True            True           True
  DayNight Night Evaluation        90 3431         152              0.90            0.95570                   5.56980                     5.56980  Overcoverage 3184   94   94   58                         0.02868                      0.38158 146.29087 0.00000 189.45297    0.0 335.74384   0.0            True            True           True
  DayNight   Day  Reference        80 2208         381              0.80            0.82745                   2.74457                     2.74457  Overcoverage 1681  146  145  235                         0.07991                      0.61842  10.78053 0.00103 507.48506    0.0 518.26559   0.0            True            True           True
  DayNight   Day  Reference        85 2208         267              0.85            0.87908                   2.90761                     2.90761  Overcoverage 1833  107  107  160                         0.05515                      0.59925  15.50849 0.00008 440.56810    0.0 456.07659   0.0            True            True           True
  DayNight   Day  Reference        90 2208         166              0.90            0.92482                   2.48188                     2.48188  Overcoverage 1965   76   76   90                         0.03724                      0.54217  16.38801 0.00005 299.97909    0.0 316.36709   0.0            True            True           True
  DayNight Night  Reference        80 2209         188              0.80            0.91489                  11.48936                    11.48936  Overcoverage 1909  111  111   77                         0.05495                      0.40957 221.16033 0.00000 171.43697    0.0 392.59730   0.0            True            True           True
  DayNight Night  Reference        85 2209         113              0.85            0.94885                   9.88456                     9.88456  Overcoverage 2020   75   75   38                         0.03580                      0.33628 218.03301 0.00000 100.82502    0.0 318.85803   0.0            True            True           True
  DayNight Night  Reference        90 2209          64              0.90            0.97103                   7.10276                     7.10276  Overcoverage 2101   43   43   21                         0.02006                      0.32812 167.29916 0.00000  77.04024    0.0 244.33940   0.0            True            True           True

## Interpretation

Signed coverage error is observed minus nominal coverage.

Positive error = overcoverage.
Negative error = undercoverage.

The IND test compares the probability of a violation
after a previous violation with the probability after
a previous non-violation.

## Limitations

- Standard chi-square p-values are asymptotic.
- Hourly electricity price errors can exhibit serial
  dependence, seasonality, and common market shocks.
- Standard test rejections are screening diagnostics.
- IND is a first-order Markov diagnostic, not a
  complete test of all temporal dependence.
- In day/night subsets, consecutive rows may be
  separated by several delivery hours.
- In each local-hour subset, transitions are generally
  between consecutive days, not adjacent hours.
- Daylight-saving transitions affect local-hour counts.
- Multiple tests across hours and interval levels
  increase the risk of false-positive findings.
- A non-rejection does not prove correct calibration.
- Test p-values must not be interpreted as health scores.
- This experiment does not establish causal model drift.

## Next steps

1. Examine violation clustering and hourly differences.
2. Consider dependence-aware uncertainty estimates.
3. Investigate calendar and market regime effects.
4. Design a health index without double-counting
   strongly related coverage statistics.
