# Experiment 07 — Calendar and Intraday Stability Analysis

## Objective

Examine whether LEAR forecasting performance varies across
Italian local delivery hours, weekdays, weekends, national
public holidays, bridge days and seasons.

This is an exploratory diagnostic study, not a retraining
or model-selection experiment.

## Data and periods

- Input: `ENTSOE_Probabilistic_LEAR_v3_predictions.csv`
- Calibration: 2025-07-01 to 2025-12-31
- Evaluation: 2026-01-01 to 2026-10-13
- Valid hourly observations: 11280
- Excluded malformed prediction intervals: 0

## Metrics

- MAE: mean absolute error of the median forecast.
- Coverage80: fraction of realized prices inside Q0.100–Q0.900.
- Coverage error: (observed coverage - 0.80) × 100 percentage points.
- Width80: Q0.900 - Q0.100.
- Interval score80: width plus penalties for observations
  falling outside the interval.

Negative coverage error indicates undercoverage.
Positive coverage error indicates overcoverage.

## Overall performance

| period      |   observations |   days |     mae |   coverage80 |   width80 |   interval_score80 |   coverage_error_pp |
|:------------|---------------:|-------:|--------:|-------------:|----------:|-------------------:|--------------------:|
| Calibration |           4417 |    184 |  7.5064 |       0.8712 |   28.3305 |            37.4165 |              7.118  |
| Evaluation  |           6863 |    286 | 13.3962 |       0.8193 |   42.2591 |            65.3875 |              1.9321 |

## Calendar day types

| period      | day_type        |   observations |   days |     mae |   coverage80 |   width80 |   interval_score80 |   coverage_error_pp |
|:------------|:----------------|---------------:|-------:|--------:|-------------:|----------:|-------------------:|--------------------:|
| Calibration | Holiday         |            120 |      5 |  8.1843 |       0.8167 |   27.3998 |            34.8205 |              1.6667 |
| Calibration | Regular weekday |           3072 |    128 |  7.1438 |       0.8743 |   27.7415 |            35.1906 |              7.4349 |
| Calibration | Weekend         |           1225 |     51 |  8.3494 |       0.8686 |   29.8987 |            43.253  |              6.8571 |
| Evaluation  | Bridge day      |             72 |      3 |  6.3975 |       0.9306 |   32.9532 |            33.8129 |             13.0556 |
| Evaluation  | Holiday         |            192 |      8 | 23.137  |       0.5938 |   42.8823 |           117.999  |            -20.625  |
| Evaluation  | Regular weekday |           4704 |    196 | 12.0605 |       0.831  |   41.0962 |            58.6319 |              3.0995 |
| Evaluation  | Weekend         |           1895 |     79 | 15.9908 |       0.809  |   45.4361 |            78.026  |              0.8971 |

## Seasons

| period      | season   |   observations |   days |     mae |   coverage80 |   width80 |   interval_score80 |   coverage_error_pp |
|:------------|:---------|---------------:|-------:|--------:|-------------:|----------:|-------------------:|--------------------:|
| Calibration | Autumn   |           2185 |     91 |  7.8532 |       0.87   |   29.0992 |            38.7624 |              7.0023 |
| Calibration | Summer   |           1488 |     62 |  7.7887 |       0.8421 |   27.5568 |            38.7238 |              4.207  |
| Calibration | Winter   |            744 |     31 |  5.9232 |       0.9328 |   27.6203 |            30.8496 |             13.2796 |
| Evaluation  | Autumn   |           1032 |     43 | 14.0464 |       0.8614 |   54.9598 |            71.4434 |              6.1434 |
| Evaluation  | Spring   |           2207 |     92 | 15.6965 |       0.7839 |   43.0746 |            76.0929 |             -1.613  |
| Evaluation  | Summer   |           2208 |     92 | 14.6696 |       0.7989 |   45.0226 |            70.787  |             -0.1087 |
| Evaluation  | Winter   |           1416 |     59 |  7.3512 |       0.8757 |   27.4224 |            35.8686 |              7.5706 |

## Intraday time blocks

| period      | time_block      |   observations |   days |     mae |   coverage80 |   width80 |   interval_score80 |   coverage_error_pp |
|:------------|:----------------|---------------:|-------:|--------:|-------------:|----------:|-------------------:|--------------------:|
| Calibration | Night 00-05     |           1105 |    184 |  4.0847 |       0.9258 |   19.3063 |            21.3892 |             12.5792 |
| Calibration | Morning 06-11   |           1104 |    184 |  8.6346 |       0.8451 |   30.3929 |            41.2972 |              4.5109 |
| Calibration | Afternoon 12-17 |           1104 |    184 | 10.9737 |       0.8098 |   35.0386 |            53.717  |              0.9783 |
| Calibration | Evening 18-23   |           1104 |    184 |  6.3357 |       0.904  |   28.5921 |            33.2773 |             10.3986 |
| Evaluation  | Night 00-05     |           1715 |    286 |  7.132  |       0.877  |   29.2466 |            36.1209 |              7.6968 |
| Evaluation  | Morning 06-11   |           1716 |    286 | 15.3161 |       0.7704 |   44.1818 |            76.6321 |             -2.9604 |
| Evaluation  | Afternoon 12-17 |           1716 |    286 | 20.641  |       0.7605 |   55.1651 |            98.9822 |             -3.951  |
| Evaluation  | Evening 18-23   |           1716 |    286 | 10.4919 |       0.8695 |   40.4354 |            49.7976 |              6.9464 |

## Definitions

- All calendar classifications use Italian local delivery dates.
- National public holidays include Easter and Easter Monday.
- Local patron-saint holidays are excluded.
- Holidays take precedence over weekend classification.
- Bridge days are Fridays following Thursday holidays or
  Mondays preceding Tuesday holidays, excluding holidays.
- Night: 00–05; morning: 06–11; afternoon: 12–17;
  evening: 18–23.

## Methodological limitations

1. The calibration period covers only July–December 2025,
   while the evaluation period covers January–October 2026.
   Seasonal comparisons are therefore confounded by time
   period and changes in market conditions.
2. Holiday and bridge-day groups may have small sample sizes.
   Inspect both observation and distinct-day counts.
3. Observations within days and across consecutive days are
   dependent. No formal significance claims are made here.
4. Calendar variables are known at forecast time, but realized
   error and coverage metrics are not.
5. Changes in market difficulty may explain some performance
   differences; they are not necessarily model degradation.
6. DST transition days may contain 23 or 25 delivery hours.


## Implications for monitoring

Hour-specific and calendar-specific performance should be
investigated before constructing a composite Health Index.

Do not introduce arbitrary calendar penalties or weights.
Any context-adjusted threshold must be estimated and
validated on appropriate historical data.

## Final conclusion

Experiment 07 investigated the performance of LEAR probabilistic
electricity price forecasts across Italian delivery hours,
weekdays, weekends, national public holidays, and seasons.
The analysis covered 11,280 hourly observations from July 2025
to October 2026.

The results revealed substantial intraday heterogeneity in
forecasting accuracy and interval calibration. During the
evaluation period, the largest mean absolute errors and
strongest undercoverage occurred around late morning and
midday. These problems were particularly visible on weekends
and public holidays, whereas nighttime forecasts generally
exhibited overcoverage. Consequently, aggregate coverage
statistics could conceal opposing calibration errors across
delivery hours.

Public holidays exhibited particularly poor performance during
the evaluation period, with an average MAE of 23.14 EUR/MWh
and observed coverage of 59.38% for nominal 80% prediction
intervals. Examination of individual holidays showed that
undercoverage occurred on multiple dates rather than being
driven by a single extreme observation. Nevertheless, the
limited number of holidays and the overlap with broader
changes in forecasting performance prevent a causal
interpretation of the holiday effect.

The findings support incorporating hour-specific calibration
and accuracy diagnostics into the future LEAR monitoring
framework. Calendar variables should initially be retained
as contextual information rather than assigned independent
penalties. Further analysis is required to distinguish
systematic calendar-related behavior from temporal changes
in electricity-market conditions.
