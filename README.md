# Probabilistic Electricity Price Forecasting: Stability Monitoring and Early Warning

## Project Overview

This repository contains research code and experimental results for evaluating the **stability and reliability of probabilistic electricity price forecasts**.

The main objective is to understand how forecasting performance changes over time, distinguish meaningful deterioration from ordinary market fluctuations, and investigate methods for monitoring forecast reliability.

A probabilistic implementation of **LEAR v3** is used as an initial forecasting testbed. The principal research focus is not developing a new forecasting algorithm, but designing and evaluating monitoring approaches that can eventually be applied to different probabilistic forecasting models.

The research addresses two related objectives:

1. **Forecast health monitoring:** Evaluating forecast reliability after actual electricity prices become available and detecting meaningful changes in performance.
2. **Early warning:** Investigating whether future deterioration can be anticipated using only information available when forecasts are issued.

The current development branch primarily contains the health-monitoring experiments and controlled detector-validation methods. Genuine forecast-time early warning remains a planned research stage.

## Data and Forecasting Model

The initial experiments use day-ahead electricity prices for the **Italy North bidding zone**, obtained from the ENTSO-E Transparency Platform.

The model-ready dataset is divided chronologically:

| Period | Dates | Observations |
|---|---|---:|
| Training | 12 January 2015 – 31 December 2024 | 87,408 |
| Validation | 1 January – 30 June 2025 | 4,343 |
| Testing | 1 July 2025 – 13 October 2026 | 11,280 |

The forecasting test period contains 470 days of hourly forecasts.

The probabilistic LEAR v3 implementation uses hour-specific L1-regularized quantile regression models to estimate multiple quantiles of the electricity price distribution.

Model configuration and regularization are selected using validation data, while the final test predictions are retained for independent forecasting evaluation and subsequent monitoring experiments.

The implementation provides probabilistic predictions for evaluating distributional accuracy, calibration, and uncertainty.

### Preliminary Forecasting Results

| Metric | Value |
|---|---:|
| Approximate mean CRPS | 8.10 EUR/MWh |
| Median forecast MAE (Q50) | 11.09 EUR/MWh |
| Empirical PI80 coverage | 83.96% |
| Mean PI80 width | 36.80 EUR/MWh |

These values describe the initial probabilistic forecasting testbed and are not intended to establish state-of-the-art forecasting performance.

## Research Methodology

The monitoring framework investigates complementary dimensions of probabilistic forecast quality.

**Probabilistic accuracy:** Evaluated primarily through the Continuous Ranked Probability Score (CRPS).

**Calibration:** Evaluated through prediction interval coverage, including deviations both below and above nominal coverage levels, and additional distributional diagnostics.

**Sharpness:** Investigated through prediction interval width, interpreted together with calibration.

**Market difficulty:** Investigated using electricity price volatility and other market characteristics.

**Benchmark-relative performance:** Used to distinguish changes in absolute forecasting performance from changes relative to reference forecasts.

**Temporal stability:** Investigated through rolling performance indicators and statistical change-detection methods.

The research follows the principle that probabilistic forecasts should be as sharp as possible while remaining appropriately calibrated.

## Monitoring Reference and Evaluation Periods

For the main monitoring experiments, the 470-day forecasting test dataset is divided into:

| Period | Dates | Duration |
|---|---|---:|
| Reference | 1 July 2025 – 28 February 2026 | 243 days |
| Evaluation | 1 March – 13 October 2026 | 227 days |

The reference period is used to characterize historical performance and establish monitoring thresholds.

The evaluation period is used to examine subsequent behavior and investigate the response of monitoring indicators under changing market conditions.

This division is separate from the training, validation, and testing split used for the forecasting model.

The monitoring boundary is an exploratory choice motivated by observed changes in market volatility. Consequently, historical alert results should not be interpreted as independently validated model failures.

## Research Experiments

The experiments are organized sequentially under `monitoring_research/`.

### Experiments 01–06: Initial Reliability Monitoring

These experiments investigate prediction interval behavior, early-warning hypotheses, adaptive interval-width analysis, multiple health indicators, PIT calibration, and conditional reliability.

The objective is to understand how different probabilistic performance measures behave and whether they provide useful information for stability monitoring.

### Experiments 07–11: Conditional Performance and Preliminary Detection

These experiments investigate calendar-related performance differences, redundancy between monitoring metrics, statistical change detection, detector evaluation, and retrospective warning episodes.

The objective is to identify complementary indicators and understand the limitations of preliminary alerting approaches.

### Experiment 12: Market Volatility and Forecast Performance

Investigates relationships between electricity price volatility and probabilistic forecasting performance, including CRPS, coverage, and prediction interval width.

### Experiment 13: Benchmark-Relative Performance

Compares absolute forecasting performance with performance relative to reference forecasts to investigate the influence of market difficulty.

### Experiment 14: Market-Aware Monitoring

Compares monitoring signals and threshold approaches that account for absolute performance, market volatility, and benchmark-relative behavior.

### Experiment 14b: Alert Diagnostics

Examines alert characteristics, calibration deviation, coverage instability, and the interpretation of monitoring signals.

Particular attention is given to both undercoverage and overcoverage relative to the nominal PI80 target.

### Experiment 15: Multidimensional Detector Comparison

Compares statistical change-detection methods across several monitoring signals.

The investigated methods include:

- Fixed percentile thresholds (P95)
- Cumulative Sum (CUSUM)
- Page-Hinkley
- Exponentially Weighted Moving Average (EWMA)
- Adaptive Windowing (ADWIN)

The experiment investigates historical alert behavior and controlled synthetic changes, considering detection sensitivity, delay, and unnecessary alerts.

### Experiment 15b: Forecast-Level Detector Validation

Extends detector evaluation by introducing controlled perturbations directly into probabilistic forecasts.

The scenarios include forecast bias, gradual changes, temporary disturbances, and changes in prediction interval width.

Probabilistic performance indicators are recalculated from the modified forecast distributions.

The experimental framework also incorporates chronological validation, controlled notification budgets, and held-out detector evaluation.

Controlled perturbation results are intended to evaluate detector behavior under known artificial changes. They should not be interpreted as independent validation against real operational model failures.

## Repository Structure

The main research files are located under `Entsoe dataRev/`.

```text
probabilistic-forecast-stability/
│
├── README.md
├── .gitignore
│
└── Entsoe dataRev/
    │
    ├── Forecasting and data-preparation scripts
    ├── LEAR model outputs and metadata
    ├── LEAR_rolling_outputs/
    ├── daily_stability_results/
    ├── market_regime_results/
    │
    └── monitoring_research/
        ├── scripts/
        │   └── experiment_01 ... experiment_15b
        ├── tables/
        ├── figures/
        └── reports/
```

The `monitoring_research` directories contain:

| Directory | Contents |
|---|---|
| `scripts/` | Python implementations of monitoring experiments |
| `tables/` | Numerical results and evaluation summaries |
| `figures/` | Graphs and visualizations of experimental findings |
| `reports/` | Experiment-specific methodological and results summaries |

## Running the Experiments

The research uses Python and common scientific computing libraries.

The monitoring experiments are designed around input data and intermediate results generated by earlier stages.

Some scripts depend on specific output filenames and relative directory paths. Therefore, the repository structure should be preserved when reproducing the experiments.

Example execution from the repository's main research directory:

```bash
cd "Entsoe dataRev"

python3 monitoring_research/scripts/experiment_12_volatility_vs_performance.py

python3 monitoring_research/scripts/experiment_13_benchmark_relative_performance.py

python3 monitoring_research/scripts/experiment_14_market_aware_monitoring.py

python3 monitoring_research/scripts/experiment_14b_alert_diagnostics.py

python3 monitoring_research/scripts/experiment_15_multidimentional_detector_comparison.py

python3 monitoring_research/scripts/experiment_15b_forecast_level_detector_selection.py
```

These examples identify the experiment entry points. Reproduction requires the corresponding input datasets and dependencies.

## Current Limitations

The current research has several limitations:

- The initial monitoring framework has mainly been developed using probabilistic LEAR v3 forecasts.
- Historical monitoring alerts do not necessarily correspond to independently verified model deterioration.
- Controlled synthetic changes provide useful detector tests but do not fully reproduce real-world forecasting failures.
- The monitoring reference period does not cover a complete annual cycle.
- The experimental framework requires further evaluation across different forecasting models and market conditions.

These limitations motivate the next stages of research.

## Future Research

Planned directions include:

1. Investigating monitoring thresholds under different market volatility regimes.
2. Extending detector validation using additional probabilistic forecasting models.
3. Evaluating monitoring approaches against documented historical deterioration events where available.
4. Developing genuine forecast-time early-warning indicators.
5. Investigating whether early-warning signals can provide useful advance notice of future reliability problems without excessive false alarms.

## Research Status

**Status:** Active research development.

This branch contains experimental monitoring and detector-validation work. Some components are exploratory and should not be interpreted as a finalized operational monitoring system.

The methodology, experimental procedures, and results may be refined as the research progresses.

## Data Source

Electricity market data were obtained from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/).

Users reproducing this work should consult ENTSO-E's applicable data access and reuse conditions.

## References

- Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness. *Journal of the Royal Statistical Society: Series B, 69*(2), 243–268.
- Weron, R. (2014). Electricity price forecasting: A review of the state-of-the-art with a look into the future. *International Journal of Forecasting, 30*(4), 1030–1081.
- Lago, J., Marcjasz, G., De Schutter, B., & Weron, R. (2021). Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark. *Applied Energy, 293*, 116983.
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys, 46*(4), Article 44.

---

**Note:** This repository contains ongoing academic research. Reported experimental findings should be interpreted within their stated assumptions, evaluation periods, and methodological limitations.
