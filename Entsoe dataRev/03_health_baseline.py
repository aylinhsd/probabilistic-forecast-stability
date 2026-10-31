
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

PREDICTIONS = ROOT / "ENTSOE_Probabilistic_LEAR_v3_predictions.csv"
MONITORING = (
    ROOT / "market_regime_results"
    / "stability_monitoring"
    / "daily_stability_monitoring.csv"
)
OUTPUT = ROOT / "market_regime_results" / "health_baseline"
OUTPUT.mkdir(parents=True, exist_ok=True)

WINDOW = 30
MIN_PERIODS = 15

# Load original predictions and existing monitoring data.
pred = pd.read_csv(PREDICTIONS)
pred = pred[pred["Split"].str.lower() == "test"].copy()

pred["date"] = pd.to_datetime(pred["Target_Date_Italy"])
pred = pred.sort_values(["date", "Target_UTC"])

monitor = pd.read_csv(MONITORING, parse_dates=["date"])

# Validate data before calculating metrics.
assert len(pred) == 11280, "Unexpected test sample size."
assert pred["date"].nunique() == 470, "Unexpected number of days."

quantile_cols = sorted(
    [c for c in pred.columns if c.startswith("Q_")],
    key=lambda c: float(c.split("_")[1])
)
quantile_levels = np.array([
    float(c.split("_")[1]) for c in quantile_cols
])

assert len(quantile_cols) == 101
assert not pred[
    ["Target_Price_EUR_MWh"] + quantile_cols
].isna().any().any(), "Missing prediction values."

y = pred["Target_Price_EUR_MWh"].to_numpy()
q = pred[quantile_cols].to_numpy()

# 1. Median forecast errors.
median = pred["Q_0.500"].to_numpy()
pred["absolute_error"] = np.abs(y - median)
pred["squared_error"] = (y - median) ** 2

# 2. Approximate PIT from the predicted quantile function.
# Flat sections are handled by interpolation on unique values.
def approximate_pit(actual, quantiles, levels):
    unique_q, inverse = np.unique(
        quantiles, return_inverse=True
    )
    unique_levels = np.array([
        levels[inverse == i].mean()
        for i in range(len(unique_q))
    ])
    return np.interp(
        actual, unique_q, unique_levels,
        left=0.0, right=1.0
    )

pred["pit"] = [
    approximate_pit(actual, values, quantile_levels)
    for actual, values in zip(y, q)
]

# 3. Interval score: lower is better.
def interval_score(actual, lower, upper, alpha):
    return (
        upper - lower
        + (2 / alpha) * np.maximum(lower - actual, 0)
        + (2 / alpha) * np.maximum(actual - upper, 0)
    )

pred["interval_score80"] = interval_score(
    y,
    pred["Q_0.100"].to_numpy(),
    pred["Q_0.900"].to_numpy(),
    alpha=0.20
)

pred["interval_score90"] = interval_score(
    y,
    pred["Q_0.050"].to_numpy(),
    pred["Q_0.950"].to_numpy(),
    alpha=0.10
)

# 4. Daily health metrics.
daily = pred.groupby("date").agg(
    n=("absolute_error", "size"),
    mae=("absolute_error", "mean"),
    mse=("squared_error", "mean"),
    pit_mean=("pit", "mean"),
    pit_var=("pit", "var"),
    interval_score80=("interval_score80", "mean"),
    interval_score90=("interval_score90", "mean")
).reset_index()

daily["rmse"] = np.sqrt(daily.pop("mse"))

# A simple PIT deviation diagnostic.
# This is not a formal PIT uniformity test.
daily["pit_mean_deviation"] = np.abs(
    daily["pit_mean"] - 0.5
)

# 5. Merge with existing CRPS and coverage metrics.
daily = daily.merge(
    monitor.drop(columns=["n"]),
    on="date",
    how="left",
    validate="one_to_one"
)

assert daily["crps"].notna().all(), "Missing daily CRPS."

# 6. Coverage deviation for multiple nominal intervals.
for nominal in [80, 85, 90]:
    daily[f"coverage{nominal}_abs_deviation"] = np.abs(
        daily[f"coverage{nominal}"] - nominal / 100
    )

# 7. Rolling health indicators.
rolling_mean_cols = [
    "crps",
    "mae",
    "interval_score80",
    "interval_score90",
    "pit_mean_deviation"
]

for col in rolling_mean_cols:
    daily[f"health_rolling_{col}"] = (
        daily[col].rolling(
            WINDOW, min_periods=MIN_PERIODS
        ).mean()
    )

# Pooled RMSE across the rolling hourly forecasts.
rolling_mse = (
    daily["rmse"].pow(2)
    .rolling(WINDOW, min_periods=MIN_PERIODS)
    .mean()
)
daily["health_rolling_rmse"] = np.sqrt(rolling_mse)

# Rolling coverage deviation must be computed AFTER
# aggregating coverage, not averaged from daily deviations.
for nominal in [80, 85, 90]:
    coverage = daily[f"coverage{nominal}"].rolling(
        WINDOW, min_periods=MIN_PERIODS
    ).mean()
    daily[f"health_rolling_coverage{nominal}_deviation"] = (
        coverage - nominal / 100
    ).abs()

# 8. Save.
output_file = OUTPUT / "daily_health_baseline.csv"
daily.to_csv(output_file, index=False)

print("=== HEALTH BASELINE RESULTS ===")
print("Days:", len(daily))
print("Hourly forecasts:", len(pred))
print("Overall MAE:", round(pred["absolute_error"].mean(), 4))
print("Overall RMSE:", round(np.sqrt(pred["squared_error"].mean()), 4))
print("Mean PIT:", round(pred["pit"].mean(), 4))
print("Mean 80% interval score:",
      round(pred["interval_score80"].mean(), 4))
print("Mean 90% interval score:",
      round(pred["interval_score90"].mean(), 4))

print("\nSaved:", output_file)
print("\nLatest rolling health indicators:")
print(daily[[
    "date",
    "health_rolling_crps",
    "health_rolling_mae",
    "health_rolling_rmse",
    "health_rolling_coverage80_deviation",
    "health_rolling_interval_score80"
]].tail(5).round(4).to_string(index=False))
