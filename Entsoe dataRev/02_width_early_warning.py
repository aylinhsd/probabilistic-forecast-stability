
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT / "market_regime_results"
    / "stability_monitoring"
    / "daily_stability_monitoring.csv"
)

df = pd.read_csv(SOURCE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

BASELINE_END = pd.Timestamp("2026-02-28")
EVAL_START = pd.Timestamp("2026-03-01")
HORIZON = 7

baseline = df[df["date"] <= BASELINE_END].copy()
evaluation = df[df["date"] >= EVAL_START].copy()

# Thresholds learned only from the reference period.
width_threshold = baseline["width80"].quantile(0.95)
crps_threshold = baseline["rolling_crps"].dropna().quantile(0.95)
coverage_threshold = baseline["rolling_coverage80"].dropna().quantile(0.05)

# Forecast-side alerts: known when the forecast is issued.
df["width_alert"] = df["width80"] > width_threshold

# Health deterioration: known only after outcomes.
df["health_bad"] = (
    (df["rolling_crps"] > crps_threshold)
    | (df["rolling_coverage80"] < coverage_threshold)
)

# First crossings into bad health, rather than every bad day.
df["health_onset"] = (
    df["health_bad"]
    & ~df["health_bad"].shift(1, fill_value=False)
)

evaluation = df[df["date"] >= EVAL_START].copy()

# Evaluate whether each alert precedes a health onset.
results = []
for i, row in evaluation.iterrows():
    if not row["width_alert"]:
        continue

    future = df.iloc[i + 1:i + HORIZON + 1]
    future = future[
        future["date"] <= df["date"].iloc[-1]
    ]

    onset_dates = future.loc[future["health_onset"], "date"]
    success = not onset_dates.empty

    results.append({
        "alert_date": row["date"],
        "width80": row["width80"],
        "followed_by_onset": success,
        "lead_days": (
            (onset_dates.iloc[0] - row["date"]).days
            if success else np.nan
        ),
    })

alerts = pd.DataFrame(results)

print("REFERENCE PERIOD")
print("Days:", len(baseline))
print("Width threshold:", round(width_threshold, 3))
print("CRPS threshold:", round(crps_threshold, 3))
print("Coverage lower threshold:", round(coverage_threshold, 4))

print("\nEVALUATION PERIOD")
print("Days:", len(evaluation))
print("Width alert days:", int(evaluation["width_alert"].sum()))
print("Health onset events:", int(evaluation["health_onset"].sum()))

if not alerts.empty:
    print("Alerts followed by onset within 7 days:",
          int(alerts["followed_by_onset"].sum()))
    print("Alert-level hit rate:",
          round(alerts["followed_by_onset"].mean(), 3))
    print("Median lead days:",
          alerts["lead_days"].median())
else:
    print("No width alerts detected.")

OUTPUT = ROOT / "market_regime_results" / "early_warning"
OUTPUT.mkdir(parents=True, exist_ok=True)

alerts.to_csv(OUTPUT / "width_alerts.csv", index=False)
df.to_csv(OUTPUT / "width_health_daily.csv", index=False)

print("\nSaved results to:", OUTPUT)
