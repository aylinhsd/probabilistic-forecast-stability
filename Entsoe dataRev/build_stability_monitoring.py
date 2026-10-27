
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "market_regime_results" / "stability_monitoring"
OUT.mkdir(parents=True, exist_ok=True)

# Load the native probabilistic LEAR v3 predictions
df = pd.read_csv(
    ROOT / "ENTSOE_Probabilistic_LEAR_v3_predictions.csv"
)
df = df[df["Split"] == "Test"].copy()

# Convert timestamps to Italian local delivery dates
df["Target_UTC"] = pd.to_datetime(df["Target_UTC"], utc=True)
df = df.sort_values("Target_UTC")
df["date"] = df["Target_UTC"].dt.tz_convert(
    "Europe/Rome"
).dt.date

# Extract observations and all predicted quantiles
y = df["Target_Price_EUR_MWh"].to_numpy()

qcols = sorted(
    [c for c in df.columns if c.startswith("Q_")],
    key=lambda c: float(c[2:])
)

taus = np.array([float(c[2:]) for c in qcols])
q = df[qcols].to_numpy()

# Approximate CRPS using integrated pinball loss
error = y[:, None] - q
pinball = np.maximum(
    taus * error,
    (taus - 1) * error
)

df["approx_crps"] = 2 * np.trapezoid(
    pinball, taus, axis=1
)

# Prediction interval coverage and width
for level, lo, hi in [
    (80, "Q_0.100", "Q_0.900"),
    (85, "Q_0.075", "Q_0.925"),
    (90, "Q_0.050", "Q_0.950")
]:
    df[f"covered_{level}"] = (
        (df["Target_Price_EUR_MWh"] >= df[lo]) &
        (df["Target_Price_EUR_MWh"] <= df[hi])
    ).astype(float)

    df[f"width_{level}"] = df[hi] - df[lo]

# Daily forecast performance and observed market conditions
daily = df.groupby("date").agg(
    n=("Target_Price_EUR_MWh", "size"),
    crps=("approx_crps", "mean"),
    coverage80=("covered_80", "mean"),
    coverage85=("covered_85", "mean"),
    coverage90=("covered_90", "mean"),
    width80=("width_80", "mean"),
    width85=("width_85", "mean"),
    width90=("width_90", "mean"),
    volatility=("Target_Price_EUR_MWh", "std"),
    mean_price=("Target_Price_EUR_MWh", "mean")
).reset_index()

daily["date"] = pd.to_datetime(daily["date"])

# Calculate rolling monitoring indicators
window = 30
min_days = 15

for metric in [
    "crps",
    "coverage80", "coverage85", "coverage90",
    "width80", "width85", "width90"
]:
    numerator = (
        daily[metric] * daily["n"]
    ).rolling(
        window, min_periods=min_days
    ).sum()

    denominator = daily["n"].rolling(
        window, min_periods=min_days
    ).sum()

    daily[f"rolling_{metric}"] = numerator / denominator

# Rolling observed daily price volatility
daily["rolling_volatility"] = (
    daily["volatility"]
    .rolling(window, min_periods=min_days)
    .mean()
)

# Absolute deviation from nominal 80% coverage
daily["coverage80_deviation"] = (
    daily["rolling_coverage80"] - 0.80
).abs()

daily["is_weekend"] = daily["date"].dt.weekday >= 5

# Save monitoring dataset
daily.to_csv(
    OUT / "daily_stability_monitoring.csv",
    index=False
)

# Create monitoring graphs
fig, axes = plt.subplots(
    3, 1, figsize=(15, 11), sharex=True
)

# Panel 1: Rolling coverage
axes[0].plot(
    daily["date"],
    daily["rolling_coverage80"] * 100,
    label="30-day rolling PI80 coverage"
)
axes[0].axhline(
    80, color="black", linestyle="--",
    label="Nominal 80% coverage"
)
axes[0].set_ylabel("Coverage (%)")
axes[0].legend()

# Panel 2: Rolling approximate CRPS
axes[1].plot(
    daily["date"],
    daily["rolling_crps"],
    label="30-day rolling approximate CRPS"
)
axes[1].set_ylabel("Approx. CRPS (EUR/MWh)")
axes[1].legend()

# Panel 3: Interval width and market volatility
axes[2].plot(
    daily["date"],
    daily["rolling_width80"],
    label="30-day rolling PI80 width"
)
axes[2].plot(
    daily["date"],
    daily["rolling_volatility"],
    label="30-day mean observed daily volatility"
)
axes[2].set_ylabel("EUR/MWh")
axes[2].legend()

# Mark previously defined market periods
for ax in axes:
    for boundary in [
        "2026-03-01",
        "2026-05-01",
        "2026-07-01",
        "2026-08-01"
    ]:
        ax.axvline(
            pd.Timestamp(boundary),
            color="gray",
            linestyle=":",
            alpha=0.7
        )
    ax.grid(alpha=0.2)

axes[0].set_title(
    "Native Probabilistic LEAR v3 - Stability Monitoring"
)
axes[-1].set_xlabel("Italian local delivery date")

fig.autofmt_xdate()
fig.tight_layout()

fig.savefig(
    OUT / "stability_monitoring.png",
    dpi=180
)
plt.close(fig)

# Print verification results
print("Days:", len(daily))
print("Test observations:", int(daily["n"].sum()))
print(
    "Mean approximate CRPS:",
    round(df["approx_crps"].mean(), 4)
)
print(
    "Overall 80% coverage:",
    round(df["covered_80"].mean() * 100, 3)
)
print("Saved:", OUT)
