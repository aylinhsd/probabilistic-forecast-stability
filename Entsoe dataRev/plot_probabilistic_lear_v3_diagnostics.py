
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "market_regime_results" / "probabilistic_lear_v3_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

prob = pd.read_csv(ROOT / "ENTSOE_Probabilistic_LEAR_v3_predictions.csv")
point = pd.read_csv(
    ROOT / "ENTSOE_LEAR_v3_predictions.csv",
    usecols=["Target_UTC", "LEAR_v3_Prediction_EUR_MWh"]
)

prob["Target_UTC"] = pd.to_datetime(prob["Target_UTC"], utc=True)
point["Target_UTC"] = pd.to_datetime(point["Target_UTC"], utc=True)

assert prob["Target_UTC"].is_unique
assert point["Target_UTC"].is_unique

df = prob.merge(point, on="Target_UTC", how="left", validate="one_to_one")
df = df[df["Split"] == "Test"].copy()
df = df.sort_values("Target_UTC")

assert df["LEAR_v3_Prediction_EUR_MWh"].notna().all()
assert len(df) == 11280

df["local"] = df["Target_UTC"].dt.tz_convert("Europe/Rome")
df["date"] = df["local"].dt.date

intervals = {
    80: ("Q_0.100", "Q_0.900"),
    85: ("Q_0.075", "Q_0.925"),
    90: ("Q_0.050", "Q_0.950")
}

for level, (lower, upper) in intervals.items():
    df[f"covered_{level}"] = (
        (df["Target_Price_EUR_MWh"] >= df[lower]) &
        (df["Target_Price_EUR_MWh"] <= df[upper])
    ).astype(int)
    df[f"width_{level}"] = df[upper] - df[lower]

periods = {
    "baseline": ("2025-07-01", "2026-03-01"),
    "march_april": ("2026-03-01", "2026-05-01"),
    "may_june": ("2026-05-01", "2026-07-01"),
    "july": ("2026-07-01", "2026-08-01"),
    "aug_oct": ("2026-08-01", "2026-10-14")
}

broad_periods = {
    "baseline": ("2025-07-01", "2026-03-01"),
    "march_july": ("2026-03-01", "2026-08-01"),
    "aug_oct": ("2026-08-01", "2026-10-14")
}

holidays = {
    "2025-08-15": "Assumption",
    "2025-11-01": "All Saints",
    "2025-12-08": "Immaculate Conception",
    "2025-12-25": "Christmas",
    "2025-12-26": "St Stephen",
    "2026-01-01": "New Year",
    "2026-01-06": "Epiphany",
    "2026-04-05": "Easter",
    "2026-04-06": "Easter Monday",
    "2026-04-25": "Liberation Day",
    "2026-05-01": "Labour Day",
    "2026-06-02": "Republic Day",
    "2026-08-15": "Assumption"
}

def subset(start, end):
    start = pd.Timestamp(start, tz="Europe/Rome")
    end = pd.Timestamp(end, tz="Europe/Rome")
    return df[(df["local"] >= start) & (df["local"] < end)].copy()

def decorate(ax, dates):
    dates = pd.to_datetime(dates)
    for day in pd.date_range(dates.min(), dates.max(), freq="D"):
        if day.weekday() >= 5:
            ax.axvspan(day, day + pd.Timedelta(days=1),
                       color="gray", alpha=0.06)
    for day in holidays:
        holiday = pd.Timestamp(day)
        if dates.min() <= holiday <= dates.max():
            ax.axvline(holiday, color="red", alpha=0.5, linewidth=0.8)
    ax.grid(alpha=0.2)
    ax.tick_params(axis="x", rotation=30)

daily = df.groupby("date").agg(
    actual=("Target_Price_EUR_MWh", "mean"),
    point=("LEAR_v3_Prediction_EUR_MWh", "mean"),
    median=("Q_0.500", "mean"),
    lower80=("Q_0.100", "mean"),
    upper80=("Q_0.900", "mean"),
    volatility=("Target_Price_EUR_MWh", "std"),
    coverage80=("covered_80", "mean"),
    coverage85=("covered_85", "mean"),
    coverage90=("covered_90", "mean"),
    width80=("width_80", "mean"),
    width85=("width_85", "mean"),
    width90=("width_90", "mean")
).reset_index()

daily["date"] = pd.to_datetime(daily["date"])

for level in (80, 85, 90):
    daily[f"rolling_coverage{level}"] = (
        daily[f"coverage{level}"].rolling(14, min_periods=7).mean() * 100
    )

daily["rolling_width80"] = daily["width80"].rolling(7, min_periods=3).mean()
daily["rolling_volatility"] = daily["volatility"].rolling(7, min_periods=3).mean()

results = []

for name, (start, end) in periods.items():
    p = subset(start, end)
    result = {
        "Period": name,
        "Observations": len(p),
        "Point_MAE": (
            p["Target_Price_EUR_MWh"] -
            p["LEAR_v3_Prediction_EUR_MWh"]
        ).abs().mean(),
        "Probabilistic_Median_MAE": (
            p["Target_Price_EUR_MWh"] - p["Q_0.500"]
        ).abs().mean()
    }
    for level in (80, 85, 90):
        result[f"Coverage_{level}_pct"] = p[f"covered_{level}"].mean() * 100
        result[f"Width_{level}"] = p[f"width_{level}"].mean()
    results.append(result)

summary = pd.DataFrame(results)
summary.to_csv(OUT / "five_period_metrics.csv", index=False)
daily.to_csv(OUT / "daily_diagnostics.csv", index=False)

print("\nFive-period evaluation:")
print(summary.round(3).to_string(index=False))

for name, (start, end) in broad_periods.items():
    d = daily[
        (daily["date"] >= pd.Timestamp(start)) &
        (daily["date"] < pd.Timestamp(end))
    ].copy()

    # Graph 1: rolling coverage
    fig, ax = plt.subplots(figsize=(14, 5))
    for level in (80, 85, 90):
        ax.plot(d["date"], d[f"rolling_coverage{level}"],
                label=f"{level}% interval coverage")
        ax.axhline(level, linestyle=":", alpha=0.6)
    decorate(ax, d["date"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Rolling coverage (%)")
    ax.set_title(f"{name}: 14-day rolling prediction interval coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_coverage.png", dpi=160)
    plt.close(fig)

    # Graph 2: interval width and observed volatility
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(d["date"], d["rolling_width80"],
            label="80% prediction interval width (7-day mean)")
    ax.plot(d["date"], d["rolling_volatility"],
            label="Observed within-day price SD (7-day mean)")
    decorate(ax, d["date"])
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"{name}: interval width versus observed price volatility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_width_vs_volatility.png", dpi=160)
    plt.close(fig)

    # Graph 3: actual vs deterministic and probabilistic forecasts
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(
        d["date"], d["lower80"], d["upper80"],
        alpha=0.2, label="Mean hourly 80% interval bounds"
    )
    ax.plot(d["date"], d["actual"], color="black",
            label="Actual daily mean price")
    ax.plot(d["date"], d["point"], linestyle="--",
            label="Point LEAR v3")
    ax.plot(d["date"], d["median"],
            label="Probabilistic LEAR v3 median")
    decorate(ax, d["date"])
    ax.set_ylabel("Electricity price (EUR/MWh)")
    ax.set_title(f"{name}: actual versus LEAR v3 forecasts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_forecast_comparison.png", dpi=160)
    plt.close(fig)

print("\nFinished. Results saved in:", OUT)
