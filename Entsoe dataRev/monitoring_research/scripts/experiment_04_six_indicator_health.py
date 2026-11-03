
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "monitoring_research"

TABLES = RESEARCH / "tables"
FIGURES = RESEARCH / "figures"
REPORTS = RESEARCH / "reports"

for folder in (TABLES, FIGURES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

INPUT = (
    ROOT / "market_regime_results"
    / "health_baseline"
    / "daily_health_baseline.csv"
)

df = pd.read_csv(INPUT, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

if df["date"].duplicated().any():
    raise ValueError("Duplicate dates in monitoring dataset")

CAL_END = pd.Timestamp("2025-12-31")
EVAL_START = pd.Timestamp("2026-01-01")

cal = df[df["date"] <= CAL_END].copy()
ev = df[df["date"] >= EVAL_START].copy()

if len(cal) != 184 or len(ev) != 286:
    raise ValueError("Unexpected calibration/evaluation lengths")

# Five rolling health indicators plus a PIT diagnostic.
# PIT mean deviation is only a partial calibration diagnostic.
indicators = {
    "CRPS": ("health_rolling_crps", "upper"),
    "MAE": ("health_rolling_mae", "upper"),
    "RMSE": ("health_rolling_rmse", "upper"),
    "Coverage 80%": ("rolling_coverage80", "both"),
    "Interval score 80%": (
        "health_rolling_interval_score80", "upper"
    ),
    "PIT mean deviation": (
        "health_rolling_pit_mean_deviation", "upper"
    ),
}

required = [v[0] for v in indicators.values()]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

threshold_rows = []
monthly_parts = []

fig, axes = plt.subplots(
    3, 2, figsize=(15, 12), sharex=True
)
axes = axes.flatten()

for ax, (name, (column, direction)) in zip(
    axes, indicators.items()
):
    reference = cal[column].dropna()

    low = reference.quantile(0.05)
    high = reference.quantile(0.95)

    threshold_rows.append({
        "indicator": name,
        "column": column,
        "calibration_mean": reference.mean(),
        "calibration_p05": low,
        "calibration_p95": high,
        "evaluation_mean": ev[column].mean(),
        "evaluation_days": ev[column].notna().sum()
    })

    df[column + "_upper_alert"] = df[column] > high

    if direction == "both":
        df[column + "_lower_alert"] = df[column] < low

    ax.plot(
        df["date"], df[column],
        linewidth=1.4, label=name
    )
    ax.axhline(
        high, linestyle="--", color="tab:red",
        label="Calibration 95th percentile"
    )

    if direction == "both":
        ax.axhline(
            low, linestyle="--", color="tab:orange",
            label="Calibration 5th percentile"
        )
        ax.axhline(
            0.80, linestyle=":", color="black",
            label="Nominal 80% coverage"
        )

    ax.axvline(
        EVAL_START, color="gray", linestyle=":",
        label="Evaluation starts"
    )

    ax.set_title(name)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    monthly = (
        df.assign(month=df["date"].dt.to_period("M"))
        .groupby("month")[column]
        .mean()
        .reset_index()
    )
    monthly["indicator"] = name
    monthly = monthly.rename(columns={column: "mean_value"})
    monthly_parts.append(monthly)

fig.suptitle(
    "Probabilistic LEAR v3 — Six Health Indicators",
    fontsize=16
)
fig.autofmt_xdate()
fig.tight_layout()

fig.savefig(
    FIGURES / "experiment_04_health_dashboard.png",
    dpi=300, bbox_inches="tight"
)
fig.savefig(
    FIGURES / "experiment_04_health_dashboard.pdf",
    bbox_inches="tight"
)
plt.close(fig)

thresholds = pd.DataFrame(threshold_rows)
monthly_results = pd.concat(
    monthly_parts, ignore_index=True
)

thresholds.to_csv(
    TABLES / "experiment_04_thresholds.csv",
    index=False
)

monthly_results.to_csv(
    TABLES / "experiment_04_monthly.csv",
    index=False
)

alert_columns = [
    c for c in df.columns if c.endswith("_alert")
]

df[["date"] + required + alert_columns].to_csv(
    TABLES / "experiment_04_daily.csv",
    index=False
)

# PIT mean deviation alone cannot assess full
# distribution calibration.
pit_note = """
PIT limitation:
The dashboard uses rolling PIT mean deviation as a
preliminary diagnostic, not a complete PIT drift test.

A mean near 0.5 does not guarantee uniform PIT values.
A proper PIT histogram and distributional calibration
test require the individual hourly PIT observations.
"""

report = f"""# Experiment 04 — Six-Indicator Health Monitoring

## Objective

Compare probabilistic LEAR v3 health indicators with
historical calibration-period reference distributions.

## Monitoring periods

Calibration: 2025-07-01 to 2025-12-31
Evaluation: 2026-01-01 to 2026-10-13

## Reference statistics

{thresholds.round(4).to_string(index=False)}

## Interpretation

Thresholds are exploratory reference quantiles, not
validated statistical alarm limits.

Higher CRPS, MAE, RMSE and interval scores indicate
worse forecast performance.

Coverage is evaluated in both directions and relative
to its nominal 80% target.

The PIT panel is a preliminary diagnostic only.

{pit_note}

## Limitations

- Health indicators require realized electricity prices.
- Rolling windows overlap and are statistically dependent.
- Six months of calibration may not represent all seasons.
- Market volatility can affect forecast difficulty.
- Indicator exceedances are not independent events.
- PIT distributional calibration remains to be evaluated.
- A composite health index has not yet been calibrated.
"""

(REPORTS / "experiment_04_report.md").write_text(report)

print("=== EXPERIMENT 04 ===")
print("Calibration days:", len(cal))
print("Evaluation days:", len(ev))

print("\n=== INDICATOR REFERENCE STATISTICS ===")
print(thresholds.round(4).to_string(index=False))

print("\nSaved:")
print("experiment_04_health_dashboard.png")
print("experiment_04_health_dashboard.pdf")
print("experiment_04_thresholds.csv")
print("experiment_04_monthly.csv")
print("experiment_04_daily.csv")
print("experiment_04_report.md")

print("\nNOTE: Full PIT calibration analysis is still required.")
