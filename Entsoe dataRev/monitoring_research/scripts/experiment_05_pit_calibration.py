
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import kstest

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "monitoring_research"

TABLES = BASE / "tables"
FIGURES = BASE / "figures"
REPORTS = BASE / "reports"

for folder in [TABLES, FIGURES, REPORTS]:
    folder.mkdir(parents=True, exist_ok=True)

INPUT = ROOT / "ENTSOE_Probabilistic_LEAR_v3_predictions.csv"

CAL_START = pd.Timestamp("2025-07-01")
CAL_END = pd.Timestamp("2025-12-31")
EVAL_START = pd.Timestamp("2026-01-01")
EVAL_END = pd.Timestamp("2026-10-13")

# Analysis convention, not a market-defined grouping.
DAY_START = 6
DAY_END = 17

# --------------------------------------------------
# 2. Load and validate predictions
# --------------------------------------------------

df = pd.read_csv(INPUT)

required = [
    "Target_Date_Italy",
    "Target_UTC",
    "Target_Price_EUR_MWh",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

quantile_pairs = []

for col in df.columns:
    match = re.fullmatch(r"Q_(0\.\d+|1(?:\.0+)?)", col)
    if match:
        quantile_pairs.append((float(match.group(1)), col))

quantile_pairs.sort()

q_levels = np.array([p for p, _ in quantile_pairs])
q_cols = [c for _, c in quantile_pairs]

if len(q_cols) < 3:
    raise ValueError("Not enough quantile columns found")

if len(np.unique(q_levels)) != len(q_levels):
    raise ValueError("Duplicate quantile probability levels")

if q_levels[0] <= 0 or q_levels[-1] >= 1:
    raise ValueError(
        "This implementation expects quantiles strictly between 0 and 1"
    )

df["date"] = pd.to_datetime(
    df["Target_Date_Italy"]
).dt.normalize()

df["target_utc"] = pd.to_datetime(
    df["Target_UTC"], utc=True
)

if df["target_utc"].isna().any():
    raise ValueError("Invalid Target_UTC values")

df["local_hour"] = (
    df["target_utc"]
    .dt.tz_convert("Europe/Rome")
    .dt.hour
)

df["local_date_from_utc"] = (
    df["target_utc"]
    .dt.tz_convert("Europe/Rome")
    .dt.tz_localize(None)
    .dt.normalize()
)

date_mismatches = (
    df["date"] != df["local_date_from_utc"]
).sum()

if date_mismatches:
    raise ValueError(
        f"{date_mismatches} local date mismatches detected"
    )

if df["target_utc"].duplicated().any():
    raise ValueError("Duplicate target UTC timestamps")

df = df[
    (df["date"] >= CAL_START)
    & (df["date"] <= EVAL_END)
].copy()

df["period"] = np.where(
    df["date"] <= CAL_END,
    "Calibration",
    "Evaluation"
)

df["day_night"] = np.where(
    df["local_hour"].between(DAY_START, DAY_END),
    "Day",
    "Night"
)

prices = pd.to_numeric(
    df["Target_Price_EUR_MWh"], errors="coerce"
).to_numpy(dtype=float)

forecasts = df[q_cols].apply(
    pd.to_numeric, errors="coerce"
).to_numpy(dtype=float)

# --------------------------------------------------
# 3. Reconstruct approximate PIT values
# --------------------------------------------------

def approximate_pit(y, values, probabilities):
    """
    Reconstruct an approximate CDF value from quantiles.

    Values outside the available quantile range are
    assigned tail midpoints. These are approximations,
    not exact PIT values.

    Non-monotonic or non-finite forecasts are excluded.
    """

    if not np.isfinite(y) or not np.all(np.isfinite(values)):
        return np.nan, "invalid_input"

    if np.any(np.diff(values) < -1e-9):
        return np.nan, "non_monotonic"

    if y < values[0]:
        return probabilities[0] / 2, "lower_tail"

    if y > values[-1]:
        return (1 + probabilities[-1]) / 2, "upper_tail"

    # Equal forecast prices can correspond to multiple
    # quantile levels. Use their probability midpoint.
    unique_values, inverse = np.unique(
        values, return_inverse=True
    )

    lower = np.full(len(unique_values), np.inf)
    upper = np.full(len(unique_values), -np.inf)

    np.minimum.at(lower, inverse, probabilities)
    np.maximum.at(upper, inverse, probabilities)

    midpoint_probs = (lower + upper) / 2

    pit = np.interp(
        y, unique_values, midpoint_probs
    )

    return float(pit), "interpolated"


results = [
    approximate_pit(y, row, q_levels)
    for y, row in zip(prices, forecasts)
]

df["pit"] = [r[0] for r in results]
df["pit_status"] = [r[1] for r in results]

valid = df.dropna(subset=["pit"]).copy()

if valid.empty:
    raise ValueError("No valid PIT values reconstructed")

# --------------------------------------------------
# 4. Statistical summaries
# --------------------------------------------------

def summarize(group):
    values = group["pit"].dropna().to_numpy()

    if len(values) == 0:
        return pd.Series({
            "n": 0,
            "pit_mean": np.nan,
            "pit_variance": np.nan,
            "mean_abs_deviation_from_0_5": np.nan,
            "tail_fraction_10pct": np.nan,
            "center_fraction_20pct": np.nan,
            "ks_statistic": np.nan,
            "ks_pvalue_naive": np.nan
        })

    # Diagnostic only: iid KS p-values are not reliable
    # for dependent hourly electricity forecasts.
    ks = kstest(values, "uniform")

    return pd.Series({
        "n": len(values),
        "pit_mean": values.mean(),
        "pit_variance": values.var(ddof=1)
            if len(values) > 1 else np.nan,
        "mean_abs_deviation_from_0_5":
            np.mean(np.abs(values - 0.5)),
        "tail_fraction_10pct":
            np.mean((values < 0.1) | (values > 0.9)),
        "center_fraction_20pct":
            np.mean((values >= 0.4) & (values <= 0.6)),
        "ks_statistic": ks.statistic,
        "ks_pvalue_naive": ks.pvalue
    })


overall = valid.groupby("period").apply(
    summarize, include_groups=False
).reset_index()

day_night = valid.groupby(
    ["period", "day_night"]
).apply(
    summarize, include_groups=False
).reset_index()

hourly = valid.groupby(
    ["period", "local_hour"]
).apply(
    summarize, include_groups=False
).reset_index()

valid["month"] = valid["date"].dt.strftime("%Y-%m")

monthly = valid.groupby("month").apply(
    summarize, include_groups=False
).reset_index()

status = (
    df.groupby(["period", "pit_status"])
    .size()
    .reset_index(name="count")
)

# Save all individual reconstructed PIT values.
df[[
    "date",
    "target_utc",
    "local_hour",
    "day_night",
    "period",
    "pit",
    "pit_status"
]].to_csv(
    TABLES / "experiment_05_hourly_pit.csv",
    index=False
)

for name, table in [
    ("overall", overall),
    ("day_night", day_night),
    ("hourly", hourly),
    ("monthly", monthly),
    ("reconstruction_status", status),
]:
    table.to_csv(
        TABLES / f"experiment_05_{name}.csv",
        index=False
    )

# --------------------------------------------------
# 5. Figures
# --------------------------------------------------

def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(
        FIGURES / f"{filename}.png",
        dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        FIGURES / f"{filename}.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)


# Figure 1: Overall PIT histograms
fig, axes = plt.subplots(
    1, 2, figsize=(13, 5), sharey=True
)

bins = np.linspace(0, 1, 11)

for ax, period in zip(
    axes, ["Calibration", "Evaluation"]
):
    values = valid.loc[
        valid["period"] == period, "pit"
    ]

    ax.hist(
        values, bins=bins, density=True,
        edgecolor="black", alpha=0.75
    )
    ax.axhline(
        1, linestyle="--", color="red",
        label="Uniform reference"
    )
    ax.set_title(f"{period} (n={len(values)})")
    ax.set_xlabel("Approximate PIT")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.2)

fig.suptitle("LEAR v3 — PIT Distribution")
save_figure(fig, "experiment_05_pit_histograms")


# Figure 2: Day/night PIT distributions
fig, axes = plt.subplots(
    2, 2, figsize=(12, 9),
    sharex=True, sharey=True
)

for i, period in enumerate(
    ["Calibration", "Evaluation"]
):
    for j, segment in enumerate(["Day", "Night"]):
        ax = axes[i, j]

        subset = valid[
            (valid["period"] == period)
            & (valid["day_night"] == segment)
        ]

        ax.hist(
            subset["pit"], bins=bins,
            density=True, edgecolor="black",
            alpha=0.75
        )
        ax.axhline(
            1, color="red", linestyle="--"
        )
        ax.set_title(
            f"{period} — {segment} (n={len(subset)})"
        )
        ax.set_xlabel("Approximate PIT")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)

fig.suptitle("LEAR v3 — Day/Night PIT Calibration")
save_figure(fig, "experiment_05_day_night_pit")


# Figure 3: PIT summary by delivery hour
fig, axes = plt.subplots(
    2, 1, figsize=(13, 9), sharex=True
)

for period in ["Calibration", "Evaluation"]:
    sub = hourly[hourly["period"] == period]

    axes[0].plot(
        sub["local_hour"],
        sub["pit_mean"],
        marker="o", label=period
    )

    axes[1].plot(
        sub["local_hour"],
        sub["tail_fraction_10pct"],
        marker="o", label=period
    )

axes[0].axhline(
    0.5, color="black", linestyle="--"
)
axes[1].axhline(
    0.2, color="black", linestyle="--"
)

axes[0].set_ylabel("Mean approximate PIT")
axes[1].set_ylabel("Fraction in outer 10% tails")
axes[1].set_xlabel("Italian local delivery hour")
axes[1].set_xticks(range(24))

for ax in axes:
    ax.grid(alpha=0.2)
    ax.legend()

fig.suptitle("LEAR v3 — Hour-Specific PIT Diagnostics")
save_figure(fig, "experiment_05_hourly_pit")


# Figure 4: Monthly PIT diagnostics
fig, axes = plt.subplots(
    2, 1, figsize=(13, 8), sharex=True
)

x = np.arange(len(monthly))

axes[0].plot(
    x, monthly["pit_mean"], marker="o"
)
axes[0].axhline(
    0.5, color="black", linestyle="--"
)
axes[0].set_ylabel("Mean approximate PIT")

axes[1].plot(
    x, monthly["tail_fraction_10pct"],
    marker="o"
)
axes[1].axhline(
    0.2, color="black", linestyle="--"
)
axes[1].set_ylabel("Fraction in outer 10% tails")

axes[1].set_xticks(x)
axes[1].set_xticklabels(
    monthly["month"], rotation=60, ha="right"
)

for ax in axes:
    ax.grid(alpha=0.2)

fig.suptitle("LEAR v3 — Monthly PIT Diagnostics")
save_figure(fig, "experiment_05_monthly_pit")


# --------------------------------------------------
# 6. Documentation
# --------------------------------------------------

report = f"""# Experiment 05 — PIT Calibration

## Objective

Investigate distributional calibration of probabilistic
LEAR v3 forecasts using approximate hourly PIT values.

## Periods

Calibration: 2025-07-01 to 2025-12-31
Evaluation: 2026-01-01 to 2026-10-13

## Quantile reconstruction

Number of quantiles: {len(q_cols)}
Minimum quantile level: {q_levels.min():.3f}
Maximum quantile level: {q_levels.max():.3f}

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

{status.to_string(index=False)}

## Overall PIT statistics

{overall.round(4).to_string(index=False)}

## Day/night PIT statistics

{day_night.round(4).to_string(index=False)}

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
"""

(REPORTS / "experiment_05_report.md").write_text(
    report, encoding="utf-8"
)

print("=== EXPERIMENT 05 ===")
print("Prediction rows:", len(df))
print("Valid PIT rows:", len(valid))
print("Quantile columns:", len(q_cols))

print("\n=== RECONSTRUCTION STATUS ===")
print(status.to_string(index=False))

print("\n=== OVERALL PIT STATISTICS ===")
print(overall.round(4).to_string(index=False))

print("\n=== DAY/NIGHT STATISTICS ===")
print(day_night.round(4).to_string(index=False))

print("\nSaved all Experiment 05 CSV, PNG, PDF and report files.")
print("IMPORTANT: KS p-values are descriptive only.")
