
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

# ==================================================
# EXPERIMENT 06: INTERVAL RELIABILITY
# ==================================================

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "monitoring_research"
TABLES = BASE / "tables"
FIGURES = BASE / "figures"
REPORTS = BASE / "reports"

for folder in (TABLES, FIGURES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

INPUT = ROOT / "ENTSOE_Probabilistic_LEAR_v3_predictions.csv"

LEVELS = [0.80, 0.85, 0.90]
ALPHA = 0.05

# Minimum observations for descriptive test reporting.
# This does not guarantee asymptotic validity.
MIN_N = 100


def plot_day_night_coverage(daynight):
    """Show day/night coverage directly, with exact percentages and targets."""
    colors = {"Day": "#64B6AC", "Night": "#9183B8"}
    ink = "#203344"
    periods = ["Reference", "Evaluation"]
    dates = {"Reference": "Jul–Dec 2025", "Evaluation": "Jan–13 Oct 2026"}
    levels = [int(round(level * 100)) for level in LEVELS]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True, facecolor="white")
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.20, top=0.76, wspace=0.16)
    fig.suptitle("Day vs night coverage", y=0.97, fontsize=20, weight="bold", color=ink)
    handles = [matplotlib.patches.Patch(facecolor=colors[s], label=s)
               for s in ("Day", "Night")]
    handles.append(matplotlib.lines.Line2D([], [], color=ink, linestyle="--", label="Target"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.915),
               ncol=3, frameon=False, fontsize=11)

    x = np.arange(len(levels))
    for ax, period in zip(axes, periods):
        sub = daynight.loc[daynight["period"].eq(period)]
        ax.set_axisbelow(True)
        for segment, offset in (("Day", -0.18), ("Night", 0.18)):
            group = sub.loc[sub["group"].eq(segment)].set_index("interval").loc[levels]
            values = group["observed_coverage"].to_numpy() * 100
            bars = ax.bar(x + offset, values, width=0.34, color=colors[segment], zorder=2)
            ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values],
                         padding=5, fontsize=10.5, weight="bold", color=ink,
                         zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
        # Each pair has its own short target line.
        ax.hlines(levels, x - 0.39, x + 0.39, colors=ink,
                  linestyles="--", linewidth=1.6, zorder=3)
        ax.set_xticks(x, [f"{level}%" for level in levels])
        ax.set_ylim(0, 106)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_xlabel("Target coverage", fontsize=11, labelpad=10)
        ax.set_title(f"{period}\n{dates[period]}", fontsize=12, color=ink, pad=13)
        ax.tick_params(axis="both", labelsize=11, length=0, pad=7)
        ax.grid(axis="y", color="#E3E8EC", linewidth=0.7)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#CCD5DC")

    axes[0].set_ylabel("Actual coverage (%)", fontsize=11, labelpad=10)
    fig.text(0.5, 0.055,
             "Day: 06:00–17:59  |  Night: 18:00–05:59  |  Italian local time",
             ha="center", fontsize=10, color="#526374")
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES / f"experiment_06_day_night_coverage.{extension}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


parser = argparse.ArgumentParser(description="Experiment 06: interval reliability")
parser.add_argument("--day-night-only", action="store_true",
                    help="Redraw only the day/night PNG and PDF from the existing summary table")
args = parser.parse_args()
if args.day_night_only:
    plot_day_night_coverage(pd.read_csv(TABLES / "experiment_06_day_night.csv"))
    print("Updated experiment_06_day_night_coverage.png and .pdf from the saved results.")
    raise SystemExit(0)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

raw = pd.read_csv(INPUT, low_memory=False)

required = [
    "Target_UTC",
    "Target_Price_EUR_MWh",
]

for col in required:
    if col not in raw.columns:
        raise ValueError(f"Missing column: {col}")

df = pd.DataFrame({
    "target_utc": pd.to_datetime(
        raw["Target_UTC"], utc=True
    ),
    "price": pd.to_numeric(
        raw["Target_Price_EUR_MWh"],
        errors="coerce"
    )
})

local = df["target_utc"].dt.tz_convert("Europe/Rome")

df["date"] = (
    local.dt.tz_localize(None).dt.normalize()
)
df["hour"] = local.dt.hour

df = df[
    (df["date"] >= "2025-07-01")
    & (df["date"] <= "2026-10-13")
].copy()

df["period"] = np.where(
    df["date"] <= "2025-12-31",
    "Reference",
    "Evaluation"
)

df["day_night"] = np.where(
    df["hour"].between(6, 17),
    "Day",
    "Night"
)

if df["target_utc"].duplicated().any():
    raise ValueError("Duplicate UTC target timestamps")

if df["price"].isna().any():
    raise ValueError("Missing observed prices")

# --------------------------------------------------
# 2. CONSTRUCT PREDICTION INTERVALS
# --------------------------------------------------

quantile_map = {}

for col in raw.columns:
    if col.startswith("Q_"):
        try:
            probability = float(col[2:])
            quantile_map[probability] = col
        except ValueError:
            pass

q_levels = np.array(sorted(quantile_map))
q_columns = [quantile_map[q] for q in q_levels]

if len(q_columns) < 3:
    raise ValueError("Insufficient quantile columns")

q_values = raw.loc[df.index, q_columns].apply(
    pd.to_numeric, errors="coerce"
).to_numpy(dtype=float)

if not np.isfinite(q_values).all():
    raise ValueError("Missing or non-finite quantile values")

crossing = np.any(np.diff(q_values, axis=1) < -1e-9, axis=1)

if crossing.any():
    raise ValueError(
        f"{crossing.sum()} forecasts contain crossing quantiles"
    )

def get_quantile(probability):
    """Interpolate forecast quantiles if necessary."""
    if probability < q_levels[0] or probability > q_levels[-1]:
        raise ValueError("Requested quantile outside available grid")

    return np.array([
        np.interp(probability, q_levels, row)
        for row in q_values
    ])

for level in LEVELS:
    lower_p = (1 - level) / 2
    upper_p = 1 - lower_p

    lower = get_quantile(lower_p)
    upper = get_quantile(upper_p)

    tag = int(round(level * 100))

    df[f"lower_{tag}"] = lower
    df[f"upper_{tag}"] = upper

    # A violation occurs strictly outside the interval.
    df[f"violation_{tag}"] = (
        (df["price"] < lower) |
        (df["price"] > upper)
    ).astype(int)

df = df.sort_values("target_utc").reset_index(drop=True)

# --------------------------------------------------
# 3. STATISTICAL TEST FUNCTIONS
# --------------------------------------------------

def log_bernoulli(k, n, p):
    """
    Bernoulli log-likelihood.

    Correctly handles p=0 and p=1.
    """
    if n == 0:
        return 0.0

    if p == 0:
        return 0.0 if k == 0 else -np.inf

    if p == 1:
        return 0.0 if k == n else -np.inf

    return k * np.log(p) + (n-k) * np.log1p(-p)


def test_statistics(violations, nominal):
    """
    Kupiec unconditional coverage (UC),
    Christoffersen independence (IND),
    and conditional coverage (CC).

    The input is an ordered binary violation sequence.
    """
    v = np.asarray(violations, dtype=int)
    n = len(v)
    x = int(v.sum())

    expected_violation = 1 - nominal
    observed_violation = x / n
    observed_coverage = 1 - observed_violation

    # Kupiec UC
    ll_null = log_bernoulli(
        x, n, expected_violation
    )
    ll_alt = log_bernoulli(
        x, n, observed_violation
    )

    lr_uc = max(0.0, 2 * (ll_alt - ll_null))
    p_uc = chi2.sf(lr_uc, 1)

    # Transition counts for first-order Markov test.
    previous = v[:-1]
    current = v[1:]

    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    n0 = n00 + n01
    n1 = n10 + n11

    pi01 = n01 / n0 if n0 else np.nan
    pi11 = n11 / n1 if n1 else np.nan

    # Standard IND test needs both previous states.
    ind_valid = n0 > 0 and n1 > 0

    if ind_valid:
        pooled = (n01 + n11) / (n0 + n1)

        ll_ind_null = (
            log_bernoulli(n01+n11, n0+n1, pooled)
        )

        ll_ind_alt = (
            log_bernoulli(n01, n0, pi01)
            + log_bernoulli(n11, n1, pi11)
        )

        lr_ind = max(
            0.0, 2 * (ll_ind_alt - ll_ind_null)
        )
        p_ind = chi2.sf(lr_ind, 1)

        lr_cc = lr_uc + lr_ind
        p_cc = chi2.sf(lr_cc, 2)
    else:
        lr_ind = np.nan
        p_ind = np.nan
        lr_cc = np.nan
        p_cc = np.nan

    if observed_coverage > nominal:
        direction = "Overcoverage"
    elif observed_coverage < nominal:
        direction = "Undercoverage"
    else:
        direction = "Exact nominal"

    return {
        "n": n,
        "violations": x,
        "nominal_coverage": nominal,
        "observed_coverage": observed_coverage,
        "signed_coverage_error_pp":
            100 * (observed_coverage - nominal),
        "absolute_coverage_error_pp":
            100 * abs(observed_coverage - nominal),
        "direction": direction,
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "p_violation_after_no_violation": pi01,
        "p_violation_after_violation": pi11,
        "LR_UC": lr_uc,
        "p_UC": p_uc,
        "LR_IND": lr_ind,
        "p_IND": p_ind,
        "LR_CC": lr_cc,
        "p_CC": p_cc,
        "UC_reject_5pct": bool(p_uc < ALPHA),
        "IND_reject_5pct":
            bool(p_ind < ALPHA) if ind_valid else np.nan,
        "CC_reject_5pct":
            bool(p_cc < ALPHA) if ind_valid else np.nan
    }


# --------------------------------------------------
# 4. OVERALL, DAY/NIGHT, AND HOURLY TESTS
# --------------------------------------------------

records = []

def analyze(subset, group_type, group_name, period):
    subset = subset.sort_values("target_utc")

    for level in LEVELS:
        tag = int(round(level * 100))
        violations = subset[
            f"violation_{tag}"
        ].to_numpy()

        if len(violations) < MIN_N:
            continue

        result = test_statistics(violations, level)

        records.append({
            "group_type": group_type,
            "group": str(group_name),
            "period": period,
            "interval": tag,
            **result
        })


for period, period_df in df.groupby("period"):
    analyze(period_df, "Overall", "All", period)

    for segment, sub in period_df.groupby("day_night"):
        analyze(sub, "DayNight", segment, period)

    for hour, sub in period_df.groupby("hour"):
        analyze(sub, "Hour", f"{hour:02d}", period)

results = pd.DataFrame(records)

# Note: IND transitions in filtered day/night sequences
# connect consecutive observations within that subset.
# They are not necessarily adjacent delivery hours.

# --------------------------------------------------
# 5. SAVE TABLES
# --------------------------------------------------

results.to_csv(
    TABLES / "experiment_06_all_tests.csv",
    index=False
)

overall = results[
    results["group_type"] == "Overall"
].copy()

daynight = results[
    results["group_type"] == "DayNight"
].copy()

hourly = results[
    results["group_type"] == "Hour"
].copy()

overall.to_csv(
    TABLES / "experiment_06_overall.csv",
    index=False
)

daynight.to_csv(
    TABLES / "experiment_06_day_night.csv",
    index=False
)

hourly.to_csv(
    TABLES / "experiment_06_hourly.csv",
    index=False
)

daily = (
    df.groupby(["date", "period"])
    .agg(
        n=("price", "size"),
        violations80=("violation_80", "sum"),
        violations85=("violation_85", "sum"),
        violations90=("violation_90", "sum")
    )
    .reset_index()
)

for tag in [80, 85, 90]:
    daily[f"coverage{tag}"] = (
        1 - daily[f"violations{tag}"] / daily["n"]
    )

daily.to_csv(
    TABLES / "experiment_06_daily_coverage.csv",
    index=False
)

# --------------------------------------------------
# 6. FIGURES
# --------------------------------------------------

def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(
        FIGURES / f"{name}.png",
        dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        FIGURES / f"{name}.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)


# Figure 1: Overall nominal vs observed coverage
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, period in zip(
    axes, ["Reference", "Evaluation"]
):
    sub = overall[
        overall["period"] == period
    ].sort_values("interval")

    x = np.arange(len(sub))

    ax.bar(
        x - 0.18,
        sub["nominal_coverage"] * 100,
        width=0.36,
        label="Nominal"
    )

    ax.bar(
        x + 0.18,
        sub["observed_coverage"] * 100,
        width=0.36,
        label="Observed"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{v}%" for v in sub["interval"]]
    )
    ax.set_ylim(65, 100)
    ax.set_ylabel("Coverage (%)")
    ax.set_title(period)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

fig.suptitle("LEAR v3 — Nominal vs Observed Coverage")
save_figure(fig, "experiment_06_coverage_comparison")


# Figure 2: Day/night observed coverage and nominal targets
plot_day_night_coverage(daynight)


# Figure 3: Hourly signed coverage error, 80% interval
fig, ax = plt.subplots(figsize=(13, 5))

for period in ["Reference", "Evaluation"]:
    sub = hourly[
        (hourly["period"] == period)
        & (hourly["interval"] == 80)
    ].copy()

    sub["hour_number"] = sub["group"].astype(int)
    sub = sub.sort_values("hour_number")

    ax.plot(
        sub["hour_number"],
        sub["signed_coverage_error_pp"],
        marker="o",
        label=period
    )

ax.axhline(0, color="black", linestyle="--")
ax.set_xticks(range(24))
ax.set_xlabel("Italian local delivery hour")
ax.set_ylabel("Coverage error (percentage points)")
ax.set_title("LEAR v3 — Hourly 80% Coverage Error")
ax.legend()
ax.grid(alpha=0.2)

save_figure(fig, "experiment_06_hourly_coverage")


# Figure 4: Rolling 30-day observed coverage
fig, ax = plt.subplots(figsize=(13, 5))

daily = daily.sort_values("date")

for tag in [80, 85, 90]:
    rolling_violations = (
        daily[f"violations{tag}"]
        .rolling(30, min_periods=15)
        .sum()
    )
    rolling_n = (
        daily["n"]
        .rolling(30, min_periods=15)
        .sum()
    )

    daily[f"rolling_coverage{tag}"] = (
        1 - rolling_violations / rolling_n
    )

    ax.plot(
        daily["date"],
        daily[f"rolling_coverage{tag}"],
        label=f"Observed {tag}%"
    )
    ax.axhline(
        tag / 100,
        linestyle="--",
        alpha=0.6
    )

ax.axvline(
    pd.Timestamp("2026-01-01"),
    color="gray",
    linestyle=":"
)

ax.set_title("LEAR v3 — Rolling 30-Day Coverage")
ax.set_ylabel("Observed coverage")
ax.legend()
ax.grid(alpha=0.2)

save_figure(fig, "experiment_06_rolling_coverage")

daily.to_csv(
    TABLES / "experiment_06_daily_coverage.csv",
    index=False
)

# --------------------------------------------------
# 7. RESEARCH REPORT
# --------------------------------------------------

report = f"""# Experiment 06 — Interval Reliability Tests

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

{overall.round(5).to_string(index=False)}

## Day/night results

{daynight.round(5).to_string(index=False)}

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
"""

(REPORTS / "experiment_06_report.md").write_text(
    report, encoding="utf-8"
)

# --------------------------------------------------
# 8. TERMINAL SUMMARY
# --------------------------------------------------

print("\n=== EXPERIMENT 06 ===")
print("Hourly observations:", len(df))

print("\n=== OVERALL RESULTS ===")
print(
    overall[[
        "period",
        "interval",
        "n",
        "observed_coverage",
        "signed_coverage_error_pp",
        "direction",
        "p_UC",
        "p_IND",
        "p_CC"
    ]].round(5).to_string(index=False)
)

print("\n=== DAY/NIGHT RESULTS ===")
print(
    daynight[[
        "period",
        "group",
        "interval",
        "observed_coverage",
        "signed_coverage_error_pp",
        "p_UC",
        "p_IND",
        "p_CC"
    ]].round(5).to_string(index=False)
)

print("\nSaved:")
print("5 CSV tables")
print("4 PNG figures")
print("4 PDF figures")
print("1 Markdown research report")

print("\nIMPORTANT:")
print("Test p-values are screening diagnostics.")
print("Both overcoverage and undercoverage matter.")
