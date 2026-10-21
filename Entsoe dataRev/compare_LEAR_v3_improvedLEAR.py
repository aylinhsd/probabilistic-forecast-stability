
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# PART 1 — FILES AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "LEAR_model_comparison"
OUT.mkdir(exist_ok=True)

LEVELS = [80, 85, 90]
MODELS = ["Previous LEAR", "LEAR v3"]

# Example week for detailed forecasting graphs
WEEK_START = "2025-07-07"
WEEK_END = "2025-07-13"


def read_csv(name):
    path = ROOT / name

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {name}. Put it in the same folder as this script."
        )

    return pd.read_csv(path, low_memory=False)


# ============================================================
# PART 2 — LOAD AND ALIGN MODEL PREDICTIONS
# ============================================================

new = read_csv("ENTSOE_LEAR_v3_predictions.csv")
old = read_csv("ENTSOE_improved_LEAR_predictions.csv")

for data in (new, old):
    data["Target_UTC"] = pd.to_datetime(
        data["Target_UTC"], utc=True
    )

n = new[
    [
        "Target_UTC",
        "Target_Price_EUR_MWh",
        "LEAR_v3_Prediction_EUR_MWh",
        "Split",
    ]
].rename(
    columns={
        "Target_Price_EUR_MWh": "Actual_v3",
        "LEAR_v3_Prediction_EUR_MWh": "LEAR v3",
        "Split": "Split_v3",
    }
)

o = old[
    [
        "Target_UTC",
        "Target_Price_EUR_MWh",
        "Improved_LEAR_Prediction_EUR_MWh",
    ]
].rename(
    columns={
        "Target_Price_EUR_MWh": "Actual_old",
        "Improved_LEAR_Prediction_EUR_MWh": "Previous LEAR",
    }
)

df = n.merge(
    o,
    on="Target_UTC",
    how="inner",
    validate="one_to_one",
)

df = df.sort_values("Target_UTC").reset_index(drop=True)

if not np.allclose(
    df["Actual_v3"],
    df["Actual_old"],
    rtol=0,
    atol=1e-7,
):
    raise ValueError(
        "Actual prices do not match between prediction files."
    )

df["Actual"] = df["Actual_v3"]

df["Date_Italy"] = df["Target_UTC"].dt.tz_convert(
    "Europe/Rome"
)

df["Delivery_Date"] = df["Date_Italy"].dt.date

df["Month"] = df["Date_Italy"].dt.strftime("%Y-%m")

# Check chronological split boundaries
expected_split = np.where(
    df["Delivery_Date"] <= pd.Timestamp("2024-12-31").date(),
    "Train",
    np.where(
        df["Delivery_Date"] <= pd.Timestamp("2025-06-30").date(),
        "Validation",
        "Test",
    ),
)

if not np.array_equal(
    df["Split_v3"].to_numpy(), expected_split
):
    raise ValueError(
        "LEAR v3 split labels do not match the agreed dates."
    )

# Calibration: January–June 2025
cal = df[
    (df["Delivery_Date"] >= pd.Timestamp("2025-01-01").date())
    & (df["Delivery_Date"] <= pd.Timestamp("2025-06-30").date())
].copy()

# Test: July 2025 onward
test = df[df["Split_v3"] == "Test"].copy()

if len(cal) < 4000 or len(test) < 10000:
    raise ValueError(
        f"Insufficient overlapping data: "
        f"Calibration={len(cal)}, Test={len(test)}"
    )

print("\nDATASET CHECK")
print("Calibration observations:", len(cal))
print("Test observations:", len(test))


# ============================================================
# PART 3 — CALCULATE PROBABILISTIC INTERVALS
# ============================================================

# Each model receives its own intervals.
# Residual quantiles are calculated ONLY from validation data.

for model in MODELS:

    residuals = cal["Actual"] - cal[model]

    for level in LEVELS:

        alpha = 1 - level / 100

        lower_offset, upper_offset = np.quantile(
            residuals,
            [alpha / 2, 1 - alpha / 2],
        )

        test[f"{model} lower {level}"] = (
            test[model] + lower_offset
        )

        test[f"{model} upper {level}"] = (
            test[model] + upper_offset
        )


# ============================================================
# PART 4 — EVALUATION METRICS
# ============================================================

def evaluate(subset, month):

    rows = []

    for model in MODELS:

        actual = subset["Actual"]
        prediction = subset[model]

        error = actual - prediction

        mae = error.abs().mean()
        rmse = np.sqrt((error ** 2).mean())
        bias = error.mean()

        for level in LEVELS:

            alpha = 1 - level / 100

            lower = subset[f"{model} lower {level}"]
            upper = subset[f"{model} upper {level}"]

            covered = (
                (actual >= lower) & (actual <= upper)
            )

            coverage = covered.mean() * 100

            coverage_gap = coverage - level

            width = (upper - lower).mean()

            interval_score = (
                (upper - lower)
                + (2 / alpha)
                * (lower - actual).clip(lower=0)
                + (2 / alpha)
                * (actual - upper).clip(lower=0)
            ).mean()

            rows.append(
                {
                    "Month": month,
                    "Model": model,
                    "Nominal_Coverage_Percent": level,
                    "Hours": len(subset),
                    "MAE_EUR_MWh": mae,
                    "RMSE_EUR_MWh": rmse,
                    "Bias_EUR_MWh": bias,
                    "Observed_Coverage_Percent": coverage,
                    "Coverage_Gap_pp": coverage_gap,
                    "Mean_Width_EUR_MWh": width,
                    "Winkler_Score_EUR_MWh": interval_score,
                }
            )

    return rows


rows = evaluate(test, "Overall")

for month, subset in test.groupby("Month"):
    rows.extend(evaluate(subset, month))

metrics = pd.DataFrame(rows)

metrics.to_csv(
    OUT / "probabilistic_comparison_metrics.csv",
    index=False,
)

test.to_csv(
    OUT / "probabilistic_aligned_predictions.csv",
    index=False,
)


# ============================================================
# PART 5 — NUMERICAL MODEL COMPARISON
# ============================================================

overall = metrics[
    (metrics["Month"] == "Overall")
    & (metrics["Nominal_Coverage_Percent"] == 80)
].set_index("Model")

previous = overall.loc["Previous LEAR"]
v3 = overall.loc["LEAR v3"]


def improvement(previous_value, new_value):
    if previous_value == 0:
        return np.nan

    return (
        (previous_value - new_value)
        / abs(previous_value)
        * 100
    )


mae_improvement = improvement(
    previous["MAE_EUR_MWh"],
    v3["MAE_EUR_MWh"],
)

rmse_improvement = improvement(
    previous["RMSE_EUR_MWh"],
    v3["RMSE_EUR_MWh"],
)

print("\n" + "=" * 65)
print("OVERALL TEST PERFORMANCE")
print("=" * 65)

print(
    f"Previous LEAR MAE: {previous['MAE_EUR_MWh']:.2f} EUR/MWh"
)
print(
    f"LEAR v3 MAE:       {v3['MAE_EUR_MWh']:.2f} EUR/MWh"
)
print(f"MAE improvement:   {mae_improvement:+.2f}%")

print()

print(
    f"Previous LEAR RMSE: {previous['RMSE_EUR_MWh']:.2f} EUR/MWh"
)
print(
    f"LEAR v3 RMSE:       {v3['RMSE_EUR_MWh']:.2f} EUR/MWh"
)
print(f"RMSE improvement:   {rmse_improvement:+.2f}%")

print("\nPREDICTION INTERVAL COVERAGE")

for level in LEVELS:

    part = metrics[
        (metrics["Month"] == "Overall")
        & (metrics["Nominal_Coverage_Percent"] == level)
    ]

    print(f"\nNominal interval: {level}%")

    for _, row in part.iterrows():

        print(
            f"{row['Model']}: "
            f"Coverage={row['Observed_Coverage_Percent']:.2f}% | "
            f"Gap={row['Coverage_Gap_pp']:+.2f} pp | "
            f"Width={row['Mean_Width_EUR_MWh']:.2f} EUR/MWh | "
            f"Winkler={row['Winkler_Score_EUR_MWh']:.2f}"
        )


# ============================================================
# PART 6 — GRAPH 1: OVERALL ACCURACY COMPARISON
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(2)
bar_width = 0.34

old_values = [
    previous["MAE_EUR_MWh"],
    previous["RMSE_EUR_MWh"],
]

new_values = [
    v3["MAE_EUR_MWh"],
    v3["RMSE_EUR_MWh"],
]

bars_old = ax.bar(
    x - bar_width / 2,
    old_values,
    bar_width,
    label="Previous LEAR",
    color="tab:orange",
)

bars_new = ax.bar(
    x + bar_width / 2,
    new_values,
    bar_width,
    label="LEAR v3",
    color="tab:blue",
)

ax.bar_label(bars_old, fmt="%.2f", padding=3)
ax.bar_label(bars_new, fmt="%.2f", padding=3)

ax.set_xticks(x)
ax.set_xticklabels(["MAE", "RMSE"])
ax.set_ylabel("Error (EUR/MWh)")
ax.set_title("Forecast Accuracy — Test Period")
ax.legend()
ax.grid(axis="y", alpha=0.2)

ax.set_ylim(
    0,
    max(old_values + new_values) * 1.25,
)

fig.text(
    0.5,
    0.02,
    f"LEAR v3 improvement: "
    f"MAE {mae_improvement:+.2f}% | "
    f"RMSE {rmse_improvement:+.2f}%",
    ha="center",
    fontsize=11,
)

fig.tight_layout(rect=[0, 0.06, 1, 1])

fig.savefig(
    OUT / "01_overall_accuracy_comparison.png",
    dpi=220,
)

plt.close(fig)


# ============================================================
# PART 7 — GRAPH 2: OVERALL COVERAGE COMPARISON
# ============================================================

fig, axs = plt.subplots(
    1, 3, figsize=(15, 5), sharey=True
)

for ax, level in zip(axs, LEVELS):

    part = metrics[
        (metrics["Month"] == "Overall")
        & (metrics["Nominal_Coverage_Percent"] == level)
    ].set_index("Model")

    values = [
        part.loc["Previous LEAR", "Observed_Coverage_Percent"],
        part.loc["LEAR v3", "Observed_Coverage_Percent"],
    ]

    bars = ax.bar(
        MODELS,
        values,
        color=["tab:orange", "tab:blue"],
        width=0.55,
    )

    ax.bar_label(
        bars,
        labels=[f"{value:.2f}%" for value in values],
        padding=3,
    )

    ax.axhline(
        level,
        color="black",
        linestyle="--",
        label=f"Target {level}%",
    )

    ax.set_title(f"{level}% Prediction Interval")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()

axs[0].set_ylabel("Observed coverage (%)")

fig.suptitle(
    "Prediction Interval Coverage — Test Period",
    fontsize=14,
)

fig.tight_layout()

fig.savefig(
    OUT / "02_overall_coverage_comparison.png",
    dpi=220,
)

plt.close(fig)


# ============================================================
# PART 8 — MONTHLY COMPARISON GRAPHS
# ============================================================

monthly_metrics = [
    (
        "Observed_Coverage_Percent",
        "03_monthly_coverage.png",
        "Observed coverage (%)",
    ),
    (
        "Mean_Width_EUR_MWh",
        "04_monthly_interval_width.png",
        "Mean interval width (EUR/MWh)",
    ),
    (
        "Winkler_Score_EUR_MWh",
        "05_monthly_winkler_score.png",
        "Mean Winkler score (EUR/MWh)",
    ),
]

for metric, filename, ylabel in monthly_metrics:

    fig, axs = plt.subplots(
        3, 1, figsize=(14, 11), sharex=True
    )

    for ax, level in zip(axs, LEVELS):

        for model, color in [
            ("Previous LEAR", "tab:orange"),
            ("LEAR v3", "tab:blue"),
        ]:

            part = metrics[
                (metrics["Month"] != "Overall")
                & (
                    metrics["Nominal_Coverage_Percent"]
                    == level
                )
                & (metrics["Model"] == model)
            ].sort_values("Month")

            ax.plot(
                part["Month"],
                part[metric],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=color,
                label=model,
            )

        if metric == "Observed_Coverage_Percent":

            ax.axhline(
                level,
                linestyle="--",
                color="black",
                alpha=0.6,
                label=f"Target {level}%",
            )

            ax.set_ylim(0, 105)

        ax.set_title(f"{level}% Prediction Interval")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()

    axs[-1].tick_params(axis="x", rotation=60)

    fig.suptitle(
        "Monthly Comparison — Previous LEAR vs LEAR v3"
    )

    fig.tight_layout()

    fig.savefig(
        OUT / filename,
        dpi=220,
    )

    plt.close(fig)


# ============================================================
# PART 9 — GRAPH 6: ALL FOUR ELEMENTS IN ONE GRAPH
# ============================================================

week = test[
    (test["Delivery_Date"] >= pd.Timestamp(WEEK_START).date())
    & (test["Delivery_Date"] <= pd.Timestamp(WEEK_END).date())
].copy()

if week.empty:
    raise ValueError("Selected week is not in the test period.")

fig, ax = plt.subplots(figsize=(16, 7))

# Actual day-ahead electricity price
ax.plot(
    week["Date_Italy"],
    week["Actual"],
    color="black",
    linewidth=1.8,
    label="Actual day-ahead price",
    zorder=5,
)

# Previous LEAR point prediction
ax.plot(
    week["Date_Italy"],
    week["Previous LEAR"],
    color="tab:orange",
    linewidth=1.4,
    linestyle="--",
    label="Previous LEAR",
    zorder=4,
)

# LEAR v3 point prediction
ax.plot(
    week["Date_Italy"],
    week["LEAR v3"],
    color="tab:blue",
    linewidth=1.6,
    label="LEAR v3",
    zorder=4,
)

# 80% interval around LEAR v3
ax.fill_between(
    week["Date_Italy"],
    week["LEAR v3 lower 80"],
    week["LEAR v3 upper 80"],
    color="tab:blue",
    alpha=0.18,
    label="LEAR v3 — 80% prediction interval",
)

# Weekly numerical coverage
week_covered = (
    (week["Actual"] >= week["LEAR v3 lower 80"])
    & (week["Actual"] <= week["LEAR v3 upper 80"])
)

week_coverage = week_covered.mean() * 100

ax.set_title(
    "Actual vs Previous LEAR vs LEAR v3\n"
    f"{WEEK_START} to {WEEK_END} | "
    f"80% Interval | "
    f"Observed weekly coverage: {week_coverage:.2f}%",
    fontsize=14,
)

ax.set_ylabel("Day-ahead electricity price (EUR/MWh)")
ax.set_xlabel("Delivery time — Italy")

ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%d %b", tz=week["Date_Italy"].dt.tz)
)

ax.legend(loc="best")
ax.grid(alpha=0.2)

fig.autofmt_xdate()
fig.tight_layout()

fig.savefig(
    OUT / "06_actual_previous_lear_v3_80_interval.png",
    dpi=240,
)

plt.close(fig)


# ============================================================
# PART 10 — SEPARATE WEEKLY INTERVAL COMPARISON
# ============================================================

for level in (80, 90):

    fig, axs = plt.subplots(
        2, 1,
        figsize=(15, 9),
        sharex=True,
        sharey=True,
    )

    for ax, model in zip(axs, MODELS):

        ax.plot(
            week["Date_Italy"],
            week["Actual"],
            color="black",
            linewidth=1.5,
            label="Actual",
        )

        ax.plot(
            week["Date_Italy"],
            week[model],
            linewidth=1.4,
            label=model,
        )

        ax.fill_between(
            week["Date_Italy"],
            week[f"{model} lower {level}"],
            week[f"{model} upper {level}"],
            alpha=0.22,
            label=f"{level}% prediction interval",
        )

        covered = (
            (week["Actual"] >= week[f"{model} lower {level}"])
            & (week["Actual"] <= week[f"{model} upper {level}"])
        )

        observed = covered.mean() * 100

        ax.set_title(
            f"{model} | "
            f"Weekly coverage: {observed:.2f}% "
            f"(Target: {level}%)"
        )

        ax.set_ylabel("Price (EUR/MWh)")
        ax.legend()
        ax.grid(alpha=0.2)

    axs[-1].xaxis.set_major_locator(
        mdates.DayLocator()
    )

    axs[-1].xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%d %b",
            tz=week["Date_Italy"].dt.tz,
        )
    )

    fig.suptitle(
        f"Probabilistic Forecast Comparison — "
        f"{level}% Intervals\n"
        f"{WEEK_START} to {WEEK_END}",
        fontsize=14,
    )

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        OUT / f"07_weekly_{level}_intervals.png",
        dpi=220,
    )

    plt.close(fig)


# ============================================================
# PART 11 — SAVE SUMMARY REPORT
# ============================================================

summary = pd.DataFrame(
    [
        {
            "Model": model,
            "MAE_EUR_MWh": overall.loc[
                model, "MAE_EUR_MWh"
            ],
            "RMSE_EUR_MWh": overall.loc[
                model, "RMSE_EUR_MWh"
            ],
            "Bias_EUR_MWh": overall.loc[
                model, "Bias_EUR_MWh"
            ],
            "Coverage_80_Percent": metrics.loc[
                (metrics["Month"] == "Overall")
                & (metrics["Model"] == model)
                & (metrics["Nominal_Coverage_Percent"] == 80),
                "Observed_Coverage_Percent",
            ].iloc[0],
        }
        for model in MODELS
    ]
)

summary.to_csv(
    OUT / "overall_model_summary.csv",
    index=False,
)

print("\n" + "=" * 65)
print("DONE")
print("=" * 65)

print("\nAll comparison graphs and CSV files saved in:")
print(OUT)

print("\nMain graph:")
print("06_actual_previous_lear_v3_80_interval.png")

print("\nMAE improvement: ", f"{mae_improvement:+.2f}%")
print("RMSE improvement:", f"{rmse_improvement:+.2f}%")
