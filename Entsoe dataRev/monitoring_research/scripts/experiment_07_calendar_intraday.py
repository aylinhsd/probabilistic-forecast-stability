
#!/usr/bin/env python3
"""
Experiment 07 — Calendar and Intraday Stability Analysis

Compare LEAR forecasting performance across Italian delivery hours,
weekdays, weekends, national holidays, bridge days and seasons.

Uses hourly forecasts and realized prices.
Does not retrain or modify LEAR.
"""

from pathlib import Path
from datetime import date, timedelta
from dateutil.easter import easter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Paths and research periods
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "monitoring_research"

TABLES = RESEARCH / "tables"
FIGURES = RESEARCH / "figures"
REPORTS = RESEARCH / "reports"

for folder in (TABLES, FIGURES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

INPUT = ROOT / "ENTSOE_Probabilistic_LEAR_v3_predictions.csv"

CAL_START = pd.Timestamp("2025-07-01")
CAL_END = pd.Timestamp("2025-12-31")
EVAL_START = pd.Timestamp("2026-01-01")
EVAL_END = pd.Timestamp("2026-10-13")

ALPHA = 0.20  # Central 80% prediction interval


# ------------------------------------------------------------
# 2. Italian national holidays
# ------------------------------------------------------------

def italian_holidays(year):
    """National public holidays; excludes local patron-saint days."""
    e = easter(year)

    fixed = [
        (1, 1), (1, 6), (4, 25), (5, 1),
        (6, 2), (8, 15), (11, 1), (12, 8),
        (12, 25), (12, 26),
    ]

    days = {date(year, month, day) for month, day in fixed}
    days.add(e)
    days.add(e + timedelta(days=1))
    return days


HOLIDAYS = set()
for year in (2025, 2026):
    HOLIDAYS.update(italian_holidays(year))


def is_bridge_day(d):
    """
    Working-day bridge between a holiday and a weekend:
    Friday following a Thursday holiday, or
    Monday preceding a Tuesday holiday.

    Does not classify other holiday-adjacent weekdays as bridges.
    """
    if d.weekday() >= 5 or d in HOLIDAYS:
        return False

    if d.weekday() == 4:
        return d - timedelta(days=1) in HOLIDAYS

    if d.weekday() == 0:
        return d + timedelta(days=1) in HOLIDAYS

    return False


def season(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


# ------------------------------------------------------------
# 3. Load forecasts
# ------------------------------------------------------------

if not INPUT.exists():
    raise FileNotFoundError(f"Prediction file not found: {INPUT}")

df = pd.read_csv(INPUT, low_memory=False)

required = [
    "Target_Date_Italy",
    "Target_Hour_Italy",
    "Target_Price_EUR_MWh",
    "Q_0.100",
    "Q_0.500",
    "Q_0.900",
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["date"] = pd.to_datetime(
    df["Target_Date_Italy"], errors="coerce"
).dt.normalize()

df["hour"] = pd.to_numeric(
    df["Target_Hour_Italy"], errors="coerce"
)

numeric_columns = [
    "Target_Price_EUR_MWh",
    "Q_0.100",
    "Q_0.500",
    "Q_0.900",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[
    df["date"].between(CAL_START, EVAL_END)
].copy()

df = df.dropna(
    subset=["date", "hour"] + numeric_columns
).copy()

df = df[df["hour"].between(0, 23)].copy()
df["hour"] = df["hour"].astype(int)

# Exclude malformed intervals.
valid = (
    (df["Q_0.100"] <= df["Q_0.500"]) &
    (df["Q_0.500"] <= df["Q_0.900"])
)

excluded_intervals = int((~valid).sum())
df = df.loc[valid].copy()

df["period"] = np.where(
    df["date"] <= CAL_END,
    "Calibration",
    "Evaluation",
)

# ------------------------------------------------------------
# 4. Calculate forecast performance
# ------------------------------------------------------------

y = df["Target_Price_EUR_MWh"]
median = df["Q_0.500"]
lower = df["Q_0.100"]
upper = df["Q_0.900"]

df["absolute_error"] = (y - median).abs()

df["covered80"] = (
    (y >= lower) & (y <= upper)
).astype(float)

df["width80"] = upper - lower

df["interval_score80"] = (
    df["width80"]
    + (2 / ALPHA) * (lower - y).clip(lower=0)
    + (2 / ALPHA) * (y - upper).clip(lower=0)
)

# ------------------------------------------------------------
# 5. Calendar variables
# ------------------------------------------------------------

df["weekday_number"] = df["date"].dt.weekday
df["weekday"] = df["date"].dt.day_name()
df["month"] = df["date"].dt.month

df["season"] = df["month"].map(season)

df["is_weekend"] = df["weekday_number"] >= 5

local_dates = df["date"].dt.date

df["is_holiday"] = local_dates.isin(HOLIDAYS)
df["is_bridge"] = local_dates.map(is_bridge_day)

df["day_type"] = np.select(
    [
        df["is_holiday"],
        df["is_bridge"],
        df["is_weekend"],
    ],
    [
        "Holiday",
        "Bridge day",
        "Weekend",
    ],
    default="Regular weekday",
)

# These categories are mutually exclusive.
# Holidays falling on weekends count as holidays.

df["time_block"] = pd.cut(
    df["hour"],
    bins=[-1, 5, 11, 17, 23],
    labels=[
        "Night 00-05",
        "Morning 06-11",
        "Afternoon 12-17",
        "Evening 18-23",
    ],
)

# ------------------------------------------------------------
# 6. Group summaries
# ------------------------------------------------------------

def summarize(group_columns):
    result = (
        df.groupby(
            ["period"] + group_columns,
            observed=True,
            dropna=False,
        )
        .agg(
            observations=("covered80", "size"),
            days=("date", "nunique"),
            mae=("absolute_error", "mean"),
            coverage80=("covered80", "mean"),
            width80=("width80", "mean"),
            interval_score80=("interval_score80", "mean"),
        )
        .reset_index()
    )

    result["coverage_error_pp"] = (
        result["coverage80"] - 0.80
    ) * 100

    return result


summaries = {
    "overall": summarize([]),
    "hourly": summarize(["hour"]),
    "weekday": summarize(["weekday_number", "weekday"]),
    "day_type": summarize(["day_type"]),
    "season": summarize(["season"]),
    "time_block": summarize(["time_block"]),
    "weekday_hour": summarize(["weekday_number", "hour"]),
    "day_type_hour": summarize(["day_type", "hour"]),
    "season_hour": summarize(["season", "hour"]),
}

for name, table in summaries.items():
    path = TABLES / f"experiment_07_{name}.csv"
    table.to_csv(path, index=False)


# ------------------------------------------------------------
# 7. Plotting helpers
# ------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "font.size": 10,
})


def save_figure(name):
    plt.savefig(
        FIGURES / f"experiment_07_{name}.png",
        bbox_inches="tight",
    )
    plt.savefig(
        FIGURES / f"experiment_07_{name}.pdf",
        bbox_inches="tight",
    )
    plt.close()


periods = ["Calibration", "Evaluation"]
colors = ["tab:blue", "tab:orange"]


def grouped_bar(table, category, value, title, ylabel, name, order):
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(order))
    width = 0.38

    for i, period in enumerate(periods):
        subset = table[table["period"] == period]
        values = (
            subset.set_index(category)
            .reindex(order)[value]
        )

        ax.bar(
            x + (i - 0.5) * width,
            values,
            width,
            label=period,
            color=colors[i],
        )

    if value == "coverage_error_pp":
        ax.axhline(0, color="black", linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    save_figure(name)


# ------------------------------------------------------------
# 8. Hourly plots
# ------------------------------------------------------------

hourly = summaries["hourly"]

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for period, color in zip(periods, colors):
    subset = hourly[hourly["period"] == period].sort_values("hour")

    axes[0].plot(
        subset["hour"],
        subset["mae"],
        marker="o",
        markersize=3,
        label=period,
        color=color,
    )

    axes[1].plot(
        subset["hour"],
        subset["coverage_error_pp"],
        marker="o",
        markersize=3,
        label=period,
        color=color,
    )

axes[0].set_ylabel("MAE (EUR/MWh)")
axes[0].set_title("LEAR accuracy by Italian delivery hour")

axes[1].axhline(0, color="black", linestyle="--")
axes[1].set_ylabel("80% coverage error (percentage points)")
axes[1].set_xlabel("Italian local delivery hour")

for ax in axes:
    ax.set_xticks(range(24))
    ax.grid(alpha=0.2)
    ax.legend()

fig.tight_layout()
save_figure("hourly_performance")


# ------------------------------------------------------------
# 9. Weekday, day type and season plots
# ------------------------------------------------------------

weekdays = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

day_types = [
    "Regular weekday",
    "Weekend",
    "Holiday",
    "Bridge day",
]

seasons = ["Winter", "Spring", "Summer", "Autumn"]

grouped_bar(
    summaries["weekday"],
    "weekday",
    "coverage_error_pp",
    "80% interval calibration by weekday",
    "Coverage error (percentage points)",
    "weekday_coverage",
    weekdays,
)

grouped_bar(
    summaries["day_type"],
    "day_type",
    "mae",
    "Forecast accuracy by calendar day type",
    "MAE (EUR/MWh)",
    "day_type_mae",
    day_types,
)

grouped_bar(
    summaries["day_type"],
    "day_type",
    "coverage_error_pp",
    "80% interval calibration by calendar day type",
    "Coverage error (percentage points)",
    "day_type_coverage",
    day_types,
)

grouped_bar(
    summaries["season"],
    "season",
    "mae",
    "Forecast accuracy by season",
    "MAE (EUR/MWh)",
    "season_mae",
    seasons,
)

grouped_bar(
    summaries["season"],
    "season",
    "coverage_error_pp",
    "80% interval calibration by season",
    "Coverage error (percentage points)",
    "season_coverage",
    seasons,
)


# ------------------------------------------------------------
# 10. Weekday x hour heatmaps
# ------------------------------------------------------------

weekday_hour = summaries["weekday_hour"]

fig, axes = plt.subplots(
    1, 2, figsize=(15, 5), sharey=True,
    constrained_layout=True,
)

matrices = []

for period in periods:
    subset = weekday_hour[weekday_hour["period"] == period]

    matrix = subset.pivot(
        index="weekday_number",
        columns="hour",
        values="coverage_error_pp",
    ).reindex(
        index=range(7),
        columns=range(24),
    )

    matrices.append(matrix)

# Common symmetric color scale for fair visual comparison.
all_values = np.concatenate([
    matrix.to_numpy().ravel() for matrix in matrices
])

finite_values = all_values[np.isfinite(all_values)]

limit = (
    max(5, np.max(np.abs(finite_values)))
    if len(finite_values)
    else 20
)

for ax, period, matrix in zip(axes, periods, matrices):
    im = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap="RdBu",
        vmin=-limit,
        vmax=limit,
    )

    ax.set_title(period)
    ax.set_xlabel("Italian local delivery hour")
    ax.set_xticks(range(0, 24, 2))
    ax.set_yticks(range(7))
    ax.set_yticklabels(weekdays)

axes[0].set_ylabel("Weekday")

fig.colorbar(
    im,
    ax=axes,
    label="80% coverage error (percentage points)",
    shrink=0.85,
)

save_figure("weekday_hour_heatmap")


# ------------------------------------------------------------
# 11. Day type x hour heatmaps
# ------------------------------------------------------------

day_type_hour = summaries["day_type_hour"]

fig, axes = plt.subplots(
    1, 2, figsize=(15, 4.5), sharey=True,
    constrained_layout=True,
)

matrices = []

for period in periods:
    subset = day_type_hour[day_type_hour["period"] == period]

    matrix = subset.pivot(
        index="day_type",
        columns="hour",
        values="coverage_error_pp",
    ).reindex(
        index=day_types,
        columns=range(24),
    )

    matrices.append(matrix)

all_values = np.concatenate([
    matrix.to_numpy().ravel() for matrix in matrices
])

finite_values = all_values[np.isfinite(all_values)]

limit = (
    max(5, np.max(np.abs(finite_values)))
    if len(finite_values)
    else 20
)

for ax, period, matrix in zip(axes, periods, matrices):
    im = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap="RdBu",
        vmin=-limit,
        vmax=limit,
    )

    ax.set_title(period)
    ax.set_xlabel("Italian local delivery hour")
    ax.set_xticks(range(0, 24, 2))
    ax.set_yticks(range(len(day_types)))
    ax.set_yticklabels(day_types)

fig.colorbar(
    im,
    ax=axes,
    label="80% coverage error (percentage points)",
    shrink=0.85,
)

save_figure("day_type_hour_heatmap")


# ------------------------------------------------------------
# 12. Save hourly analysis data
# ------------------------------------------------------------

export_columns = [
    "date",
    "period",
    "hour",
    "weekday",
    "season",
    "day_type",
    "is_weekend",
    "is_holiday",
    "is_bridge",
    "absolute_error",
    "covered80",
    "width80",
    "interval_score80",
]

df[export_columns].to_csv(
    TABLES / "experiment_07_hourly_data.csv",
    index=False,
)


# ------------------------------------------------------------
# 13. Markdown research report
# ------------------------------------------------------------

def markdown_table(table):
    return table.round(4).to_markdown(index=False)


overall = summaries["overall"]
day_type = summaries["day_type"]
season_table = summaries["season"]
time_block = summaries["time_block"]

report = f"""# Experiment 07 — Calendar and Intraday Stability Analysis

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
- Valid hourly observations: {len(df)}
- Excluded malformed prediction intervals: {excluded_intervals}

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

{markdown_table(overall)}

## Calendar day types

{markdown_table(day_type)}

## Seasons

{markdown_table(season_table)}

## Intraday time blocks

{markdown_table(time_block)}

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
"""


(REPORTS / "experiment_07_report.md").write_text(
    report, encoding="utf-8"
)

# ------------------------------------------------------------
# 14. Console summary
# ------------------------------------------------------------

print("\nEXPERIMENT 07 — COMPLETE")
print("=" * 60)

print(f"Valid hourly observations: {len(df)}")
print(f"Excluded malformed intervals: {excluded_intervals}")

print("\nOverall performance:")
print(overall.round(4).to_string(index=False))

print("\nCalendar day-type performance:")
print(day_type.round(4).to_string(index=False))

print("\nSeasonal performance:")
print(season_table.round(4).to_string(index=False))

print("\nOutputs saved to:")
print(TABLES)
print(FIGURES)
print(REPORTS)
