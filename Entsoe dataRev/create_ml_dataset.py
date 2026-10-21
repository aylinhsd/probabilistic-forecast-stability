
from pathlib import Path
import pandas as pd

# ==================================================
# PART 1 — FILE LOCATIONS AND SETTINGS
# ==================================================

folder = Path(__file__).resolve().parent

input_file = folder / "ENTSOE_daily_blocks.csv"
output_file = folder / "ENTSOE_ML_dataset.csv"
report_file = folder / "ENTSOE_ML_dataset_report.xlsx"

TIMEZONE = "Europe/Rome"

HISTORY_DAYS = 7
HISTORY_HOURS = 24 * HISTORY_DAYS


# ==================================================
# PART 2 — LOAD AND VALIDATE INPUT DATA
# ==================================================

df = pd.read_csv(input_file)

required_columns = [
    "Timestamp_UTC",
    "Date_Italy",
    "Hour_Italy",
    "Hour_Position",
    "Price_EUR_MWh"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

# Parse timestamps and dates from the CSV.
df["Timestamp_UTC"] = pd.to_datetime(
    df["Timestamp_UTC"],
    utc=True
)

df["Date_Italy"] = pd.to_datetime(
    df["Date_Italy"]
).dt.normalize()

# Validate numeric values.
df["Price_EUR_MWh"] = pd.to_numeric(
    df["Price_EUR_MWh"],
    errors="raise"
)

df["Hour_Position"] = pd.to_numeric(
    df["Hour_Position"],
    errors="raise"
).astype(int)

if df[required_columns].isna().any().any():
    raise ValueError("Input contains missing values.")

df = df.sort_values(
    "Timestamp_UTC"
).reset_index(drop=True)

if df["Timestamp_UTC"].duplicated().any():
    raise ValueError("Duplicate UTC timestamps.")

# Verify continuous hourly UTC timestamps.
differences = df["Timestamp_UTC"].diff().dropna()

if not differences.eq(
    pd.Timedelta(hours=1)
).all():
    raise ValueError(
        "Missing or irregular UTC hourly observations."
    )

# Verify Italian dates against UTC timestamps.
calculated_dates = (
    df["Timestamp_UTC"]
    .dt.tz_convert(TIMEZONE)
    .dt.tz_localize(None)
    .dt.normalize()
)

if not calculated_dates.equals(df["Date_Italy"]):
    raise ValueError(
        "Italian dates do not match UTC timestamps."
    )

# Count observations in each Italian delivery day.
daily_counts = df.groupby("Date_Italy").size()

expected_counts = {}

for day in daily_counts.index:

    start_utc = (
        day.tz_localize(TIMEZONE)
        .tz_convert("UTC")
    )

    next_day_utc = (
        (day + pd.Timedelta(days=1))
        .tz_localize(TIMEZONE)
        .tz_convert("UTC")
    )

    expected_counts[day] = int(
        (next_day_utc - start_utc)
        / pd.Timedelta(hours=1)
    )

expected_counts = pd.Series(expected_counts)

if not daily_counts.equals(expected_counts):
    raise ValueError(
        "Incomplete Italian delivery days found."
    )

# Verify hour positions: 1, 2, ..., 23/24/25.
expected_positions = (
    df.groupby("Date_Italy").cumcount() + 1
)

if not df["Hour_Position"].equals(
    expected_positions.rename("Hour_Position")
):
    raise ValueError(
        "Hour positions are not sequential."
    )

print("PART 2: Input validation passed.")


# ==================================================
# PART 3 — CREATE HISTORICAL FEATURES
# ==================================================

# Historical daily price statistics.
daily_stats = (
    df.groupby("Date_Italy")["Price_EUR_MWh"]
    .agg(
        mean="mean",
        minimum="min",
        maximum="max",
        std="std"
    )
)

# Price lookup indexed by UTC timestamp.
price_lookup = df.set_index(
    "Timestamp_UTC"
)["Price_EUR_MWh"]

# One record per Italian target delivery day.
days = pd.DataFrame({
    "Target_Date_Italy": daily_counts.index
})

feature_columns = []

# A. Historical daily statistics: D-2 through D-8.
for days_back in range(2, HISTORY_DAYS + 2):

    history_date = (
        days["Target_Date_Italy"]
        - pd.Timedelta(days=days_back)
    )

    for statistic in [
        "mean", "minimum", "maximum", "std"
    ]:

        column = f"{statistic}_D-{days_back}"

        days[column] = history_date.map(
            daily_stats[statistic]
        )

        feature_columns.append(column)


# ==================================================
# PART 4 — FIXED FORECAST TIME AND UTC CUTOFF
# ==================================================

# Forecast D at 10:00 Italian time on D-1.
previous_day = (
    days["Target_Date_Italy"]
    - pd.Timedelta(days=1)
)

# Create local 10:00, then convert to UTC.
issue_local = (
    previous_day + pd.Timedelta(hours=10)
).dt.tz_localize(TIMEZONE)

days["Issue_Time_UTC"] = (
    issue_local.dt.tz_convert("UTC")
)

# Historical input cutoff:
# 00:00 Italian time at the beginning of D-1.
cutoff_local = previous_day.dt.tz_localize(
    TIMEZONE
)

days["Input_Cutoff_UTC"] = (
    cutoff_local.dt.tz_convert("UTC")
)

if not (
    days["Input_Cutoff_UTC"]
    < days["Issue_Time_UTC"]
).all():
    raise ValueError(
        "Input cutoff must precede issuance."
    )


# ==================================================
# PART 3 CONTINUED — UTC HOURLY LAG FEATURES
# ==================================================

# Lag 1 = last hourly price before the cutoff.
# Lag 2 = price two UTC hours before the cutoff.
# ...
# Lag 168 = price 168 UTC hours before the cutoff.
#
# Every lookup timestamp is earlier than the
# cutoff, regardless of daylight-saving changes.

for lag in range(1, HISTORY_HOURS + 1):

    column = f"Price_UTC_Lag_{lag}h"

    history_timestamps = (
        days["Input_Cutoff_UTC"]
        - pd.Timedelta(hours=lag)
    )

    days[column] = history_timestamps.map(
        price_lookup
    )

    feature_columns.append(column)


# ==================================================
# PART 5 — CREATE COMPLETE FORECASTING BLOCKS
# ==================================================

# Check historical features for each target day.
days["Features_Complete"] = (
    days[feature_columns].notna().all(axis=1)
)

excluded_days = days.loc[
    ~days["Features_Complete"],
    "Target_Date_Italy"
]

usable_days = days.loc[
    days["Features_Complete"]
].copy()

# Verify that all hourly historical feature timestamps
# occur strictly before the input cutoff.
#
# This is guaranteed by construction, but checked
# explicitly to detect future code changes.

for lag in range(1, HISTORY_HOURS + 1):

    history_timestamp = (
        usable_days["Input_Cutoff_UTC"]
        - pd.Timedelta(hours=lag)
    )

    if not (
        history_timestamp
        < usable_days["Input_Cutoff_UTC"]
    ).all():
        raise ValueError(
            f"Invalid historical timestamp at lag {lag}."
        )

# Join daily features to every hourly target.
ml = df.rename(columns={
    "Date_Italy": "Target_Date_Italy",
    "Timestamp_UTC": "Target_UTC",
    "Price_EUR_MWh": "Target_Price_EUR_MWh"
}).merge(
    usable_days,
    on="Target_Date_Italy",
    how="inner",
    validate="many_to_one"
)

# Calendar information known in advance.
ml["Day_Of_Week"] = (
    ml["Target_Date_Italy"].dt.dayofweek
)

ml["Month"] = (
    ml["Target_Date_Italy"].dt.month
)

# Use UTC to calculate target clock hour in Italy.
ml["Target_Hour_Italy"] = (
    ml["Target_UTC"]
    .dt.tz_convert(TIMEZONE)
    .dt.hour
)

# Check complete target delivery days.
final_counts = ml.groupby(
    "Target_Date_Italy"
).size()

expected_final = expected_counts.reindex(
    final_counts.index
)

if not final_counts.equals(expected_final):
    raise ValueError(
        "One or more target days lost hourly observations."
    )

# Check that the target occurs after issuance.
if not (
    ml["Target_UTC"]
    > ml["Issue_Time_UTC"]
).all():
    raise ValueError(
        "Target timestamps must follow issuance."
    )

ml = ml.sort_values(
    "Target_UTC"
).reset_index(drop=True)

print("PARTS 3–5: Feature validation passed.")


# ==================================================
# PART 6 — CHRONOLOGICAL DATA SPLITS
# ==================================================

def assign_split(date):

    if date.year <= 2020:
        return "Train"

    elif date.year == 2021:
        return "Validation"

    elif date.year in [2022, 2023]:
        return "Test"

    else:
        return "Robustness"


ml["Split"] = ml["Target_Date_Italy"].apply(
    assign_split
)


# ==================================================
# PART 7 — SAVE DATASET AND REPORT
# ==================================================

ml.to_csv(output_file, index=False)

summary = (
    ml.groupby("Split")
    .agg(
        Hourly_Rows=(
            "Target_Price_EUR_MWh", "size"
        ),
        Delivery_Days=(
            "Target_Date_Italy", "nunique"
        ),
        First_Date=(
            "Target_Date_Italy", "min"
        ),
        Last_Date=(
            "Target_Date_Italy", "max"
        )
    )
    .reset_index()
)

overview = pd.DataFrame({
    "Metric": [
        "Original hourly observations",
        "Original complete days",
        "Historical UTC hours",
        "Historical daily statistics",
        "Excluded target days",
        "Final target days",
        "Final target hours",
        "23-hour target days",
        "24-hour target days",
        "25-hour target days"
    ],
    "Value": [
        len(df),
        len(daily_counts),
        HISTORY_HOURS,
        HISTORY_DAYS * 4,
        len(excluded_days),
        len(final_counts),
        len(ml),
        int((final_counts == 23).sum()),
        int((final_counts == 24).sum()),
        int((final_counts == 25).sum())
    ]
})

excluded_report = pd.DataFrame({
    "Excluded_Target_Date": excluded_days
})

with pd.ExcelWriter(
    report_file,
    engine="openpyxl"
) as writer:

    overview.to_excel(
        writer,
        sheet_name="Overview",
        index=False
    )

    summary.to_excel(
        writer,
        sheet_name="Data Splits",
        index=False
    )

    excluded_report.to_excel(
        writer,
        sheet_name="Excluded Days",
        index=False
    )

print("\n========== FINAL ML DATASET ==========")

print("Original hours:", len(df))
print("Original days:", len(daily_counts))
print("Excluded days:", len(excluded_days))
print("Final days:", len(final_counts))
print("Final hours:", len(ml))

print("\nDelivery day lengths:")
print(final_counts.value_counts().sort_index())

print("\nChronological splits:")
print(summary.to_string(index=False))

print("\nExcluded dates:")
print(excluded_days.to_list())

print("\nFiles created:")
print(output_file)
print(report_file)

print("\nDONE!")
