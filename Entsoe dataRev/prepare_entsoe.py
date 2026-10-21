
from pathlib import Path
import pandas as pd
import re

# ============================================================
# PART 1 — FILE LOCATIONS
# ============================================================

folder = Path(__file__).resolve().parent

output_csv = folder / "ENTSOE_NORD_hourly_forecasting.csv"
output_report = folder / "ENTSOE_preparation_report.xlsx"

# Read only the original yearly CSV files.
files = sorted(folder.glob("entsoe_20??.csv"))

if not files:
    raise RuntimeError("No yearly ENTSO-E CSV files found.")

print(f"\nFound {len(files)} yearly files.\n")

datasets = []
file_summary = []
issues = []

# ============================================================
# PART 2 — READ AND CHECK EACH YEAR
# ============================================================

for file in files:

    print(f"Reading: {file.name}")

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    time_col = "MTU (UTC)"
    price_col = "Day-ahead Price (EUR/MWh)"

    if time_col not in df.columns or price_col not in df.columns:
        raise RuntimeError(
            f"Required columns missing in {file.name}"
        )

    # Original ENTSO-E interval example:
    # 04/01/2015 23:00:00 - 05/01/2015 00:00:00

    interval_parts = (
        df[time_col]
        .astype("string")
        .str.strip()
        .str.split(" - ", n=1, expand=True)
    )

    if interval_parts.shape[1] != 2:
        raise RuntimeError(
            f"Invalid interval format in {file.name}"
        )

    # Parse both start and end times in UTC.
    start = pd.to_datetime(
        interval_parts[0].str.strip(),
        format="%d/%m/%Y %H:%M:%S",
        utc=True,
        errors="coerce"
    )

    end = pd.to_datetime(
        interval_parts[1].str.strip(),
        format="%d/%m/%Y %H:%M:%S",
        utc=True,
        errors="coerce"
    )

    prices = pd.to_numeric(
        df[price_col].astype("string")
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )

    data = pd.DataFrame({
        "Timestamp_UTC": start,
        "End_UTC": end,
        "Price_EUR_MWh": prices
    })

    data["Duration_Minutes"] = (
        (data["End_UTC"] - data["Timestamp_UTC"])
        .dt.total_seconds() / 60
    )

    # Validate timestamps and prices.
    invalid_time = (
        data["Timestamp_UTC"].isna()
        | data["End_UTC"].isna()
    )

    invalid_duration = (
        ~data["Duration_Minutes"].isin([15, 60])
    )

    missing_prices = data["Price_EUR_MWh"].isna()

    duplicate_starts = (
        data["Timestamp_UTC"].duplicated().sum()
    )

    file_summary.append({
        "File": file.name,
        "Observations": len(data),
        "First Timestamp UTC": data["Timestamp_UTC"].min(),
        "Last Timestamp UTC": data["Timestamp_UTC"].max(),
        "15-minute Rows": int(
            (data["Duration_Minutes"] == 15).sum()
        ),
        "60-minute Rows": int(
            (data["Duration_Minutes"] == 60).sum()
        ),
        "Invalid Timestamps": int(invalid_time.sum()),
        "Invalid Durations": int(invalid_duration.sum()),
        "Missing Prices": int(missing_prices.sum()),
        "Duplicate Start Times": int(duplicate_starts)
    })

    if (
        invalid_time.any()
        or invalid_duration.any()
        or missing_prices.any()
        or duplicate_starts > 0
    ):
        issues.append({
            "File": file.name,
            "Issue": "Invalid or duplicate source observations",
            "Details": (
                f"Invalid timestamps: {invalid_time.sum()}, "
                f"invalid durations: {invalid_duration.sum()}, "
                f"missing prices: {missing_prices.sum()}, "
                f"duplicate starts: {duplicate_starts}"
            )
        })

    datasets.append(data)

# ============================================================
# PART 3 — COMBINE ALL YEARS
# ============================================================

combined = pd.concat(
    datasets,
    ignore_index=True
)

combined = combined.sort_values(
    "Timestamp_UTC"
).reset_index(drop=True)

# Stop if original data is invalid.
if issues:
    print("\nSource data issues detected:")
    for item in issues:
        print(item)

    raise RuntimeError(
        "Source data validation failed. "
        "Review the issues above before continuing."
    )

# ============================================================
# PART 4 — CHECK SOURCE INTERVAL CONTINUITY
# ============================================================

# Consecutive delivery intervals should not overlap.
previous_end = combined["End_UTC"].shift(1)

overlap = (
    previous_end.notna()
    & (combined["Timestamp_UTC"] < previous_end)
)

if overlap.any():
    examples = combined.loc[
        overlap,
        ["Timestamp_UTC", "End_UTC"]
    ].head(10)

    print("\nOverlapping source intervals:")
    print(examples)

    raise RuntimeError(
        f"{overlap.sum()} overlapping intervals found."
    )

# Identify any gaps between source intervals.
gap = (
    previous_end.notna()
    & (combined["Timestamp_UTC"] > previous_end)
)

source_gaps = combined.loc[
    gap,
    ["Timestamp_UTC"]
].copy()

source_gaps["Previous_End_UTC"] = previous_end.loc[gap]

source_gaps["Gap_Hours"] = (
    source_gaps["Timestamp_UTC"]
    - source_gaps["Previous_End_UTC"]
).dt.total_seconds() / 3600

# ============================================================
# PART 5 — CONVERT DATA TO HOURLY RESOLUTION
# ============================================================

combined["Hour_UTC"] = (
    combined["Timestamp_UTC"].dt.floor("h")
)

# Count observations and their durations per hour.
hourly = (
    combined.groupby("Hour_UTC")
    .agg(
        Price_EUR_MWh=("Price_EUR_MWh", "mean"),
        Observations=("Price_EUR_MWh", "size"),
        Total_Minutes=("Duration_Minutes", "sum"),
        Minimum_Duration=("Duration_Minutes", "min"),
        Maximum_Duration=("Duration_Minutes", "max")
    )
    .reset_index()
)

# Valid hourly groups:
# 1 observation of 60 minutes
# OR
# 4 observations of 15 minutes

valid_hourly = (
    (hourly["Observations"] == 1)
    & (hourly["Total_Minutes"] == 60)
    & (hourly["Minimum_Duration"] == 60)
)

valid_quarter_hourly = (
    (hourly["Observations"] == 4)
    & (hourly["Total_Minutes"] == 60)
    & (hourly["Minimum_Duration"] == 15)
    & (hourly["Maximum_Duration"] == 15)
)

invalid_hours = hourly.loc[
    ~(valid_hourly | valid_quarter_hourly)
]

if not invalid_hours.empty:

    invalid_hours.to_csv(
        folder / "ENTSOE_invalid_hourly_groups.csv",
        index=False
    )

    raise RuntimeError(
        f"{len(invalid_hours)} invalid hourly groups. "
        "See ENTSOE_invalid_hourly_groups.csv."
    )

# ============================================================
# PART 6 — CHECK MISSING HOURS
# ============================================================

expected_hours = pd.date_range(
    start=hourly["Hour_UTC"].min(),
    end=hourly["Hour_UTC"].max(),
    freq="h",
    tz="UTC"
)

missing_hours = expected_hours.difference(
    pd.DatetimeIndex(hourly["Hour_UTC"])
)

missing_df = pd.DataFrame({
    "Missing_Hour_UTC": missing_hours
})

# ============================================================
# PART 7 — PREPARE FINAL FORECASTING DATASET
# ============================================================

hourly = hourly.rename(
    columns={"Hour_UTC": "Timestamp_UTC"}
)

# Italian local timestamps are useful later
# for calendar features and daily forecasting.
hourly["Timestamp_Italy"] = (
    hourly["Timestamp_UTC"]
    .dt.tz_convert("Europe/Rome")
)

hourly["Date_Italy"] = (
    hourly["Timestamp_Italy"]
    .dt.strftime("%Y-%m-%d")
)

hourly["Hour_Italy"] = (
    hourly["Timestamp_Italy"].dt.hour
)

# Keep a simple final structure.
final = hourly[
    [
        "Timestamp_UTC",
        "Timestamp_Italy",
        "Date_Italy",
        "Hour_Italy",
        "Price_EUR_MWh"
    ]
].copy()

final = final.sort_values(
    "Timestamp_UTC"
).reset_index(drop=True)

# ============================================================
# PART 8 — SAVE CSV AND VALIDATION REPORT
# ============================================================

final.to_csv(
    output_csv,
    index=False
)

# Summarize the hourly conversion.
conversion_summary = pd.DataFrame([{
    "Source Files": len(files),
    "Original Observations": len(combined),
    "Final Hourly Observations": len(final),
    "Hours From 60-minute Data": int(
        valid_hourly.sum()
    ),
    "Hours From 15-minute Data": int(
        valid_quarter_hourly.sum()
    ),
    "Missing Hours Within Coverage": len(missing_hours),
    "Source Interval Gaps": len(source_gaps),
    "First Timestamp UTC": final["Timestamp_UTC"].min(),
    "Last Timestamp UTC": final["Timestamp_UTC"].max()
}])

# Excel cannot store timezone-aware datetime values.
# Convert timestamps to strings in report tables.
def excel_safe(df):
    result = df.copy()

    for col in result.columns:
        if isinstance(result[col].dtype, pd.DatetimeTZDtype):
            result[col] = result[col].astype(str)

    return result

with pd.ExcelWriter(
    output_report,
    engine="openpyxl"
) as writer:

    excel_safe(conversion_summary).to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

    excel_safe(pd.DataFrame(file_summary)).to_excel(
        writer,
        sheet_name="Yearly Files",
        index=False
    )

    excel_safe(missing_df).to_excel(
        writer,
        sheet_name="Missing Hours",
        index=False
    )

    excel_safe(source_gaps).to_excel(
        writer,
        sheet_name="Source Gaps",
        index=False
    )

    for sheet in writer.sheets.values():
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions

        for column in sheet.columns:
            letter = column[0].column_letter

            width = max(
                len(str(cell.value or ""))
                for cell in column
            ) + 2

            sheet.column_dimensions[letter].width = min(
                max(width, 14),
                42
            )

# ============================================================
# PART 9 — PRINT FINAL RESULTS
# ============================================================

print("\n========== ENTSO-E PREPARATION ==========")

print("Source files:", len(files))
print("Original observations:", len(combined))
print("Final hourly observations:", len(final))

print(
    "First timestamp:",
    final["Timestamp_UTC"].min()
)

print(
    "Last timestamp:",
    final["Timestamp_UTC"].max()
)

print("Missing hours:", len(missing_hours))
print("Source interval gaps:", len(source_gaps))

print("\nFiles created:")
print(output_csv)
print(output_report)

print("\nDONE!")
