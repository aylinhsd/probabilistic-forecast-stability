
from pathlib import Path
import pandas as pd

# ==================================================
# PART 1 — FILE LOCATIONS
# ==================================================

folder = Path(__file__).resolve().parent

input_file = folder / "ENTSOE_NORD_hourly_forecasting.csv"

output_blocks = folder / "ENTSOE_daily_blocks.csv"
output_summary = folder / "ENTSOE_daily_blocks_summary.xlsx"

# ==================================================
# PART 2 — LOAD THE HOURLY DATASET
# ==================================================

df = pd.read_csv(input_file)

# Remove extra semicolons and spaces from column names
df.columns = df.columns.str.strip().str.strip(";").str.strip()

print("CSV columns:", df.columns.tolist())

# Parse UTC timestamps.
df["Timestamp_UTC"] = pd.to_datetime(
    df["Timestamp_UTC"],
    utc=True,
    errors="raise"
)

# Recalculate Italian timestamps from UTC.
# This preserves daylight-saving transitions.
df["Timestamp_Italy"] = (
    df["Timestamp_UTC"].dt.tz_convert("Europe/Rome")
)

# Make sure prices are numeric.
# Remove extra semicolons and spaces from price values
df["Price_EUR_MWh"] = pd.to_numeric(
    df["Price_EUR_MWh"]
    .astype("string")
    .str.strip()
    .str.strip(";")
    .str.strip(),
    errors="raise"
)

df = df.sort_values("Timestamp_UTC").reset_index(drop=True)

# Check for duplicate hourly timestamps.
if df["Timestamp_UTC"].duplicated().any():
    raise RuntimeError("Duplicate UTC timestamps found.")

# Check for missing prices.
if df["Price_EUR_MWh"].isna().any():
    raise RuntimeError("Missing prices found.")

# ==================================================
# PART 3 — CREATE ITALIAN LOCAL DATES
# ==================================================

df["Date_Italy"] = (
    df["Timestamp_Italy"].dt.strftime("%Y-%m-%d")
)

df["Hour_Italy"] = df["Timestamp_Italy"].dt.hour

# Number observations sequentially within each day.
# This handles repeated local hours in autumn.
df["Hour_Position"] = (
    df.groupby("Date_Italy").cumcount() + 1
)

# ==================================================
# PART 4 — CHECK EXPECTED HOURS PER DAY
# ==================================================

daily = (
    df.groupby("Date_Italy")
    .agg(
        Actual_Hours=("Price_EUR_MWh", "size"),
        First_UTC=("Timestamp_UTC", "min"),
        Last_UTC=("Timestamp_UTC", "max"),
        Average_Price=("Price_EUR_MWh", "mean"),
        Min_Price=("Price_EUR_MWh", "min"),
        Max_Price=("Price_EUR_MWh", "max")
    )
    .reset_index()
)

# Determine the correct number of hours for each
# Italian calendar day using local midnight boundaries.
local_start = pd.to_datetime(
    daily["Date_Italy"]
).dt.tz_localize("Europe/Rome")

next_local_start = (
    pd.to_datetime(daily["Date_Italy"])
    + pd.Timedelta(days=1)
).dt.tz_localize("Europe/Rome")

daily["Expected_Hours"] = (
    (next_local_start - local_start)
    .dt.total_seconds() / 3600
).astype(int)

daily["Complete"] = (
    daily["Actual_Hours"] == daily["Expected_Hours"]
)

# ==================================================
# PART 5 — IDENTIFY USABLE DAILY BLOCKS
# ==================================================

# Map daily completeness back to hourly observations.
df = df.merge(
    daily[["Date_Italy", "Expected_Hours", "Complete"]],
    on="Date_Italy",
    how="left",
    validate="many_to_one"
)

# Retain only complete days as forecasting targets.
usable = df.loc[df["Complete"]].copy()

# ==================================================
# PART 6 — SAVE DAILY BLOCKS
# ==================================================

usable[
    [
        "Date_Italy",
        "Hour_Position",
        "Hour_Italy",
        "Timestamp_UTC",
        "Timestamp_Italy",
        "Price_EUR_MWh"
    ]
].to_csv(
    output_blocks,
    index=False
)

# ==================================================
# PART 7 — SAVE DAILY SUMMARY
# ==================================================

# Excel cannot store timezone-aware timestamps.
daily_report = daily.copy()

for col in ["First_UTC", "Last_UTC"]:
    daily_report[col] = daily_report[col].astype(str)

overview = pd.DataFrame([{
    "Total Calendar Days": len(daily),
    "Complete Days": int(daily["Complete"].sum()),
    "Incomplete Days": int((~daily["Complete"]).sum()),
    "Normal 24-hour Days": int(
        (daily["Expected_Hours"] == 24).sum()
    ),
    "Spring 23-hour Days": int(
        (daily["Expected_Hours"] == 23).sum()
    ),
    "Autumn 25-hour Days": int(
        (daily["Expected_Hours"] == 25).sum()
    ),
    "First Date": daily["Date_Italy"].min(),
    "Last Date": daily["Date_Italy"].max()
}])

with pd.ExcelWriter(
    output_summary,
    engine="openpyxl"
) as writer:

    overview.to_excel(
        writer,
        sheet_name="Overview",
        index=False
    )

    daily_report.to_excel(
        writer,
        sheet_name="Daily Summary",
        index=False
    )

    daily_report.loc[
        ~daily_report["Complete"]
    ].to_excel(
        writer,
        sheet_name="Incomplete Days",
        index=False
    )

# ==================================================
# PART 8 — PRINT RESULTS
# ==================================================

print("\n========== DAILY BLOCKS ==========")

print("Total calendar days:", len(daily))

print(
    "Complete days:",
    int(daily["Complete"].sum())
)

print(
    "Incomplete days:",
    int((~daily["Complete"]).sum())
)

print(
    "23-hour days:",
    int((daily["Expected_Hours"] == 23).sum())
)

print(
    "25-hour days:",
    int((daily["Expected_Hours"] == 25).sum())
)

print("\nFirst date:", daily["Date_Italy"].min())
print("Last date:", daily["Date_Italy"].max())

print("\nFiles created:")
print(output_blocks)
print(output_summary)

print("\nDONE!")
