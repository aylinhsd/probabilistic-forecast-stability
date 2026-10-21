
from pathlib import Path
import pandas as pd
import re

# 1- Folder containing annual CSV files
data_folder = Path("/Users/aylin/Documents/FBK/ENTSO-E dataRev")

# Only original annual files, e.g. entsoe_2025.csv
csv_files = sorted(data_folder.glob("entsoe_20??.csv"))

results = []

for file in csv_files:

    print(f"Checking: {file.name}")

    df = pd.read_csv(file)

    required_columns = {
        "MTU (UTC)",
        "Day-ahead Price (EUR/MWh)"
    }

    if not required_columns.issubset(df.columns):
        print("  Skipped: missing required columns")
        continue

    # STEP 2: Parse timestamps in UTC
    df["Start_Time"] = pd.to_datetime(
        df["MTU (UTC)"].str.split(" - ").str[0],
        format="%d/%m/%Y %H:%M:%S",
        utc=True
    )

    # Keep Italian local time as a separate column
    df["Italian_Time"] = df["Start_Time"].dt.tz_convert(
        "Europe/Rome"
    )

    df = df.sort_values("Start_Time")

    # Extract year from filename
    year = int(re.search(r"20\d{2}", file.stem).group())

    price_col = "Day-ahead Price (EUR/MWh)"

    # STEP 3: General data-quality checks
    total_rows = len(df)
    missing_prices = df[price_col].isna().sum()
    duplicates = df["Start_Time"].duplicated().sum()

    intervals = df["Start_Time"].diff().dropna()

    if intervals.empty:
        print("  Skipped: insufficient observations")
        continue

    # Most common time interval
    resolution = intervals.mode().iloc[0]

    # STEP 4: Determine expected timestamps
    # For 2026, check only through the last downloaded interval.
    # For earlier years, check the entire UTC calendar year.

    
    # STEP 4: Determine expected timestamps

    # Explicitly define the start of the year in UTC
    expected_start = pd.Timestamp(
        f"{year}-01-01",
        tz="UTC"
    )

    # For 2026, stop at the last available observation.
    # For historical years, use the full calendar year.
    if year == 2026:
        expected_end = df["Start_Time"].max() + resolution
    else:
        expected_end = pd.Timestamp(
            f"{year + 1}-01-01",
            tz="UTC"
        )

    # Ensure the end timestamp is in UTC
    expected_end = expected_end.tz_convert("UTC")

    # Generate expected timestamps
    expected_times = pd.date_range(
        start=expected_start,
        end=expected_end,
        freq=resolution,
        inclusive="left"
    )

    # Compare with actual timestamps
    actual_times = pd.DatetimeIndex(df["Start_Time"])

    missing_times = expected_times.difference(actual_times)

    # STEP 5: Detect hours with multiple distinct prices
    distinct_prices = df.groupby(
        df["Start_Time"].dt.floor("h")
    )[price_col].nunique()

    multiple_price_hours = distinct_prices[
        distinct_prices > 1
    ]

    count_multiple = len(multiple_price_hours)

    if count_multiple > 0:
        first_hour_italy = str(
            multiple_price_hours.index.min().tz_convert(
                "Europe/Rome"
            )
        )

        last_hour_italy = str(
            multiple_price_hours.index.max().tz_convert(
                "Europe/Rome"
            )
        )
    else:
        first_hour_italy = ""
        last_hour_italy = ""

    # STEP 6: Store yearly summary
    results.append({
        "Year": year,
        "First data timestamp (UTC)": str(
            df["Start_Time"].min()
        ),
        "Last data timestamp (UTC)": str(
            df["Start_Time"].max()
        ),
        "Rows": total_rows,
        "Resolution": str(resolution),
        "Missing timestamps": len(missing_times),
        "Missing prices": missing_prices,
        "Duplicates": duplicates,
        "Hours with multiple prices": count_multiple,
        "First observed hour (Italy)": first_hour_italy,
        "Last observed hour (Italy)": last_hour_italy
    })

# STEP 7: Create and save the combined summary
summary = pd.DataFrame(results)

if not summary.empty:

    summary = summary.sort_values("Year")

    print("\n========== ALL YEARS SUMMARY ==========")
    print(summary.to_string(index=False))
    print("=======================================")

    output_file = (
        data_folder / "entsoe_summary_2015-2026.csv"
    )

    summary.to_csv(
        output_file,
        index=False
    )

    print(f"\nSummary saved to: {output_file}")

    # STEP 8: Show years containing multiple prices per hour
    multiple_summary = summary[
        summary["Hours with multiple prices"] > 0
    ]

    print("\n====== YEARS WITH MULTIPLE PRICES PER HOUR ======")

    if not multiple_summary.empty:
        print(
            multiple_summary[
                [
                    "Year",
                    "Hours with multiple prices",
                    "First observed hour (Italy)",
                    "Last observed hour (Italy)"
                ]
            ].to_string(index=False)
        )
    else:
        print("No hours with multiple distinct prices found.")

    print("================================================")

else:
    print("No matching ENTSO-E CSV files found.")
