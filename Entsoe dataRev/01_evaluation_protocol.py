
from pathlib import Path
import pandas as pd
import numpy as np

# =====================================================
# 1. FILE LOCATIONS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "ENTSOE_ML_dataset.csv"

OUTPUT_FILE = BASE_DIR / "01_evaluation_protocol_report.xlsx"

# =====================================================
# 2. LOAD DATA
# =====================================================

print("\nSTEP 1: LOADING DATA")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)
print("\nAvailable columns:")
print(df.columns.tolist())

# =====================================================
# 3. IDENTIFY IMPORTANT COLUMNS
# =====================================================

required = [
    "Target_Date_Italy",
    "Target_UTC",
    "Target_Price_EUR_MWh",
    "Split",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Required columns are missing: {missing}"
    )

df["Target_Date_Italy"] = pd.to_datetime(
    df["Target_Date_Italy"]
).dt.date

df["Target_UTC"] = pd.to_datetime(
    df["Target_UTC"], utc=True
)

df["Target_Price_EUR_MWh"] = pd.to_numeric(
    df["Target_Price_EUR_MWh"],
    errors="coerce"
)

# =====================================================
# 4. CHRONOLOGICAL SPLIT SUMMARY
# =====================================================

print("\nSTEP 2: CHECKING DATA SPLITS")

split_summary = (
    df.groupby("Split", dropna=False)
    .agg(
        First_Delivery_Date=("Target_Date_Italy", "min"),
        Last_Delivery_Date=("Target_Date_Italy", "max"),
        Number_Of_Hours=("Target_UTC", "size"),
        Number_Of_Days=("Target_Date_Italy", "nunique"),
        Missing_Prices=("Target_Price_EUR_MWh",
                        lambda x: x.isna().sum()),
    )
    .reset_index()
)

print(split_summary.to_string(index=False))

# =====================================================
# 5. DUPLICATE TIMESTAMPS
# =====================================================

print("\nSTEP 3: CHECKING DUPLICATES")

duplicate_rows = df[
    df.duplicated("Target_UTC", keep=False)
].copy()

print("Duplicate timestamp rows:", len(duplicate_rows))

# =====================================================
# 6. DAYLIGHT-SAVING TRANSITIONS
# =====================================================

print("\nSTEP 4: CHECKING DELIVERY DAY LENGTH")

day_lengths = (
    df.groupby("Target_Date_Italy")
    .size()
    .reset_index(name="Number_Of_Hours")
)

day_length_summary = (
    day_lengths["Number_Of_Hours"]
    .value_counts()
    .sort_index()
    .reset_index()
)

day_length_summary.columns = [
    "Hours_Per_Day",
    "Number_Of_Days"
]

print(day_length_summary.to_string(index=False))

unusual_days = day_lengths[
    day_lengths["Number_Of_Hours"] != 24
]

print("\nDays with 23 or 25 hours:")
print(unusual_days.to_string(index=False))

# =====================================================
# 7. MISSING FEATURE VALUES
# =====================================================

print("\nSTEP 5: CHECKING MISSING FEATURES")

lag_columns = [
    c for c in df.columns
    if c.startswith("Price_UTC_Lag_")
]

daily_stat_columns = [
    c for c in df.columns
    if c.startswith((
        "mean_D-",
        "minimum_D-",
        "maximum_D-",
        "std_D-"
    ))
]

print("UTC lag features found:", len(lag_columns))
print("Daily-stat features found:", len(daily_stat_columns))

feature_columns = lag_columns + daily_stat_columns

missing_features = pd.DataFrame({
    "Feature": feature_columns,
    "Missing_Values": [
        df[c].isna().sum()
        for c in feature_columns
    ]
})

print(
    missing_features[
        missing_features["Missing_Values"] > 0
    ].to_string(index=False)
)

# =====================================================
# 8. CHECK CUT-OFF ANCHORED LAG CONSISTENCY
# =====================================================

print("\nSTEP 6: CHECKING LAG CONSISTENCY")

# All target hours for the same delivery date
# should have identical cutoff-anchored lag features.

inconsistent_days = []

if lag_columns:
    daily_unique_counts = (
        df.groupby("Target_Date_Italy")[lag_columns]
        .nunique(dropna=False)
    )

    inconsistent_mask = (
        daily_unique_counts > 1
    ).any(axis=1)

    inconsistent_days = (
        daily_unique_counts.index[inconsistent_mask]
        .tolist()
    )

print(
    "Days with inconsistent cutoff-anchored lags:",
    len(inconsistent_days)
)

# =====================================================
# 9. VERIFY CHRONOLOGICAL SPLITS
# =====================================================

print("\nSTEP 7: CHECKING SPLIT YEARS")

expected_splits = {
    "Train": (2015, 2020),
    "Validation": (2021, 2021),
    "Test": (2022, 2023),
    "Robustness": (2024, 2026),
}

year_checks = []

for split_name, (first_year, last_year) in expected_splits.items():

    part = df[df["Split"] == split_name]

    if part.empty:
        year_checks.append({
            "Split": split_name,
            "Expected_Years": f"{first_year}-{last_year}",
            "Actual_Years": "No rows",
            "Correct": False,
        })
        continue

    actual_years = pd.to_datetime(
        part["Target_Date_Italy"].astype(str)
    ).dt.year

    correct = actual_years.between(
        first_year, last_year
    ).all()

    year_checks.append({
        "Split": split_name,
        "Expected_Years": f"{first_year}-{last_year}",
        "Actual_Years": (
            f"{actual_years.min()}-{actual_years.max()}"
        ),
        "Correct": bool(correct),
    })

year_checks_df = pd.DataFrame(year_checks)

print(year_checks_df.to_string(index=False))

# =====================================================
# 10. SAVE REPORT
# =====================================================

print("\nSTEP 8: SAVING EXCEL REPORT")

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    split_summary.to_excel(
        writer,
        sheet_name="Split Summary",
        index=False
    )

    day_length_summary.to_excel(
        writer,
        sheet_name="Day Length Summary",
        index=False
    )

    unusual_days.to_excel(
        writer,
        sheet_name="DST Days",
        index=False
    )

    missing_features.to_excel(
        writer,
        sheet_name="Missing Features",
        index=False
    )

    year_checks_df.to_excel(
        writer,
        sheet_name="Split Year Checks",
        index=False
    )

    pd.DataFrame({
        "Inconsistent_Delivery_Date": inconsistent_days
    }).to_excel(
        writer,
        sheet_name="Lag Consistency",
        index=False
    )

    duplicate_rows.to_excel(
        writer,
        sheet_name="Duplicate Timestamps",
        index=False
    )

print("\nREPORT SAVED:")
print(OUTPUT_FILE)

print("\nEVALUATION PROTOCOL DATA CHECK COMPLETE")
