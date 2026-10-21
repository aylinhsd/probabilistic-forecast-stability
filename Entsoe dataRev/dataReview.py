
import pandas as pd

# Read the downloaded ENTSO-E file
df = pd.read_csv("entsoe_2025.csv")

# Display the first five rows
print(df.head())


# Check the number of rows and columns
print("\n1. DATASET SIZE:")
print(df.shape)

# Check the column names
print("\n2. COLUMN NAMES:")
print(df.columns.tolist())

# Check the data types
print("\n3. DATA TYPES:")
print(df.dtypes)

# Check the last five rows
print("\n4. LAST FIVE ROWS:")
print(df.tail())



# STEP 5: Convert the timestamp column into datetime

df["Start_Time"] = pd.to_datetime(
    df["MTU (UTC)"].str.split(" - ").str[0],
    format="%d/%m/%Y %H:%M:%S",
    utc=True
)

# Sort observations by time
df = df.sort_values("Start_Time")

# Calculate the time difference between consecutive rows
time_steps = df["Start_Time"].diff()

print("\n5. TIME STEP FREQUENCIES:")
print(time_steps.value_counts())

# Check for duplicate timestamps
print("\n6. DUPLICATE TIMESTAMPS:")
print(df["Start_Time"].duplicated().sum())

# Check for missing day-ahead prices
print("\n7. MISSING DAY-AHEAD PRICES:")
print(df["Day-ahead Price (EUR/MWh)"].isna().sum())


# STEP 8: Investigate the actual price resolution

price_col = "Day-ahead Price (EUR/MWh)"

# Group observations into UTC hours
hourly_groups = df.groupby(
    df["Start_Time"].dt.floor("h")
)

# Count distinct prices within each hour
unique_prices_per_hour = hourly_groups[price_col].nunique()

# Count observations within each hour
rows_per_hour = hourly_groups.size()

print("\n8. NUMBER OF ROWS PER HOUR:")
print(rows_per_hour.value_counts().sort_index())

print("\n9. DISTINCT PRICES PER HOUR:")
print(unique_prices_per_hour.value_counts().sort_index())

# Find hours with more than one price
different_price_hours = unique_prices_per_hour[
    unique_prices_per_hour > 1
]

print("\n10. HOURS WITH MULTIPLE PRICES:")
print(len(different_price_hours))

print("\nEXAMPLES:")
print(different_price_hours.head(10))



# # STEP 9: Create a separate hourly price dataset

# price_col = "Day-ahead Price (EUR/MWh)"

# hourly_df = (
#     df.groupby(df["Start_Time"].dt.floor("h"))[price_col]
#     .first()
#     .reset_index()
# )

# # Rename columns
# hourly_df.columns = ["Timestamp_UTC", "Price_EUR_MWh"]

# print("\n11. HOURLY DATASET SIZE:")
# print(hourly_df.shape)

# print("\n12. FIRST FIVE HOURLY OBSERVATIONS:")
# print(hourly_df.head())

# # Save hourly data
# hourly_df.to_csv(
#     "entsoe_italy_north_2025_hourly.csv",
#     index=False
# )

# print("\nHourly dataset saved successfully!")


# STEP 11: Check full calendar-year coverage

year = 2025

# Expected hourly timestamps for the complete year
expected_times = pd.date_range(
    start=f"{year}-01-01 00:00:00",
    end=f"{year + 1}-01-01 00:00:00",
    freq="h",
    tz="UTC",
    inclusive="left"
)

# Actual timestamps from the downloaded data
actual_times = pd.DatetimeIndex(df["Start_Time"])

# Find missing timestamps
missing_times = expected_times.difference(actual_times)

print("\n13. EXPECTED HOURS:", len(expected_times))
print("14. ACTUAL HOURS:", len(actual_times))
print("15. MISSING HOURS:", len(missing_times))

print("\nFIRST 10 MISSING TIMESTAMPS:")
print(missing_times[:10])

print("\nLAST 10 MISSING TIMESTAMPS:")
print(missing_times[-10:])

print("\nFIRST AVAILABLE TIMESTAMP:")
print(actual_times.min())

print("\nLAST AVAILABLE TIMESTAMP:")
print(actual_times.max())


# STEP 12: Print a compact summary for each year

print("\n========== YEARLY DATA QUALITY SUMMARY ==========")
print("Year:", year)
print("Total rows:", len(df))
print("First timestamp:", df["Start_Time"].min())
print("Last timestamp:", df["Start_Time"].max())
print("Missing prices:", df["Day-ahead Price (EUR/MWh)"].isna().sum())
print("Duplicate timestamps:", df["Start_Time"].duplicated().sum())
print("Time interval frequencies:")
print(df["Start_Time"].diff().value_counts())
print("=================================================")

# STEP 13: Examine price variation within each hour

price_col = "Day-ahead Price (EUR/MWh)"

hourly_groups = df.groupby(
    df["Start_Time"].dt.floor("h")
)

unique_prices_per_hour = hourly_groups[price_col].nunique()

print("\nDISTINCT PRICES PER HOUR:")
print(unique_prices_per_hour.value_counts().sort_index())

print("\nFIRST HOURS WITH MULTIPLE PRICES:")
print(unique_prices_per_hour[unique_prices_per_hour > 1].head(10))

print("\nLAST HOURS WITH ONE PRICE:")
print(unique_prices_per_hour[unique_prices_per_hour == 1].tail(10))

# STEP 14: Inspect the transition to 15-minute prices

price_col = "Day-ahead Price (EUR/MWh)"

transition_data = df[
    (df["Start_Time"] >= "2025-09-30 20:00:00+00:00") &
    (df["Start_Time"] < "2025-10-01 03:00:00+00:00")
]

print("\nPRICE RESOLUTION TRANSITION:")
print(
    transition_data[
        ["Start_Time", price_col]
    ].to_string(index=False)
)
