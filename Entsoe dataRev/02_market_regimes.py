
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. FILES
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "ENTSOE_ML_dataset.csv"

OUTPUT_EXCEL = BASE_DIR / "02_market_regime_report.xlsx"
OUTPUT_PLOT = BASE_DIR / "02_market_regime_prices.png"
OUTPUT_VOLATILITY = BASE_DIR / "02_market_regime_volatility.png"

HIGH_PRICE_THRESHOLD = 200.0  # EUR/MWh, illustrative

# =====================================================
# 2. LOAD DATA
# =====================================================

print("\nSTEP 1: LOADING DATA")

df = pd.read_csv(INPUT_FILE)

df["Target_Date_Italy"] = pd.to_datetime(
    df["Target_Date_Italy"]
)

df["Target_Price_EUR_MWh"] = pd.to_numeric(
    df["Target_Price_EUR_MWh"],
    errors="coerce"
)

df["Year"] = df["Target_Date_Italy"].dt.year

df = df.sort_values(
    ["Target_Date_Italy", "Target_UTC"]
).reset_index(drop=True)

print("Observations:", len(df))
print("First date:", df["Target_Date_Italy"].min())
print("Last date:", df["Target_Date_Italy"].max())

# =====================================================
# 3. ANNUAL PRICE STATISTICS
# =====================================================

print("\nSTEP 2: ANNUAL PRICE STATISTICS")

annual = (
    df.groupby("Year")["Target_Price_EUR_MWh"]
    .agg(
        Observations="count",
        Mean_Price="mean",
        Median_Price="median",
        Minimum_Price="min",
        Maximum_Price="max",
        Price_Std="std"
    )
    .reset_index()
)

print(annual.round(2).to_string(index=False))

# =====================================================
# 4. EXTREME AND NEGATIVE PRICES
# =====================================================

print("\nSTEP 3: EXTREME PRICE FREQUENCY")

df["Negative_Price"] = (
    df["Target_Price_EUR_MWh"] < 0
)

df["High_Price"] = (
    df["Target_Price_EUR_MWh"] > HIGH_PRICE_THRESHOLD
)

extremes = (
    df.groupby("Year")
    .agg(
        Total_Hours=("Target_Price_EUR_MWh", "count"),
        Negative_Hours=("Negative_Price", "sum"),
        High_Price_Hours=("High_Price", "sum")
    )
    .reset_index()
)

extremes["Negative_Percent"] = (
    100 * extremes["Negative_Hours"]
    / extremes["Total_Hours"]
)

extremes["High_Price_Percent"] = (
    100 * extremes["High_Price_Hours"]
    / extremes["Total_Hours"]
)

print(extremes.round(2).to_string(index=False))

# =====================================================
# 5. DAILY PRICE AND VOLATILITY STATISTICS
# =====================================================

print("\nSTEP 4: DAILY STATISTICS")

daily = (
    df.groupby("Target_Date_Italy")["Target_Price_EUR_MWh"]
    .agg(
        Daily_Mean="mean",
        Daily_Minimum="min",
        Daily_Maximum="max",
        Daily_Std="std"
    )
    .reset_index()
)

daily["Year"] = daily["Target_Date_Italy"].dt.year

print(daily.head(10).round(2).to_string(index=False))

# =====================================================
# 6. PRICE-LEVEL REGIMES
# =====================================================

print("\nSTEP 5: EXPLORATORY PRICE REGIMES")

# These categories describe realized prices.
# They are NOT yet forecast-time regime labels.

daily["Exploratory_Regime"] = np.select(
    [
        daily["Daily_Mean"] < 50,
        daily["Daily_Mean"].between(
            50, 150, inclusive="left"
        ),
        daily["Daily_Mean"] >= 150
    ],
    [
        "Low-price",
        "Moderate-price",
        "High-price"
    ],
    default="Unknown"
)

regime_counts = (
    daily.groupby(["Year", "Exploratory_Regime"])
    .size()
    .reset_index(name="Number_Of_Days")
)

print(regime_counts.to_string(index=False))

# =====================================================
# 7. PLOT HISTORICAL DAILY PRICES
# =====================================================

print("\nSTEP 6: PLOTTING PRICES")

plt.figure(figsize=(15, 5))

plt.plot(
    daily["Target_Date_Italy"],
    daily["Daily_Mean"],
    linewidth=0.8
)

plt.axhline(
    HIGH_PRICE_THRESHOLD,
    linestyle="--",
    label="Illustrative 200 EUR/MWh threshold"
)

plt.title(
    "Italy North - Historical Daily Average "
    "Day-Ahead Electricity Prices"
)

plt.xlabel("Delivery date")
plt.ylabel("Daily average price (EUR/MWh)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=200)
plt.close()

# =====================================================
# 8. PLOT DAILY VOLATILITY
# =====================================================

print("\nSTEP 7: PLOTTING VOLATILITY")

plt.figure(figsize=(15, 5))

plt.plot(
    daily["Target_Date_Italy"],
    daily["Daily_Std"],
    linewidth=0.8
)

plt.title(
    "Italy North - Within-Day Electricity "
    "Price Standard Deviation"
)

plt.xlabel("Delivery date")
plt.ylabel("Within-day price SD (EUR/MWh)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_VOLATILITY, dpi=200)
plt.close()

# =====================================================
# 9. SAVE RESULTS
# =====================================================

print("\nSTEP 8: SAVING RESULTS")

with pd.ExcelWriter(
    OUTPUT_EXCEL,
    engine="openpyxl"
) as writer:

    annual.to_excel(
        writer,
        sheet_name="Annual Statistics",
        index=False
    )

    extremes.to_excel(
        writer,
        sheet_name="Extreme Prices",
        index=False
    )

    daily.to_excel(
        writer,
        sheet_name="Daily Statistics",
        index=False
    )

    regime_counts.to_excel(
        writer,
        sheet_name="Regime Counts",
        index=False
    )

print("\nFILES SAVED:")
print(OUTPUT_EXCEL)
print(OUTPUT_PLOT)
print(OUTPUT_VOLATILITY)

print("\nMARKET REGIME ANALYSIS COMPLETE")
