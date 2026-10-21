
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# PART 1 — FILE LOCATION
# ==========================================

folder = Path(__file__).resolve().parent

input_file = folder / "ENTSOE_LEAR_style_predictions.csv"

# ==========================================
# PART 2 — READ DATA
# ==========================================

df = pd.read_csv(input_file)

df["Target_UTC"] = pd.to_datetime(
    df["Target_UTC"],
    utc=True
)

# Convert to Italian local calendar dates
df["Date_Italy"] = (
    df["Target_UTC"]
    .dt.tz_convert("Europe/Rome")
    .dt.date
)

# ==========================================
# PART 3 — SELECT EVALUATION PERIODS
# ==========================================

periods = [
    "Validation",
    "Test",
    "Robustness"
]

# ==========================================
# PART 4 — CREATE ONE GRAPH PER PERIOD
# ==========================================

for split in periods:

    subset = df[
        df["Split"] == split
    ].copy()

    if subset.empty:
        print(f"No data found for {split}")
        continue

    # Calculate MAE and RMSE using ALL hours
    actual = subset["Target_Price_EUR_MWh"]
    predicted = subset["LEAR_Style_Prediction_EUR_MWh"]

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    # Aggregate prices to daily averages for display
    daily = (
        subset.groupby("Date_Italy")
        .agg(
            Actual=("Target_Price_EUR_MWh", "mean"),
            Predicted=("LEAR_Style_Prediction_EUR_MWh", "mean")
        )
        .reset_index()
    )

    daily["Date_Italy"] = pd.to_datetime(
        daily["Date_Italy"]
    )

    # ======================================
    # PART 5 — PLOT
    # ======================================

    plt.figure(figsize=(15, 6))

    plt.plot(
        daily["Date_Italy"],
        daily["Actual"],
        label="Actual daily average price",
        linewidth=1.5
    )

    plt.plot(
        daily["Date_Italy"],
        daily["Predicted"],
        label="LEAR predicted daily average price",
        linewidth=1.5,
        linestyle="--"
    )

    plt.title(
        f"LEAR Forecast vs Actual Prices — {split}\n"
        f"Hourly MAE = {mae:.2f} EUR/MWh | "
        f"Hourly RMSE = {rmse:.2f} EUR/MWh"
    )

    plt.xlabel("Italian delivery date")
    plt.ylabel("Daily average electricity price (EUR/MWh)")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    # ======================================
    # PART 6 — SAVE GRAPH
    # ======================================

    output_file = folder / f"LEAR_Full_{split}.png"

    plt.savefig(output_file, dpi=300)

    print(f"\n{split}")
    print(f"Hours: {len(subset)}")
    print(f"Days: {len(daily)}")
    print(f"Hourly MAE: {mae:.2f} EUR/MWh")
    print(f"Hourly RMSE: {rmse:.2f} EUR/MWh")
    print(f"Saved: {output_file}")

    plt.show()
    plt.close()

print("\nAll full-period graphs completed!")
