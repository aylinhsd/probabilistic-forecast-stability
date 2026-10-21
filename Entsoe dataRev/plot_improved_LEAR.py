
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# PART 1 — FILE LOCATIONS
# ==========================================

folder = Path(__file__).resolve().parent

input_file = folder / "ENTSOE_improved_LEAR_predictions.csv"

# ==========================================
# PART 2 — LOAD DATA
# ==========================================

df = pd.read_csv(input_file)

df["Target_UTC"] = pd.to_datetime(
    df["Target_UTC"], utc=True
)

df["Date_Italy"] = (
    df["Target_UTC"]
    .dt.tz_convert("Europe/Rome")
    .dt.date
)

actual_col = "Target_Price_EUR_MWh"
pred_col = "Improved_LEAR_Prediction_EUR_MWh"

# ==========================================
# PART 3 — DEFINE PERIODS
# ==========================================

periods = [
    ("Validation", 2021),
    ("Test", 2022),
    ("Robustness", 2024)
]

# ==========================================
# PART 4 — WEEKLY GRAPHS
# ==========================================

for split, year in periods:

    start = pd.Timestamp(
        f"{year}-08-01",
        tz="Europe/Rome"
    ).tz_convert("UTC")

    end = pd.Timestamp(
        f"{year}-08-08",
        tz="Europe/Rome"
    ).tz_convert("UTC")

    week = df[
        (df["Split"] == split) &
        (df["Target_UTC"] >= start) &
        (df["Target_UTC"] < end)
    ].copy()

    if week.empty:
        print(f"No weekly data for {year}")
        continue

    week = week.sort_values("Target_UTC")

    actual = week[actual_col]
    predicted = week[pred_col]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    local_time = week["Target_UTC"].dt.tz_convert(
        "Europe/Rome"
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        local_time,
        actual,
        label="Actual electricity price",
        linewidth=2
    )

    plt.plot(
        local_time,
        predicted,
        label="Improved LEAR prediction",
        linewidth=2,
        linestyle="--"
    )

    plt.title(
        f"Improved LEAR vs Actual — {year} ({split})\n"
        f"MAE = {mae:.2f} | RMSE = {rmse:.2f} EUR/MWh"
    )

    plt.xlabel("Italian local time")
    plt.ylabel("Electricity price (EUR/MWh)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    output = folder / f"Improved_LEAR_Weekly_{year}.png"

    plt.savefig(output, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved: {output}")

# ==========================================
# PART 5 — FULL-PERIOD GRAPHS
# ==========================================

for split in ["Validation", "Test", "Robustness"]:

    subset = df[df["Split"] == split].copy()

    if subset.empty:
        continue

    actual = subset[actual_col]
    predicted = subset[pred_col]

    # Metrics are calculated using ALL hourly prices.
    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    # Daily averages are used ONLY for visualization.
    daily = (
        subset.groupby("Date_Italy")
        .agg(
            Actual=(actual_col, "mean"),
            Predicted=(pred_col, "mean")
        )
        .reset_index()
    )

    daily["Date_Italy"] = pd.to_datetime(
        daily["Date_Italy"]
    )

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
        label="Improved LEAR daily average prediction",
        linewidth=1.5,
        linestyle="--"
    )

    plt.title(
        f"Improved LEAR vs Actual — Full {split} Period\n"
        f"Hourly MAE = {mae:.2f} | "
        f"Hourly RMSE = {rmse:.2f} EUR/MWh"
    )

    plt.xlabel("Italian delivery date")
    plt.ylabel("Daily average electricity price (EUR/MWh)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    output = folder / f"Improved_LEAR_Full_{split}.png"

    plt.savefig(output, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved: {output}")

print("\nAll improved LEAR graphs completed!")
