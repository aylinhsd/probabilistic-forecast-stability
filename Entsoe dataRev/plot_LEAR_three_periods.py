
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# PART 1 — FILE LOCATIONS
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

# ==========================================
# PART 3 — DEFINE THE THREE PERIODS
# ==========================================

periods = [
    {
        "year": 2021,
        "split": "Validation",
        "start": "2021-08-01",
        "end": "2021-08-08"
    },
    {
        "year": 2022,
        "split": "Test",
        "start": "2022-08-01",
        "end": "2022-08-08"
    },
    {
        "year": 2024,
        "split": "Robustness",
        "start": "2024-08-01",
        "end": "2024-08-08"
    }
]

# ==========================================
# PART 4 — GENERATE THE GRAPHS
# ==========================================

for period in periods:

    year = period["year"]
    split = period["split"]

    start = pd.Timestamp(
        period["start"],
        tz="UTC"
    )

    end = pd.Timestamp(
        period["end"],
        tz="UTC"
    )

    # Select observations for this week
    week = df[
        (df["Target_UTC"] >= start) &
        (df["Target_UTC"] < end) &
        (df["Split"] == split)
    ].copy()

    if week.empty:
        print(f"No data found for {year}")
        continue

    week = week.sort_values("Target_UTC")

    # ======================================
    # PART 5 — CALCULATE WEEKLY ERRORS
    # ======================================

    actual = week["Target_Price_EUR_MWh"]

    predicted = week["LEAR_Style_Prediction_EUR_MWh"]

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    # ======================================
    # PART 6 — CREATE GRAPH
    # ======================================

    plt.figure(figsize=(14, 6))

    plt.plot(
        week["Target_UTC"],
        actual,
        label="Actual electricity price",
        linewidth=2
    )

    plt.plot(
        week["Target_UTC"],
        predicted,
        label="LEAR predicted price",
        linewidth=2,
        linestyle="--"
    )

    plt.title(
        f"LEAR Forecast vs Actual Prices — {year} ({split})\n"
        f"MAE = {mae:.2f} EUR/MWh | "
        f"RMSE = {rmse:.2f} EUR/MWh"
    )

    plt.xlabel("Time (UTC)")
    plt.ylabel("Electricity price (EUR/MWh)")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    # ======================================
    # PART 7 — SAVE GRAPH
    # ======================================

    output_file = folder / f"LEAR_{year}_{split}.png"

    plt.savefig(output_file, dpi=300)

    print(f"\n{year} — {split}")
    print(f"Hours: {len(week)}")
    print(f"MAE: {mae:.2f} EUR/MWh")
    print(f"RMSE: {rmse:.2f} EUR/MWh")
    print(f"Saved: {output_file}")

    plt.show()
    plt.close()

print("\nAll three graphs completed!")
