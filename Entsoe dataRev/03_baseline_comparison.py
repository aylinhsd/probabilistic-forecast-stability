
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent

# Existing LEAR prediction files
ORIGINAL = BASE / "ENTSOE_LEAR_style_predictions.csv"
IMPROVED = BASE / "ENTSOE_improved_LEAR_predictions.csv"

REPORT = BASE / "03_baseline_comparison.xlsx"
PLOT = BASE / "03_baseline_yearly_MAE.png"

# -------------------------------------------------
# 1. LOAD THE TWO MODELS
# -------------------------------------------------

original = pd.read_csv(ORIGINAL)
improved = pd.read_csv(IMPROVED)

required_original = [
    "Target_UTC",
    "Target_Price_EUR_MWh",
    "Split",
    "LEAR_Style_Prediction_EUR_MWh"
]

required_improved = [
    "Target_UTC",
    "Target_Price_EUR_MWh",
    "Split",
    "Improved_LEAR_Prediction_EUR_MWh"
]

for name, data, required in [
    ("Original", original, required_original),
    ("Improved", improved, required_improved)
]:
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(
            f"{name} file missing columns: {missing}"
        )

for data in [original, improved]:
    data["Target_UTC"] = pd.to_datetime(
        data["Target_UTC"], utc=True
    )

    if data["Target_UTC"].duplicated().any():
        raise ValueError("Duplicate timestamps found")

# -------------------------------------------------
# 2. MERGE ON THE SAME TARGET HOURS
# -------------------------------------------------

df = original[
    required_original
].merge(
    improved[required_improved],
    on="Target_UTC",
    how="inner",
    suffixes=("_Original", "_Improved"),
    validate="one_to_one"
)

if len(df) != len(original) or len(df) != len(improved):
    raise ValueError(
        "The models do not cover identical target hours."
    )

if not np.allclose(
    df["Target_Price_EUR_MWh_Original"],
    df["Target_Price_EUR_MWh_Improved"]
):
    raise ValueError("Actual prices differ between files")

if not (
    df["Split_Original"] == df["Split_Improved"]
).all():
    raise ValueError("Split labels differ between files")

df["Actual"] = df["Target_Price_EUR_MWh_Original"]
df["Split"] = df["Split_Original"]

df["Original_LEAR"] = (
    df["LEAR_Style_Prediction_EUR_MWh"]
)

df["Improved_LEAR"] = (
    df["Improved_LEAR_Prediction_EUR_MWh"]
)

df["Year"] = df["Target_UTC"].dt.year

# -------------------------------------------------
# 3. EVALUATION METRICS
# -------------------------------------------------

def calculate_metrics(data, prediction_column):
    actual = data["Actual"].to_numpy()
    predicted = data[prediction_column].to_numpy()

    error = actual - predicted

    return {
        "Hours": len(data),
        "MAE": np.mean(np.abs(error)),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "Bias_Actual_Minus_Forecast": np.mean(error)
    }

period_results = []
yearly_results = []

models = ["Original_LEAR", "Improved_LEAR"]

for split, group in df.groupby("Split"):
    for model in models:
        result = calculate_metrics(group, model)
        period_results.append({
            "Split": split,
            "Model": model,
            **result
        })

for year, group in df.groupby("Year"):
    for model in models:
        result = calculate_metrics(group, model)
        yearly_results.append({
            "Year": year,
            "Model": model,
            **result
        })

period_table = pd.DataFrame(period_results)
yearly_table = pd.DataFrame(yearly_results)

print("\nPERIOD METRICS")
print(period_table.round(2).to_string(index=False))

print("\nYEARLY METRICS")
print(yearly_table.round(2).to_string(index=False))

# -------------------------------------------------
# 4. PLOT YEARLY MAE
# -------------------------------------------------

pivot = yearly_table.pivot(
    index="Year",
    columns="Model",
    values="MAE"
)

ax = pivot.plot(
    kind="bar",
    figsize=(13, 5),
    width=0.8
)

ax.set_title("Italy North - Yearly LEAR Forecast MAE")
ax.set_xlabel("Year")
ax.set_ylabel("MAE (EUR/MWh)")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT, dpi=200)
plt.close()


# -------------------------------------------------
# 5. SAVE EXCEL REPORT
# -------------------------------------------------

# Create a copy for Excel export.
# Remove timezone information only from this copy.
# The timestamps remain in UTC.

excel_predictions = df[[
    "Target_UTC",
    "Split",
    "Actual",
    "Original_LEAR",
    "Improved_LEAR"
]].copy()

excel_predictions["Target_UTC"] = (
    excel_predictions["Target_UTC"]
    .dt.tz_localize(None)
)

with pd.ExcelWriter(REPORT, engine="openpyxl") as writer:

    period_table.to_excel(
        writer,
        sheet_name="Period Metrics",
        index=False
    )

    yearly_table.to_excel(
        writer,
        sheet_name="Yearly Metrics",
        index=False
    )

    excel_predictions.to_excel(
        writer,
        sheet_name="Aligned Predictions",
        index=False
    )

print("\nSaved:")
print(REPORT)
print(PLOT)
print("\nBASELINE COMPARISON COMPLETE")
