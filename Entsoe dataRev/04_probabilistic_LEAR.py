
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# 1. FILE SETTINGS
# ==================================================

BASE = Path(__file__).resolve().parent

INPUT = BASE / "ENTSOE_improved_LEAR_predictions.csv"

OUTPUT_CSV = BASE / "04_probabilistic_LEAR_predictions.csv"
OUTPUT_XLSX = BASE / "04_probabilistic_LEAR_report.xlsx"
OUTPUT_PLOT = BASE / "04_probabilistic_LEAR_2022.png"

PRED_COL = "Improved_LEAR_Prediction_EUR_MWh"
ACTUAL_COL = "Target_Price_EUR_MWh"

QUANTILES = np.round(
    np.arange(0.05, 1.00, 0.05), 2
)

# ==================================================
# 2. LOAD AND CHECK DATA
# ==================================================

print("\nSTEP 1: LOADING LEAR PREDICTIONS")

df = pd.read_csv(INPUT)

required = [
    "Target_UTC", "Split", ACTUAL_COL, PRED_COL
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["Target_UTC"] = pd.to_datetime(
    df["Target_UTC"], utc=True
)

df["Target_Date_Italy"] = (
    df["Target_UTC"]
    .dt.tz_convert("Europe/Rome")
    .dt.date
)

df = df.sort_values("Target_UTC").copy()

if df["Target_UTC"].duplicated().any():
    raise ValueError("Duplicate forecast timestamps")

if df[[ACTUAL_COL, PRED_COL]].isna().any().any():
    raise ValueError("Missing actual or predicted prices")

# ==================================================
# 3. CALIBRATION ERRORS FROM 2021
# ==================================================

print("\nSTEP 2: CALIBRATION")

calibration = df[
    df["Split"] == "Validation"
].copy()

evaluation = df[
    df["Split"].isin(["Test", "Robustness"])
].copy()

if calibration.empty or evaluation.empty:
    raise ValueError("Calibration or evaluation data missing")

calibration["Residual"] = (
    calibration[ACTUAL_COL] - calibration[PRED_COL]
)

print("Calibration observations:", len(calibration))
print("Evaluation observations:", len(evaluation))

# Estimate quantiles of 2021 forecast residuals
residual_quantiles = {
    float(q): float(
        calibration["Residual"].quantile(q)
    )
    for q in QUANTILES
}

print("\nResidual quantiles:")
for q, value in residual_quantiles.items():
    print(f"q{int(q * 100):02d}: {value:.2f} EUR/MWh")

# ==================================================
# 4. GENERATE PROBABILISTIC FORECASTS
# ==================================================

print("\nSTEP 3: GENERATING QUANTILE FORECASTS")

quantile_columns = {}

for q in QUANTILES:

    col = f"Q{int(q * 100):02d}"

    evaluation[col] = (
        evaluation[PRED_COL] + residual_quantiles[float(q)]
    )

    quantile_columns[float(q)] = col

# 80% prediction interval: q10 to q90
evaluation["Lower_80"] = evaluation["Q10"]
evaluation["Upper_80"] = evaluation["Q90"]

evaluation["Covered_80"] = (
    (evaluation[ACTUAL_COL] >= evaluation["Lower_80"])
    &
    (evaluation[ACTUAL_COL] <= evaluation["Upper_80"])
)

evaluation["Interval_Width_80"] = (
    evaluation["Upper_80"] - evaluation["Lower_80"]
)

# ==================================================
# 5. EVALUATION METRICS
# ==================================================

print("\nSTEP 4: PROBABILISTIC EVALUATION")

def pinball_loss(actual, predicted, q):
    error = actual - predicted
    return np.maximum(
        q * error,
        (q - 1) * error
    )

def evaluate(group):

    actual = group[ACTUAL_COL].to_numpy()

    coverage = group["Covered_80"].mean()

    width = group["Interval_Width_80"].mean()

    # Pinball loss at each quantile
    losses = []

    for q in QUANTILES:
        predicted = group[
            quantile_columns[float(q)]
        ].to_numpy()

        losses.append(
            pinball_loss(actual, predicted, q).mean()
        )

    # Numerical integration over q05 to q95.
    # This is a partial CRPS integral, NOT full CRPS.
    partial_crps = 2 * np.trapezoid(
        losses, QUANTILES
    )

    return {
        "Hours": len(group),
        "Coverage_80_Percent": coverage * 100,
        "Mean_Interval_Width": width,
        "Mean_Pinball_Loss": np.mean(losses),
        "Partial_CRPS_05_95": partial_crps
    }

results = []

for split, group in evaluation.groupby("Split"):

    results.append({
        "Period": split,
        **evaluate(group)
    })

evaluation["Year"] = pd.to_datetime(
    evaluation["Target_Date_Italy"]
).dt.year

for year, group in evaluation.groupby("Year"):

    results.append({
        "Period": str(year),
        **evaluate(group)
    })

metrics = pd.DataFrame(results)

print("\nPROBABILISTIC METRICS")
print(metrics.round(3).to_string(index=False))

# ==================================================
# 6. PLOT AN EXAMPLE WEEK IN 2022
# ==================================================

print("\nSTEP 5: PLOTTING")

week = evaluation[
    (evaluation["Target_Date_Italy"] >=
     pd.to_datetime("2022-08-01").date())
    &
    (evaluation["Target_Date_Italy"] <=
     pd.to_datetime("2022-08-07").date())
].copy()

if not week.empty:

    x = week["Target_UTC"].dt.tz_convert(
        "Europe/Rome"
    )

    plt.figure(figsize=(15, 6))

    plt.plot(
        x, week[ACTUAL_COL],
        label="Actual price",
        linewidth=1.5
    )

    plt.plot(
        x, week[PRED_COL],
        label="Improved LEAR point forecast",
        linewidth=1.3
    )

    plt.fill_between(
        x,
        week["Lower_80"],
        week["Upper_80"],
        alpha=0.25,
        label="80% prediction interval"
    )

    plt.title(
        "Probabilistic LEAR - August 1-7, 2022"
    )

    plt.xlabel("Italian delivery time")
    plt.ylabel("Electricity price (EUR/MWh)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200)
    plt.close()

# ==================================================
# 7. SAVE RESULTS
# ==================================================

print("\nSTEP 6: SAVING RESULTS")

evaluation["Target_UTC"] = (
    evaluation["Target_UTC"]
    .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
)

evaluation.to_csv(OUTPUT_CSV, index=False)

quantile_table = pd.DataFrame([
    {
        "Quantile": q,
        "Residual_Offset_EUR_MWh": offset
    }
    for q, offset in residual_quantiles.items()
])

with pd.ExcelWriter(
    OUTPUT_XLSX, engine="openpyxl"
) as writer:

    metrics.to_excel(
        writer, sheet_name="Evaluation",
        index=False
    )

    quantile_table.to_excel(
        writer, sheet_name="Residual Quantiles",
        index=False
    )

print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_XLSX)
print(OUTPUT_PLOT)

print("\nPROBABILISTIC LEAR COMPLETE")
