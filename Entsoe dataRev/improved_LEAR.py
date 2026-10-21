
"""
Improved preliminary LEAR-style forecasting model.

Changes:
1. Separate LASSO model for each Italian clock hour.
2. Alpha selection using 2021 validation data only.
3. Compare against the original LEAR predictions.

Requires:
pandas, numpy, scikit-learn, openpyxl, matplotlib
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================
# PART 1 — FILES AND SETTINGS
# ============================================

ROOT = Path(__file__).resolve().parent

SOURCE = ROOT / "ENTSOE_ML_dataset.csv"

OLD_PREDICTIONS = ROOT / "ENTSOE_LEAR_style_predictions.csv"

NEW_PREDICTIONS = ROOT / "ENTSOE_improved_LEAR_predictions.csv"

REPORT = ROOT / "ENTSOE_improved_LEAR_report.xlsx"

ALPHAS = [0.03, 0.1, 0.3, 1.0]

# ============================================
# PART 2 — LOAD DATA
# ============================================

print("Loading ML dataset...")

df = pd.read_csv(SOURCE, low_memory=False)

lag_cols = [
    f"Price_UTC_Lag_{i}h"
    for i in range(1, 169)
]

stat_cols = [
    f"{s}_D-{i}"
    for i in range(2, 9)
    for s in ["mean", "minimum", "maximum", "std"]
]

required = [
    "Target_Date_Italy",
    "Target_UTC",
    "Target_Price_EUR_MWh",
    "Target_Hour_Italy",
    "Day_Of_Week",
    "Month",
    "Split",
] + lag_cols + stat_cols

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df[required].isna().any().any():
    raise ValueError("Missing values in required columns")

df["Target_UTC"] = pd.to_datetime(
    df["Target_UTC"], utc=True
)

if df["Target_UTC"].duplicated().any():
    raise ValueError("Duplicate target timestamps")

df["Target_Hour_Italy"] = pd.to_numeric(
    df["Target_Hour_Italy"]
).astype(int)

if not df["Target_Hour_Italy"].between(0, 23).all():
    raise ValueError("Unexpected Italian clock hour")

# ============================================
# PART 3 — PREPARE FEATURES
# ============================================

# The target hour is handled by separate models.
# Weekday and month remain input features.

calendar = pd.get_dummies(
    df[["Day_Of_Week", "Month"]]
    .astype(int)
    .astype(str),
    prefix=["weekday", "month"],
    dtype=np.float64
)

feature_names = (
    lag_cols + stat_cols + calendar.columns.tolist()
)

X = np.column_stack([
    df[lag_cols + stat_cols].to_numpy(
        dtype=np.float64
    ),
    calendar.to_numpy(dtype=np.float64)
])

y = df["Target_Price_EUR_MWh"].to_numpy(
    dtype=np.float64
)

splits = df["Split"].astype(str).to_numpy()

hours = df["Target_Hour_Italy"].to_numpy()

if not np.isfinite(X).all() or not np.isfinite(y).all():
    raise ValueError("Invalid numerical values")

train = splits == "Train"
valid = splits == "Validation"

print("Total observations:", len(df))
print("Features per hourly model:", X.shape[1])

# ============================================
# PART 4 — TRAIN 24 HOURLY MODELS
# ============================================

# For each alpha, train a separate model
# for every Italian clock hour.

alpha_results = []
best_alpha = None
best_mae = np.inf

for alpha in ALPHAS:

    validation_predictions = np.full(
        len(df), np.nan
    )

    for hour in range(24):

        train_mask = train & (hours == hour)
        valid_mask = valid & (hours == hour)

        if not train_mask.any() or not valid_mask.any():
            raise ValueError(
                f"Missing training/validation rows for hour {hour}"
            )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X[train_mask]
        )

        X_valid = scaler.transform(
            X[valid_mask]
        )

        model = Lasso(
            alpha=alpha,
            max_iter=10000,
            tol=0.0001,
            selection="cyclic"
        )

        model.fit(
            X_train,
            y[train_mask]
        )

        validation_predictions[valid_mask] = (
            model.predict(X_valid)
        )

    # Calculate MAE across ALL validation hours.

    mae = mean_absolute_error(
        y[valid],
        validation_predictions[valid]
    )

    rmse = np.sqrt(
        mean_squared_error(
            y[valid],
            validation_predictions[valid]
        )
    )

    alpha_results.append({
        "Alpha": alpha,
        "Validation_MAE": mae,
        "Validation_RMSE": rmse
    })

    print(
        f"Alpha {alpha}: "
        f"Validation MAE = {mae:.3f}, "
        f"RMSE = {rmse:.3f}",
        flush=True
    )

    if mae < best_mae:
        best_mae = mae
        best_alpha = alpha

print("\nSelected alpha:", best_alpha)

# ============================================
# PART 5 — FIT SELECTED MODELS
# ============================================

# Train again using the selected alpha.
# Training still uses only 2015–2020.

predictions = np.full(len(df), np.nan)

coefficient_rows = []

for hour in range(24):

    train_mask = train & (hours == hour)
    hour_mask = hours == hour

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X[train_mask]
    )

    model = Lasso(
        alpha=best_alpha,
        max_iter=10000,
        tol=0.0001,
        selection="cyclic"
    )

    model.fit(
        X_train,
        y[train_mask]
    )

    predictions[hour_mask] = model.predict(
        scaler.transform(X[hour_mask])
    )

    for name, coefficient in zip(
        feature_names, model.coef_
    ):
        coefficient_rows.append({
            "Hour_Italy": hour,
            "Feature": name,
            "Coefficient": coefficient
        })

if not np.isfinite(predictions).all():
    raise ValueError("Some predictions are missing")

# ============================================
# PART 6 — SAVE PREDICTIONS
# ============================================

output = df[[
    "Target_Date_Italy",
    "Target_UTC",
    "Target_Hour_Italy",
    "Target_Price_EUR_MWh",
    "Split"
]].copy()

output["Improved_LEAR_Prediction_EUR_MWh"] = predictions

output["Error_EUR_MWh"] = (
    output["Target_Price_EUR_MWh"] - predictions
)

output["Absolute_Error_EUR_MWh"] = (
    output["Error_EUR_MWh"].abs()
)

output.to_csv(
    NEW_PREDICTIONS,
    index=False
)

# ============================================
# PART 7 — EVALUATE PERFORMANCE
# ============================================

metrics = []

for split in [
    "Train",
    "Validation",
    "Test",
    "Robustness"
]:

    mask = splits == split

    if not mask.any():
        continue

    mae = mean_absolute_error(
        y[mask], predictions[mask]
    )

    rmse = np.sqrt(
        mean_squared_error(
            y[mask], predictions[mask]
        )
    )

    metrics.append({
        "Split": split,
        "Hours": int(mask.sum()),
        "MAE_EUR_MWh": mae,
        "RMSE_EUR_MWh": rmse
    })

metrics_df = pd.DataFrame(metrics)

print("\nIMPROVED LEAR RESULTS")
print(metrics_df.to_string(index=False))

# ============================================
# PART 8 — COMPARE WITH ORIGINAL LEAR
# ============================================

comparison = pd.DataFrame()

if OLD_PREDICTIONS.exists():

    old = pd.read_csv(OLD_PREDICTIONS)

    old["Target_UTC"] = pd.to_datetime(
        old["Target_UTC"], utc=True
    )

    merged = output.merge(
        old[[
            "Target_UTC",
            "LEAR_Style_Prediction_EUR_MWh"
        ]],
        on="Target_UTC",
        how="inner",
        validate="one_to_one"
    )

    comparison_rows = []

    for split in [
        "Validation",
        "Test",
        "Robustness"
    ]:

        subset = merged[
            merged["Split"] == split
        ]

        if subset.empty:
            continue

        actual = subset["Target_Price_EUR_MWh"]

        old_pred = subset[
            "LEAR_Style_Prediction_EUR_MWh"
        ]

        new_pred = subset[
            "Improved_LEAR_Prediction_EUR_MWh"
        ]

        comparison_rows.append({
            "Split": split,
            "Hours": len(subset),
            "Original_MAE": mean_absolute_error(
                actual, old_pred
            ),
            "Improved_MAE": mean_absolute_error(
                actual, new_pred
            ),
            "Original_RMSE": np.sqrt(
                mean_squared_error(actual, old_pred)
            ),
            "Improved_RMSE": np.sqrt(
                mean_squared_error(actual, new_pred)
            )
        })

    comparison = pd.DataFrame(comparison_rows)

    print("\nORIGINAL VS IMPROVED")
    print(comparison.to_string(index=False))

# ============================================
# PART 9 — SAVE EXCEL REPORT
# ============================================

output["Year"] = pd.to_datetime(
    output["Target_Date_Italy"]
).dt.year

yearly = output.groupby("Year").agg(
    Hours=("Error_EUR_MWh", "size"),
    MAE_EUR_MWh=("Absolute_Error_EUR_MWh", "mean"),
    RMSE_EUR_MWh=(
        "Error_EUR_MWh",
        lambda z: np.sqrt(np.mean(np.square(z)))
    )
).reset_index()

coefficients = pd.DataFrame(coefficient_rows)

methodology = pd.DataFrame({
    "Setting": [
        "Model",
        "Training",
        "Validation",
        "Test",
        "Robustness",
        "Hour-specific models",
        "Selected alpha",
        "Rolling re-estimation"
    ],
    "Value": [
        "Preliminary hour-specific LASSO / LEAR-style",
        "2015–2020",
        "2021",
        "2022–2023",
        "2024–2026",
        "24",
        str(best_alpha),
        "Not yet implemented"
    ]
})

with pd.ExcelWriter(
    REPORT,
    engine="openpyxl"
) as writer:

    methodology.to_excel(
        writer,
        sheet_name="Methodology",
        index=False
    )

    metrics_df.to_excel(
        writer,
        sheet_name="Period Metrics",
        index=False
    )

    yearly.to_excel(
        writer,
        sheet_name="Yearly Metrics",
        index=False
    )

    pd.DataFrame(alpha_results).to_excel(
        writer,
        sheet_name="Alpha Selection",
        index=False
    )

    coefficients.to_excel(
        writer,
        sheet_name="Coefficients",
        index=False
    )

    if not comparison.empty:
        comparison.to_excel(
            writer,
            sheet_name="Model Comparison",
            index=False
        )

print("\nSaved:", NEW_PREDICTIONS)
print("Saved:", REPORT)
print("\nDONE!")
