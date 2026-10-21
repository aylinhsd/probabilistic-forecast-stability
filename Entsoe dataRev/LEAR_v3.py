"""LEAR v3: fixed 24-hour-specific LASSO baseline for Italy North day-ahead prices.

Train: through 2024-12-31; validation: 2025-01-01 to 2025-06-30;
test: from 2025-07-01. No rolling retraining in this version.

Requires: pandas, numpy, scikit-learn, openpyxl.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# PART 1 — FILES AND SETTINGS
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "ENTSOE_ML_dataset_updated.csv"
PREDICTIONS_FILE = ROOT / "ENTSOE_LEAR_v3_predictions.csv"
REPORT_FILE = ROOT / "ENTSOE_LEAR_v3_report.xlsx"
ALPHAS = [0.03, 0.1, 0.3, 1.0]  # Same candidates as the previous script

# PART 2 — LOAD AND VALIDATE DATA
print("Loading:", SOURCE, flush=True)
df = pd.read_csv(SOURCE, low_memory=False)
lag_cols = [f"Price_UTC_Lag_{i}h" for i in range(1, 169)]
stat_cols = [
    f"{stat}_D-{day}"
    for day in range(1, 8)
    for stat in ("mean", "minimum", "maximum", "std")
]
required = [
    "Target_Date_Italy", "Target_UTC", "Target_Price_EUR_MWh",
    "Target_Hour_Italy", "Day_Of_Week", "Month", "Split",
] + lag_cols + stat_cols
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing dataset columns: {missing}")
if df[required].isna().any().any():
    raise ValueError("Missing values in required columns")

df["Target_UTC"] = pd.to_datetime(df["Target_UTC"], utc=True)
df["Target_Date_Italy"] = pd.to_datetime(df["Target_Date_Italy"])
if df["Target_UTC"].duplicated().any():
    raise ValueError("Duplicate Target_UTC timestamps")
if not df["Target_Hour_Italy"].between(0, 23).all():
    raise ValueError("Invalid Italian clock hours")

# Check the actual dates, not only the CSV's split labels.
dates = df["Target_Date_Italy"]
expected = np.where(
    dates <= pd.Timestamp("2024-12-31"), "Train",
    np.where(dates <= pd.Timestamp("2025-06-30"), "Validation", "Test"),
)
if not np.array_equal(df["Split"].to_numpy(), expected):
    raise ValueError("Split labels do not match the agreed date boundaries")

# PART 3 — FEATURES (same structure as improved_LEAR.py)
calendar = pd.get_dummies(
    df[["Day_Of_Week", "Month"]].astype(int).astype(str),
    prefix=["weekday", "month"], dtype=np.float64,
)
feature_names = lag_cols + stat_cols + calendar.columns.tolist()
X = np.column_stack([
    df[lag_cols + stat_cols].to_numpy(dtype=np.float64),
    calendar.to_numpy(dtype=np.float64),
])
y = df["Target_Price_EUR_MWh"].to_numpy(dtype=np.float64)
hours = df["Target_Hour_Italy"].to_numpy(dtype=int)
splits = df["Split"].to_numpy()
train = splits == "Train"
valid = splits == "Validation"
test = splits == "Test"
if not (train.any() and valid.any() and test.any()):
    raise ValueError("Train, Validation and Test must all contain rows")
if not np.isfinite(X).all() or not np.isfinite(y).all():
    raise ValueError("Non-finite feature or target values")
print("Rows:", len(df), "| Features:", X.shape[1])
for name, mask in (("Train", train), ("Validation", valid), ("Test", test)):
    print(f"{name}: {mask.sum()} rows")

# PART 4 — CHOOSE ALPHA ON VALIDATION ONLY
alpha_results = []
best_alpha = None
best_mae = float("inf")
for alpha in ALPHAS:
    validation_predictions = np.full(len(df), np.nan)
    for hour in range(24):
        train_mask = train & (hours == hour)
        valid_mask = valid & (hours == hour)
        if not train_mask.any() or not valid_mask.any():
            raise ValueError(f"Missing train/validation observations for hour {hour}")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])  # Fit scaler on train only
        model = Lasso(
            alpha=alpha,
            max_iter=100000,
            tol=0.0001,
            selection="cyclic"
        )
        model.fit(X_train, y[train_mask])
        validation_predictions[valid_mask] = model.predict(scaler.transform(X[valid_mask]))
    mae = mean_absolute_error(y[valid], validation_predictions[valid])
    rmse = np.sqrt(mean_squared_error(y[valid], validation_predictions[valid]))
    alpha_results.append({"Alpha": alpha, "Validation_MAE": mae, "Validation_RMSE": rmse})
    print(f"Alpha {alpha}: validation MAE={mae:.3f}, RMSE={rmse:.3f}", flush=True)
    if mae < best_mae:
        best_mae = mae
        best_alpha = alpha
print("Selected alpha:", best_alpha, flush=True)

# PART 5 — FINAL 24 MODELS: FIT ON TRAIN ONLY
predictions = np.full(len(df), np.nan)
coefficient_rows = []
for hour in range(24):
    train_mask = train & (hours == hour)
    hour_mask = hours == hour
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    model = Lasso(
        alpha=best_alpha,
        max_iter=100000,
        tol=0.0001,
        selection="cyclic"
    )
    model.fit(X_train, y[train_mask])
    predictions[hour_mask] = model.predict(scaler.transform(X[hour_mask]))
    coefficient_rows.extend(
        {"Hour_Italy": hour, "Feature": name, "Coefficient": float(coef)}
        for name, coef in zip(feature_names, model.coef_)
    )
if not np.isfinite(predictions).all():
    raise ValueError("Some predictions are missing")

# PART 6 — SAVE PREDICTIONS
output_cols = [
    "Target_Date_Italy", "Target_UTC", "Target_Hour_Italy",
    "Target_Price_EUR_MWh", "Split",
]
if "Issue_Time_UTC" in df.columns:
    output_cols.insert(2, "Issue_Time_UTC")
output = df[output_cols].copy()
output["LEAR_v3_Prediction_EUR_MWh"] = predictions
output["Error_EUR_MWh"] = y - predictions  # Bias: actual minus predicted
output["Absolute_Error_EUR_MWh"] = np.abs(output["Error_EUR_MWh"])
output.to_csv(PREDICTIONS_FILE, index=False)

# PART 7 — METRICS
metrics = []
for name, mask in (("Train", train), ("Validation", valid), ("Test", test)):
    error = y[mask] - predictions[mask]
    metrics.append({
        "Split": name,
        "Hours": int(mask.sum()),
        "MAE_EUR_MWh": mean_absolute_error(y[mask], predictions[mask]),
        "RMSE_EUR_MWh": np.sqrt(mean_squared_error(y[mask], predictions[mask])),
        "Bias_Actual_Minus_Predicted_EUR_MWh": float(np.mean(error)),
    })
metrics_df = pd.DataFrame(metrics)
print("\nLEAR v3 RESULTS")
print(metrics_df.to_string(index=False))
print("Note: Train metrics are in-sample, not out-of-sample.")

# PART 8 — YEARLY RESULTS AND EXCEL REPORT
output["Year"] = output["Target_Date_Italy"].dt.year
yearly = output.groupby(["Split", "Year"], sort=True).agg(
    Hours=("Error_EUR_MWh", "size"),
    MAE_EUR_MWh=("Absolute_Error_EUR_MWh", "mean"),
    RMSE_EUR_MWh=("Error_EUR_MWh", lambda z: float(np.sqrt(np.mean(np.square(z))))),
    Bias_Actual_Minus_Predicted_EUR_MWh=("Error_EUR_MWh", "mean"),
).reset_index()
methodology = pd.DataFrame([
    ("Model", "24 hour-specific LASSO models (LEAR-style fixed baseline)"),
    ("Training", "2015-01-12 to 2024-12-31"),
    ("Validation", "2025-01-01 to 2025-06-30"),
    ("Test", "2025-07-01 to latest available date"),
    ("Daily historical statistics", "D-1 through D-7"),
    ("Hourly price lags", "1 through 168 hours"),
    ("Selected alpha", str(best_alpha)),
    ("Bias convention", "Actual minus predicted; positive = underforecast"),
    ("Scaling", "Fit on training only, separately for each hour"),
    ("Rolling retraining", "Not implemented"),
    ("Train metrics", "In-sample; do not interpret as independent test"),
], columns=["Setting", "Value"])
with pd.ExcelWriter(REPORT_FILE, engine="openpyxl") as writer:
    methodology.to_excel(writer, sheet_name="Methodology", index=False)
    metrics_df.to_excel(writer, sheet_name="Period Metrics", index=False)
    yearly.to_excel(writer, sheet_name="Yearly Metrics", index=False)
    pd.DataFrame(alpha_results).to_excel(writer, sheet_name="Alpha Selection", index=False)
    pd.DataFrame(coefficient_rows).to_excel(writer, sheet_name="Coefficients", index=False)
print("\nSaved:", PREDICTIONS_FILE)
print("Saved:", REPORT_FILE)
print("DONE!")
