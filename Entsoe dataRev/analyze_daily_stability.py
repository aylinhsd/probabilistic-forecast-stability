#!/usr/bin/env python3
"""Daily and 7/30-day rolling evaluation of point-model residual-quantile baselines.

Run: python analyze_daily_stability.py
The script does not retrain any model and never modifies source CSVs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "daily_stability_results"
FILES = {
    "LEAR_v3": "ENTSOE_LEAR_v3_predictions.csv",
    "Previous_LEAR": "ENTSOE_improved_LEAR_predictions.csv",
    "LEAR_style": "ENTSOE_LEAR_style_predictions.csv",
    "Seasonal_naive": "ENTSOE_seasonal_naive_predictions.csv",
}
QUANTILES = np.array(sorted(set(np.round(np.arange(.01, 1, .01), 3)) | {.075, .925}))
PRED_NAMES = {
    "LEAR_v3": ["LEAR_v3_Prediction_EUR_MWh"],
    "Previous_LEAR": ["Improved_LEAR_Prediction_EUR_MWh"],
    "LEAR_style": ["LEAR_Style_Prediction_EUR_MWh"],
    "Seasonal_naive": ["Seasonal_Naive_Prediction_EUR_MWh",
                       "Predicted_Price_EUR_MWh"],
}

def pick(df, names, role):
    for name in names:
        if name in df:
            return name
    raise ValueError(f"Cannot identify {role}; columns: {list(df.columns)}")

def load(name, filename):
    df = pd.read_csv(ROOT / filename, low_memory=False)
    timestamp = pd.to_datetime(df[pick(df, ["Target_UTC"], "UTC time")], utc=True)
    if timestamp.isna().any() or timestamp.duplicated().any():
        raise ValueError(f"{name}: duplicate or missing UTC timestamps")
    y = pd.to_numeric(df[pick(df, ["Target_Price_EUR_MWh"], "actual")]).to_numpy(float)
    try:
        pc = pick(df, PRED_NAMES[name], "prediction")
    except ValueError:
        options = [c for c in df if "pred" in c.lower() and "error" not in c.lower()
                   and "quantile" not in c.lower()]
        if len(options) != 1:
            raise ValueError(f"{name}: prediction column ambiguous: {options}")
        pc = options[0]
    p = pd.to_numeric(df[pc]).to_numpy(float)
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError(f"{name}: nonfinite prices")
    # Common chronological calibration/evaluation dates across ALL models.
    # Legacy files call 2025 "Robustness", while LEAR v3 calls it Validation/Test.
    local_date = timestamp.dt.tz_convert("Europe/Rome").dt.date
    validation = ((local_date >= pd.Timestamp("2025-01-01").date()) &
                  (local_date <= pd.Timestamp("2025-06-30").date())).to_numpy()
    test = (local_date >= pd.Timestamp("2025-07-01").date()).to_numpy()
    if not validation.any() or not test.any():
        raise ValueError(f"{name}: missing validation or test data")
    offsets = np.quantile(y[validation] - p[validation], QUANTILES)
    # Fixed validation residual shifts: the same probabilistic baseline as earlier.
    qhat = p[test, None] + offsets
    result = pd.DataFrame({"Model": name, "UTC": timestamp[test].astype(str).to_numpy(),
        "Date": timestamp[test].dt.tz_convert("Europe/Rome").dt.date.astype(str).to_numpy(),
        "Actual": y[test], "Point": p[test]})
    result = pd.concat([result, pd.DataFrame(qhat, columns=[
        f"q{q:.3f}" for q in QUANTILES], index=result.index)], axis=1)
    print(f"{name}: validation={validation.sum()}, test={test.sum()}, prediction={pc}", flush=True)
    return result

QCOLS = [f"q{q:.3f}" for q in QUANTILES]
def metrics(frame):
    y, p = frame.Actual.to_numpy(), frame.Point.to_numpy()
    q = frame[QCOLS].to_numpy()
    d = y[:, None] - q
    loss = np.maximum(QUANTILES * d, (QUANTILES - 1) * d)
    # Approximate CRPS: tails beyond Q01 and Q99 omitted.
    crps = 2 * np.trapezoid(loss, QUANTILES, axis=1)
    m = {"N_Hours": len(frame), "MAE": np.mean(np.abs(y-p)),
         "RMSE": np.sqrt(np.mean((y-p)**2)),
         "Mean_Pinball": loss.mean(), "Approx_CRPS": crps.mean()}
    for coverage, low, high in [(80,.1,.9),(85,.075,.925),(90,.05,.95)]:
        a = q[:, np.where(np.isclose(QUANTILES, low))[0][0]]
        b = q[:, np.where(np.isclose(QUANTILES, high))[0][0]]
        m[f"PI{coverage}_Coverage_Pct"] = 100 * np.mean((y >= a) & (y <= b))
        m[f"PI{coverage}_Width"] = np.mean(b-a)
        m[f"PI{coverage}_Coverage_Error_PP"] = m[f"PI{coverage}_Coverage_Pct"]-coverage
    return m

def main():
    OUT.mkdir(exist_ok=True)
    frames = []
    for name, filename in FILES.items():
        if not (ROOT / filename).exists():
            print(f"SKIPPING {name}: {filename} not found", flush=True)
            continue
        frames.append(load(name, filename))
    if not frames:
        raise FileNotFoundError(f"No prediction CSV files found in {ROOT}")
    all_data = pd.concat(frames, ignore_index=True)
    stamps = [set(g.UTC) for _, g in all_data.groupby("Model")]
    common = set.intersection(*stamps)
    if not common:
        raise ValueError("No common test timestamps between models")
    all_data = all_data[all_data.UTC.isin(common)].copy()
    actual_range = all_data.groupby("UTC").Actual.agg(["min","max"])
    if not np.allclose(actual_range["min"], actual_range["max"], atol=1e-8):
        raise ValueError("Actual prices disagree across model files")
    daily, rolling, overall = [], [], []
    for name, frame in all_data.groupby("Model"):
        overall.append({"Model": name, **metrics(frame)})
        days = sorted(frame.Date.unique())
        for day, group in frame.groupby("Date"):
            daily.append({"Model": name, "Date": day, **metrics(group)})
        for window in (7, 30):
            for end in range(window-1, len(days)):
                selected = frame[frame.Date.isin(days[end-window+1:end+1])]
                rolling.append({"Model": name, "Window_Days": window,
                                "End_Date": days[end], **metrics(selected)})
    daily, rolling, overall = map(pd.DataFrame, (daily, rolling, overall))
    daily.to_csv(OUT / "daily_metrics.csv", index=False)
    rolling.to_csv(OUT / "rolling_metrics.csv", index=False)
    overall.to_csv(OUT / "overall_metrics.csv", index=False)
    diagnostics = daily.groupby("Model").agg(
        Days=("Date", "count"),
        Daily_CRPS_Mean=("Approx_CRPS", "mean"),
        Daily_CRPS_SD=("Approx_CRPS", "std"),
        Daily_PI80_Coverage_SD_PP=("PI80_Coverage_Pct", "std"),
        Daily_PI80_Width_SD=("PI80_Width", "std")).reset_index()
    diagnostics.to_csv(OUT / "descriptive_stability.csv", index=False)
    for column in ("Approx_CRPS", "MAE", "PI80_Coverage_Pct", "PI80_Width"):
        for window in (0,7,30):
            fig, ax = plt.subplots(figsize=(12,4))
            table = daily if window == 0 else rolling[rolling.Window_Days.eq(window)]
            xcol = "Date" if window == 0 else "End_Date"
            for name, group in table.groupby("Model"):
                ax.plot(pd.to_datetime(group[xcol]), group[column],
                        label=name, linewidth=0.8 if window == 0 else 1.5)
            if column == "PI80_Coverage_Pct":
                ax.axhline(80, linestyle="--", color="black", linewidth=1)
            ax.set(title=f"{'Daily' if window == 0 else str(window)+'-day rolling'} {column}",
                   xlabel="Italy-local date", ylabel=column)
            ax.legend(fontsize=8)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(OUT / f"{'daily' if window == 0 else str(window)+'day'}_{column}.png", dpi=150)
            plt.close(fig)
    (OUT / "methodology.json").write_text(json.dumps({
        "models": list(all_data.Model.unique()), "common_test_hours": len(common),
        "quantiles": QUANTILES.tolist(),
        "probabilistic_baseline": "point prediction plus pooled validation residual quantile",
        "crps": "2 * trapezoidal integral of pinball loss Q01-Q99 (tails omitted)",
        "rolling": "pooled hourly observations in trailing 7 or 30 Italy-local days",
        "warning": "Daily coverage based on ~24 observations has high sampling variability; diagnostics are not a stability index."
    }, indent=2) + "\n")
    print("\n", overall[["Model","N_Hours","MAE","Approx_CRPS",
                           "PI80_Coverage_Pct","PI80_Width"]].to_string(index=False))
    print(f"\nSaved results to {OUT}", flush=True)

if __name__ == "__main__":
    main()
