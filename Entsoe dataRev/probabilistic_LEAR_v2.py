"""Fixed residual-quantile prediction intervals for LEAR v3.

Calibration: Jan-Jun 2025 validation residuals; evaluation: Jul 2025 onward.
Test outcomes are never used to fit interval bounds. Requires pandas, numpy,
openpyxl. This is a simple baseline, not full predictive CDF/CRPS.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'ENTSOE_LEAR_v3_predictions.csv'
OUTPUT = ROOT / 'ENTSOE_LEAR_probabilistic_v1_predictions.csv'
REPORT = ROOT / 'ENTSOE_LEAR_probabilistic_v1_report.xlsx'
LEVELS = (0.80, 0.85, 0.90)
ACTUAL = 'Target_Price_EUR_MWh'
POINT = 'LEAR_v3_Prediction_EUR_MWh'

print('Loading:', SOURCE, flush=True)
df = pd.read_csv(SOURCE, low_memory=False)
required = ['Target_Date_Italy', 'Target_UTC', 'Split', ACTUAL, POINT]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f'Missing columns: {missing}')
if df[required].isna().any().any():
    raise ValueError('Missing values in required columns')
df['Target_Date_Italy'] = pd.to_datetime(df['Target_Date_Italy'])
df['Target_UTC'] = pd.to_datetime(df['Target_UTC'], utc=True)
if df['Target_UTC'].duplicated().any():
    raise ValueError('Duplicate Target_UTC values')
expected = np.where(df['Target_Date_Italy'] <= pd.Timestamp('2024-12-31'), 'Train',
                    np.where(df['Target_Date_Italy'] <= pd.Timestamp('2025-06-30'), 'Validation', 'Test'))
if not np.array_equal(df['Split'].to_numpy(), expected):
    raise ValueError('Split labels do not match date boundaries')
valid = df['Split'].eq('Validation')
test = df['Split'].eq('Test')
if valid.sum() < 100 or test.sum() < 100:
    raise ValueError('Insufficient validation or test rows')
residuals = (df.loc[valid, ACTUAL] - df.loc[valid, POINT]).to_numpy(dtype=float)
if not np.isfinite(residuals).all():
    raise ValueError('Nonfinite calibration residuals')

# Asymmetric residual quantiles: interval = point prediction + residual quantile.
# One pooled distribution across all 24 clock hours (baseline).
calibration = []
for level in LEVELS:
    noncoverage = 1.0 - level
    lo_q = float(np.quantile(residuals, noncoverage / 2))
    hi_q = float(np.quantile(residuals, 1 - noncoverage / 2))
    suffix = str(round(level * 100))
    df[f'Lower_{suffix}_EUR_MWh'] = df[POINT] + lo_q
    df[f'Upper_{suffix}_EUR_MWh'] = df[POINT] + hi_q
    calibration.append({'Nominal_Coverage': level, 'Lower_Residual_Quantile': lo_q,
                        'Upper_Residual_Quantile': hi_q,
                        'Calibration_Rows': int(valid.sum())})

# Winkler interval score (smaller is better), empirical coverage, average width.
def evaluate(subset, period):
    actual = subset[ACTUAL].to_numpy(dtype=float)
    results = []
    for level in LEVELS:
        suffix = str(round(level * 100))
        lo = subset[f'Lower_{suffix}_EUR_MWh'].to_numpy(dtype=float)
        hi = subset[f'Upper_{suffix}_EUR_MWh'].to_numpy(dtype=float)
        alpha = 1.0 - level
        covered = (actual >= lo) & (actual <= hi)
        width = hi - lo
        interval_score = width + (2 / alpha) * (lo - actual) * (actual < lo) + (2 / alpha) * (actual - hi) * (actual > hi)
        results.append({'Period': period, 'Nominal_Coverage': level,
                        'Rows': len(subset), 'Observed_Coverage': float(covered.mean()),
                        'Coverage_Gap': float(covered.mean() - level),
                        'Mean_Width_EUR_MWh': float(width.mean()),
                        'Mean_Interval_Score_EUR_MWh': float(interval_score.mean())})
    return results

metrics = evaluate(df.loc[valid], 'Validation') + evaluate(df.loc[test], 'Test')
metrics_df = pd.DataFrame(metrics)
monthly = []
test_df = df.loc[test].copy()
test_df['Month_UTC'] = test_df['Target_UTC'].dt.strftime('%Y-%m')
for month, part in test_df.groupby('Month_UTC', sort=True):
    monthly.extend(evaluate(part, month))
monthly_df = pd.DataFrame(monthly)

# Only report intervals for validation/test: training is not a calibrated evaluation period.
result = df.loc[valid | test].copy()
result.to_csv(OUTPUT, index=False)
methodology = pd.DataFrame([
    ('Point model', 'Fixed LEAR v3, trained through 2024'),
    ('Calibration sample', 'Validation residuals, 2025-01-01 through 2025-06-30'),
    ('Test sample', '2025-07-01 onward; never used for interval fitting'),
    ('Interval construction', 'Pooled asymmetric empirical residual quantiles'),
    ('Nominal coverage', '80%, 85%, 90%'),
    ('Coverage rule', 'Lower <= actual <= upper'),
    ('Interval score', 'Winkler interval score; lower is better'),
    ('Limitation', 'Alpha was selected on same validation period; assess on test'),
    ('Limitation', 'No hourly conditioning, time-varying calibration, or full CRPS yet'),
], columns=['Setting', 'Value'])
with pd.ExcelWriter(REPORT, engine='openpyxl') as writer:
    methodology.to_excel(writer, sheet_name='Methodology', index=False)
    pd.DataFrame(calibration).to_excel(writer, sheet_name='Calibration Quantiles', index=False)
    metrics_df.to_excel(writer, sheet_name='Coverage and Scores', index=False)
    monthly_df.to_excel(writer, sheet_name='Monthly Test Stability', index=False)
print('\nPROBABILISTIC LEAR v1 RESULTS')
print(metrics_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print('\nSaved:', OUTPUT)
print('Saved:', REPORT)
print('DONE!')
