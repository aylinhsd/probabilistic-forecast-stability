"""Preliminary LEAR-style pooled LASSO baseline for ENTSO-E Italy North.

Requires: pandas, numpy, scikit-learn, openpyxl.
Run in same directory as ENTSOE_ML_dataset.csv.
"""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'ENTSOE_ML_dataset.csv'
PREDICTIONS = ROOT / 'ENTSOE_LEAR_style_predictions.csv'
REPORT = ROOT / 'ENTSOE_LEAR_style_report.xlsx'
# Regularization strengths, selected using validation year 2021 only.
ALPHAS = [0.03, 0.1, 0.3, 1.0]

print('Loading dataset...', flush=True)
df = pd.read_csv(SOURCE, low_memory=False)
lag_cols = [f'Price_UTC_Lag_{i}h' for i in range(1, 169)]
stat_cols = [f'{s}_D-{i}' for i in range(2, 9) for s in ['mean','minimum','maximum','std']]
required = ['Target_Date_Italy', 'Target_UTC', 'Target_Price_EUR_MWh', 'Split',
            'Day_Of_Week', 'Month', 'Target_Hour_Italy'] + lag_cols + stat_cols
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f'Missing required columns: {missing}')
if df[required].isna().any().any():
    raise ValueError('Missing values in required columns')
if df['Target_UTC'].duplicated().any():
    raise ValueError('Duplicate target UTC timestamps')

# Hour/day/month are known before the forecast. One-hot encoding avoids
# treating the labels as continuous numerical magnitudes.
calendar = pd.get_dummies(
    df[['Target_Hour_Italy','Day_Of_Week','Month']].astype(int).astype(str),
    prefix=['hour','weekday','month'], dtype=np.float32
)
X = np.column_stack((df[lag_cols + stat_cols].to_numpy(dtype=np.float32),
                     calendar.to_numpy(dtype=np.float32)))
feature_names = lag_cols + stat_cols + calendar.columns.tolist()
y = df['Target_Price_EUR_MWh'].to_numpy(dtype=np.float64)
splits = df['Split'].astype(str).to_numpy()
train = splits == 'Train'
valid = splits == 'Validation'
if not train.any() or not valid.any():
    raise ValueError('Train and Validation splits are required')
if not np.isfinite(X).all() or not np.isfinite(y).all():
    raise ValueError('Non-finite features or target')

print(f'{X.shape[0]:,} hourly observations, {X.shape[1]} model features', flush=True)
print('Fitting scaler on training years only...', flush=True)
scaler = StandardScaler()
X_train = scaler.fit_transform(X[train]).astype(np.float64)
X_valid = scaler.transform(X[valid]).astype(np.float64)

results = []
best_alpha = None
best_mae = np.inf
best_model = None
for alpha in ALPHAS:
    start = time.time()
    model = Lasso(alpha=alpha, max_iter=1500, tol=0.002, selection='cyclic')
    model.fit(X_train, y[train])
    p = model.predict(X_valid)
    mae = mean_absolute_error(y[valid], p)
    rmse = np.sqrt(mean_squared_error(y[valid], p))
    results.append({'Alpha': alpha, 'Validation_MAE': mae,
                    'Validation_RMSE': rmse,
                    'Nonzero_Coefficients': int(np.count_nonzero(model.coef_)),
                    'Seconds': round(time.time()-start, 1)})
    print(f'alpha={alpha:g}: validation MAE={mae:.3f}, RMSE={rmse:.3f}, '
          f'nonzero={np.count_nonzero(model.coef_)}, seconds={results[-1]["Seconds"]}', flush=True)
    if mae < best_mae:
        best_mae = mae
        best_alpha = alpha
        best_model = model

print(f'Chosen alpha (2021 validation MAE): {best_alpha}', flush=True)
# Fit only on 2015-2020. Do not use 2021+ for fitting, to maintain a clean
# initial evaluation on 2022-2023 and 2024-2026.
# Report metrics by period, and save full hourly predictions.
preds = np.empty(len(df), dtype=np.float64)
for start in range(0, len(df), 10000):
    end = min(start+10000, len(df))
    preds[start:end] = best_model.predict(
        scaler.transform(X[start:end]).astype(np.float64))

output = df[['Target_Date_Italy','Target_UTC','Target_Price_EUR_MWh','Split']].copy()
output['LEAR_Style_Prediction_EUR_MWh'] = preds
output['Error_EUR_MWh'] = output['Target_Price_EUR_MWh'] - preds
output['Absolute_Error_EUR_MWh'] = np.abs(output['Error_EUR_MWh'])
output.to_csv(PREDICTIONS, index=False)

periods = []
for split in ['Train','Validation','Test','Robustness']:
    mask = splits == split
    if mask.any():
        periods.append({'Split': split, 'Hours': int(mask.sum()),
                        'MAE_EUR_MWh': mean_absolute_error(y[mask], preds[mask]),
                        'RMSE_EUR_MWh': np.sqrt(mean_squared_error(y[mask], preds[mask]))})
metrics = pd.DataFrame(periods)
output['Year'] = pd.to_datetime(output['Target_Date_Italy']).dt.year
yearly = output.groupby('Year').agg(
    Hours=('Error_EUR_MWh','size'),
    MAE_EUR_MWh=('Absolute_Error_EUR_MWh','mean'),
    RMSE_EUR_MWh=('Error_EUR_MWh', lambda z: np.sqrt(np.mean(np.square(z))))
).reset_index()
coefs = pd.DataFrame({'Feature': feature_names,
                      'Standardized_Coefficient': best_model.coef_})
coefs['Absolute_Coefficient'] = coefs['Standardized_Coefficient'].abs()
coefs = coefs.sort_values('Absolute_Coefficient', ascending=False)
method = pd.DataFrame({'Setting': ['Method','Target','Historical price features',
    'Calendar features','Training period','Validation period','Test period',
    'Robustness period','Selected alpha','Selection criterion',
    'Important limitation'],
    'Value': ['Pooled LASSO regression (preliminary LEAR-style; not canonical LEAR)',
    'Hourly price EUR/MWh', '168 UTC cutoff-anchored lags + 28 daily stats',
    'One-hot hour, weekday, month','2015-2020','2021','2022-2023','2024-2026',
    best_alpha,'Lowest validation MAE in 2021',
    'Static coefficients; no rolling re-estimation; limited feature set']})
with pd.ExcelWriter(REPORT, engine='openpyxl') as writer:
    method.to_excel(writer, sheet_name='Methodology', index=False)
    metrics.to_excel(writer, sheet_name='Period Metrics', index=False)
    yearly.to_excel(writer, sheet_name='Yearly Metrics', index=False)
    pd.DataFrame(results).to_excel(writer, sheet_name='Alpha Selection', index=False)
    coefs.to_excel(writer, sheet_name='Coefficients', index=False)
print('\nRESULTS\n' + metrics.to_string(index=False), flush=True)
print('\nSaved:', PREDICTIONS, '\nSaved:', REPORT, flush=True)
