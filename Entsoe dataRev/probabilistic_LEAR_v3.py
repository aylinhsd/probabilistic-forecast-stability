"""Fixed, hour-specific conditional quantile LEAR v3 for Italy North.

Train through 2024-12-31; validation 2025-01-01 to 2025-06-30;
test from 2025-07-01. Validation chooses alpha; models fit on Train only.
No test outcomes are used for tuning, fitting or quantile rearrangement.

Usage:
  python probabilistic_LEAR_v3.py --smoke-test
  python probabilistic_LEAR_v3.py --full
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'ENTSOE_ML_dataset_updated.csv'
CONFIG = ROOT / 'evaluation_config.json'


def pinball_mean(y, q, tau):
    e = y - q
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


def read_quantiles():
    if not CONFIG.exists():
        raise FileNotFoundError(f'Missing evaluation config: {CONFIG}')
    settings = json.loads(CONFIG.read_text())['settings']
    levels = np.asarray(settings['calibration_quantiles'], dtype=float)
    if len(levels) != 101 or np.any(np.diff(levels) <= 0) or not (0 < levels[0] < levels[-1] < 1):
        raise ValueError('Expected exactly 101 strictly increasing quantiles from evaluation_config.json')
    return levels


def load_dataset():
    lag_cols = [f'Price_UTC_Lag_{i}h' for i in range(1, 169)]
    stat_cols = [f'{stat}_D-{day}' for day in range(1, 8)
                 for stat in ('mean', 'minimum', 'maximum', 'std')]
    required = ['Target_Date_Italy', 'Target_UTC', 'Target_Price_EUR_MWh',
                'Target_Hour_Italy', 'Day_Of_Week', 'Month', 'Split'] + lag_cols + stat_cols
    df = pd.read_csv(SOURCE, low_memory=False)
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if df[required].isna().any().any():
        raise ValueError('Missing required data')
    dates = pd.to_datetime(df['Target_Date_Italy'])
    expected = np.where(dates <= pd.Timestamp('2024-12-31'), 'Train',
                        np.where(dates <= pd.Timestamp('2025-06-30'), 'Validation', 'Test'))
    if not np.array_equal(df['Split'].to_numpy(), expected):
        raise ValueError('Split/date boundaries mismatch')
    utc = pd.to_datetime(df['Target_UTC'], utc=True)
    if utc.duplicated().any():
        raise ValueError('Duplicate UTC target timestamps')
    if not np.array_equal(utc.dt.tz_convert('Europe/Rome').dt.date,
                          dates.dt.date):
        raise ValueError('Local target dates disagree with UTC timestamps')
    if not np.array_equal(utc.dt.tz_convert('Europe/Rome').dt.hour.to_numpy(),
                          df['Target_Hour_Italy'].to_numpy()):
        raise ValueError('Local target hours disagree with UTC timestamps')
    calendar = pd.get_dummies(df[['Day_Of_Week', 'Month']].astype(int).astype(str),
                              prefix=['weekday', 'month'], dtype=np.float64)
    X = np.column_stack([df[lag_cols + stat_cols].to_numpy(dtype=np.float64),
                         calendar.to_numpy(dtype=np.float64)])
    y = df['Target_Price_EUR_MWh'].to_numpy(dtype=np.float64)
    hours = df['Target_Hour_Italy'].to_numpy(dtype=int)
    split = df['Split'].to_numpy()
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError('Nonfinite features/target')
    if X.shape[1] != 215 or not np.isin(hours, np.arange(24)).all():
        raise ValueError('Unexpected feature count or local hour')
    for s in ('Train', 'Validation', 'Test'):
        if not np.any(split == s):
            raise ValueError(f'Empty {s} split')
    return df, X, y, hours, split


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke-test', action='store_true', help='Only hour 12 and 3 quantiles, no test evaluation')
    mode.add_argument('--full', action='store_true', help='All 24 hours and 101 quantiles')
    parser.add_argument('--alphas', nargs='+', type=float, default=[0.001, 0.01, 0.1],
                        help='L1 regularization candidates (validation tuned)')
    args = parser.parse_args()
    if not all(a >= 0 and np.isfinite(a) for a in args.alphas):
        raise ValueError('Alphas must be finite and nonnegative')
    all_levels = read_quantiles()
    levels = np.array([0.1, 0.5, 0.9]) if args.smoke_test else all_levels
    model_hours = [12] if args.smoke_test else list(range(24))
    print('Loading data...', flush=True)
    df, X, y, hours, split = load_dataset()
    print(f'Rows={len(df)} Features={X.shape[1]} Quantiles={len(levels)} Hours={len(model_hours)}', flush=True)
    print(f'Alpha candidates={args.alphas}', flush=True)
    train = split == 'Train'
    valid = split == 'Validation'
    test = split == 'Test'
    # Tune a single alpha across the selected quantiles and hours.
    # Using a global alpha keeps the tuning budget controlled and auditable.
    validation_results = []
    start = time.perf_counter()
    for alpha in args.alphas:
        losses = []
        nobs = []
        for h in model_hours:
            tr = train & (hours == h)
            va = valid & (hours == h)
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xva = scaler.transform(X[va])
            for tau in levels:
                model = QuantileRegressor(quantile=float(tau), alpha=float(alpha), solver='highs')
                model.fit(Xtr, y[tr])
                losses.append(pinball_mean(y[va], model.predict(Xva), tau))
                nobs.append(int(va.sum()))
        score = float(np.average(losses, weights=nobs))
        validation_results.append({'alpha': alpha, 'validation_mean_pinball': score})
        print(f'alpha={alpha:g} validation mean pinball={score:.6f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    selected = min(validation_results, key=lambda x: x['validation_mean_pinball'])['alpha']
    print(f'Selected alpha: {selected:g}', flush=True)

    # Final fits use TRAIN only, consistent with the original fixed LEAR v3.
    # Store only Validation/Test predictions; avoid falsely treating train fits as OOS.
    included = valid | test
    if args.smoke_test:
        included &= hours == 12
    selected_indices = np.flatnonzero(included)
    preds = np.full((len(selected_indices), len(levels)), np.nan, dtype=np.float64)
    positions = np.full(len(df), -1, dtype=int)
    positions[selected_indices] = np.arange(len(selected_indices))
    for h in model_hours:
        tr = train & (hours == h)
        target_indices = np.flatnonzero(included & (hours == h))
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xt = scaler.transform(X[target_indices])
        for j, tau in enumerate(levels):
            model = QuantileRegressor(quantile=float(tau), alpha=float(selected), solver='highs')
            model.fit(Xtr, y[tr])
            preds[positions[target_indices], j] = model.predict(Xt)
        print(f'Final fit hour={h:02d} done elapsed={time.perf_counter()-start:.1f}s', flush=True)
    if not np.isfinite(preds).all():
        raise ValueError('Missing/nonfinite quantile predictions')
    crossing_rows = int(np.any(np.diff(preds, axis=1) < -1e-8, axis=1).sum())
    # Sorting is monotonic rearrangement. It enforces noncrossing at each target
    # without accessing actual outcomes; quantify its frequency for the report.
    preds = np.sort(preds, axis=1)
    print(f'Quantile crossing rows before rearrangement: {crossing_rows}/{len(preds)}', flush=True)
    cols = ['Target_Date_Italy', 'Target_UTC', 'Target_Hour_Italy',
            'Target_Price_EUR_MWh', 'Split']
    if 'Issue_Time_UTC' in df.columns:
        cols.insert(2, 'Issue_Time_UTC')
    out = df.iloc[selected_indices][cols].copy()
    for j, tau in enumerate(levels):
        out[f'Q_{tau:.3f}'] = preds[:, j]
    out = out.sort_values('Target_UTC')
    prefix = 'probabilistic_LEAR_v3_smoke' if args.smoke_test else 'ENTSOE_Probabilistic_LEAR_v3'
    path = ROOT / f'{prefix}_predictions.csv'
    report = ROOT / f'{prefix}_metadata.json'
    out.to_csv(path, index=False)
    summary = {
        'model': 'Fixed 24-hour-specific L1-regularized conditional quantile regression',
        'mode': 'smoke' if args.smoke_test else 'full',
        'source': SOURCE.name, 'training': 'through 2024-12-31',
        'validation': '2025-01-01 through 2025-06-30',
        'test': 'from 2025-07-01', 'model_hours': model_hours,
        'quantiles': [float(q) for q in levels],
        'alpha_candidates': args.alphas, 'validation_results': validation_results,
        'selected_alpha': selected, 'validation_selection_metric': 'mean pinball loss',
        'fit_scope': 'Train only', 'scaling': 'Train only, per local delivery hour',
        'crossing_rows_before_rearrangement': crossing_rows,
        'rows_saved': len(out), 'elapsed_seconds': round(time.perf_counter()-start, 2),
        'notes': 'Q columns are direct conditional quantile predictions; not validation-residual offsets. '
                 'Train predictions intentionally omitted. Rearrangement uses predictions only.'
    }
    report.write_text(json.dumps(summary, indent=2) + '\n')
    print('Saved:', path, flush=True)
    print('Saved:', report, flush=True)
    print('Completed in', summary['elapsed_seconds'], 'seconds', flush=True)


if __name__ == '__main__':
    main()
