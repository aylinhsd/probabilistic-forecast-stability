"""Chronological rolling LEAR for Italy North day-ahead prices.

Run: python LEAR_rolling_v1.py --dry-run
     python LEAR_rolling_v1.py --frequency monthly --start 2025-07-01 --end 2025-07-07
     python LEAR_rolling_v1.py --frequency daily --start 2025-07-01 --end 2026-10-13

At issue time for delivery D (on D-1), training/calibration targets are
restricted to delivery dates <= D-2. This conservative availability rule
assumes day-ahead prices for D-2 have been published by the issue time.
The CSV's historical features must also have been constructed without leakage.
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
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
LAGS = [f'Price_UTC_Lag_{i}h' for i in range(1, 169)]
STATS = [f'{stat}_D-{day}' for day in range(1, 8) for stat in ('mean','minimum','maximum','std')]
LEVELS = (80, 85, 90)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=BASE/'ENTSOE_ML_dataset_updated.csv')
    p.add_argument('--output', type=Path, default=BASE/'LEAR_rolling_outputs')
    p.add_argument('--start', default='2025-07-01')
    p.add_argument('--end', default='2026-10-13')
    p.add_argument('--frequency', choices=['daily','weekly','monthly'], default='monthly')
    p.add_argument('--window', choices=['expanding','rolling'], default='expanding')
    p.add_argument('--window-days', type=int, default=1095, help='Training days for fixed-length rolling window')
    p.add_argument('--calibration-days', type=int, default=181, help='Past days of OOS residuals for empirical quantiles')
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--max-iter', type=int, default=100000)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--max-days', type=int, default=None, help='Short smoke test, e.g. 2')
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


def retrain_due(day, last, frequency):
    if last is None or frequency == 'daily':
        return True
    if frequency == 'weekly':
        return (day - last).days >= 7
    return day.to_period('M') != last.to_period('M')


def main():
    a = args_parser()
    if a.window_days < 30 or a.calibration_days < 7 or a.alpha <= 0:
        raise ValueError('Invalid window-days, calibration-days or alpha')
    start, end = pd.Timestamp(a.start).normalize(), pd.Timestamp(a.end).normalize()
    if start > end:
        raise ValueError('Start date must be <= end date')
    if start < pd.Timestamp('2025-07-01'):
        raise ValueError('Start >= 2025-07-01 required: validation residuals are the initial calibration set')
    dates = pd.date_range(start, end, freq='D')
    if a.max_days is not None:
        dates = dates[:a.max_days]
    print(f'Period: {dates.min().date()} to {dates.max().date()}, {len(dates)} days', flush=True)
    print(f'Retraining: {a.frequency}; window: {a.window}; alpha: {a.alpha}', flush=True)
    print('Training and calibration information cutoff: delivery D-2', flush=True)
    if a.dry_run:
        print('DRY RUN: configuration valid; no CSV loaded, no models trained.')
        print('Input exists:', a.data.is_file(), a.data)
        return
    if not a.data.is_file():
        raise FileNotFoundError(a.data)
    a.output.mkdir(parents=True, exist_ok=True)
    output_file = a.output/'rolling_predictions.csv'
    config_file = a.output/'configuration.json'
    config = dict(data=str(a.data.resolve()), start=str(start.date()), end=str(end.date()),
                  frequency=a.frequency, window=a.window, window_days=a.window_days,
                  calibration_days=a.calibration_days, alpha=a.alpha, max_iter=a.max_iter)
    if a.resume and config_file.exists():
        old_config = json.loads(config_file.read_text())
        if old_config != config:
            raise ValueError('Resume configuration differs from saved run; use another output folder')
    elif a.resume and output_file.exists():
        raise ValueError('Cannot resume without configuration.json')
    elif not a.resume and output_file.exists():
        raise FileExistsError(f'{output_file} already exists. Use --resume or a new --output folder.')
    config_file.write_text(json.dumps(config, indent=2))
    print('Reading dataset...', flush=True)
    df = pd.read_csv(a.data, low_memory=False)
    needed = ['Target_Date_Italy','Target_UTC','Target_Hour_Italy','Target_Price_EUR_MWh',
              'Day_Of_Week','Month','Split'] + LAGS + STATS
    absent = sorted(set(needed)-set(df.columns))
    if absent:
        raise ValueError(f'Missing required columns: {absent}')
    if df[needed].isna().any().any():
        raise ValueError('Missing values in required dataset columns')
    df['Target_Date_Italy'] = pd.to_datetime(df['Target_Date_Italy']).dt.normalize()
    df['Target_UTC'] = pd.to_datetime(df['Target_UTC'], utc=True)
    df = df.sort_values('Target_UTC').reset_index(drop=True)
    if df['Target_UTC'].duplicated().any():
        raise ValueError('Duplicate UTC targets')
    calendar = pd.get_dummies(df[['Day_Of_Week','Month']].astype(int).astype(str),
                              prefix=['weekday','month'], dtype=np.float64)
    # Fixed columns across all retraining dates; avoid changing feature dimensions.
    X = np.column_stack((df[LAGS+STATS].to_numpy(dtype=np.float64),
                         calendar.to_numpy(dtype=np.float64)))
    y = df['Target_Price_EUR_MWh'].to_numpy(dtype=np.float64)
    hrs = df['Target_Hour_Italy'].to_numpy(dtype=int)
    ds = df['Target_Date_Italy']
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite features or prices')
    if not ((df['Split'].eq('Validation') & ds.between('2025-01-01','2025-06-30')).any()):
        raise ValueError('Missing initial validation period')
    if not (ds.eq(start).any() and ds.eq(dates.max()).any()):
        raise ValueError('Dataset does not contain requested start/end dates')

    # Initial calibration is strictly out-of-sample for a model trained through 2024.
    # Recreate baseline validation predictions from the uploaded fixed LEAR v3 output.
    fixed_file = a.data.parent/'ENTSOE_LEAR_v3_predictions.csv'
    if not fixed_file.is_file():
        raise FileNotFoundError('Initial OOS calibration requires '+str(fixed_file))
    base = pd.read_csv(fixed_file, usecols=['Target_UTC','Split','Target_Price_EUR_MWh',
                                            'LEAR_v3_Prediction_EUR_MWh'])
    base['Target_UTC'] = pd.to_datetime(base['Target_UTC'], utc=True)
    base = base.loc[base['Split'].eq('Validation')].copy()
    base['Date'] = base['Target_UTC'].dt.tz_convert('Europe/Rome').dt.tz_localize(None).dt.normalize()
    base['Residual'] = base['Target_Price_EUR_MWh']-base['LEAR_v3_Prediction_EUR_MWh']
    residuals = base[['Date','Residual']].copy()
    if residuals.empty or residuals['Residual'].isna().any():
        raise ValueError('Baseline validation residuals unavailable')
    print('Initial calibration residuals:',len(residuals),flush=True)

    completed = set()
    if a.resume and output_file.exists():
        old = pd.read_csv(output_file, usecols=['Target_Date_Italy'])
        completed = set(pd.to_datetime(old['Target_Date_Italy']).dt.normalize().unique())
        # Recover rolling-model residuals already observed for completed dates.
        prev = pd.read_csv(output_file, usecols=['Target_Date_Italy','Error_EUR_MWh'])
        prev['Date'] = pd.to_datetime(prev['Target_Date_Italy']).dt.normalize()
        residuals = pd.concat([residuals, prev[['Date','Error_EUR_MWh']].rename(
            columns={'Error_EUR_MWh':'Residual'})],ignore_index=True)
        print('Resume: existing forecast days:',len(completed),flush=True)
    models = None
    last_fit = None
    # For a resumed run, rebuild model at the first remaining day, not every old day.
    total_t0 = time.perf_counter()
    for day in dates:
        if day in completed:
            continue
        day_t0 = time.perf_counter()
        cutoff = day - pd.Timedelta(days=2)
        train_mask = (ds <= cutoff).to_numpy()
        if a.window == 'rolling':
            train_mask &= (ds >= cutoff-pd.Timedelta(days=a.window_days-1)).to_numpy()
        target_mask = ds.eq(day).to_numpy()
        if not target_mask.any():
            raise ValueError(f'No observations for {day.date()}')
        if retrain_due(day, last_fit, a.frequency):
            models = []
            for h in range(24):
                idx = train_mask & (hrs == h)
                if idx.sum() < 100:
                    raise ValueError(f'Too few training examples for hour {h} on {day.date()}')
                scaler = StandardScaler()
                xx = scaler.fit_transform(X[idx])
                model = Lasso(alpha=a.alpha, max_iter=a.max_iter, tol=1e-4, selection='cyclic')
                model.fit(xx,y[idx])
                models.append((scaler,model))
            last_fit = day
            print(f'{day.date()}: fitted 24 models on {train_mask.sum()} rows',flush=True)
        pred = np.full(target_mask.sum(),np.nan)
        target_idx = np.flatnonzero(target_mask)
        for h,(scaler,model) in enumerate(models):
            loc = np.flatnonzero(hrs[target_idx] == h)
            if len(loc):
                pred[loc] = model.predict(scaler.transform(X[target_idx[loc]]))
        if not np.isfinite(pred).all():
            raise ValueError('Non-finite predictions')
        # Use only residuals from already observed days, never current/future targets.
        cal = residuals.loc[(residuals['Date'] <= cutoff) &
                            (residuals['Date'] >= cutoff-pd.Timedelta(days=a.calibration_days-1)),
                            'Residual'].to_numpy(dtype=float)
        if len(cal) < 200:
            raise ValueError(f'Not enough historical OOS calibration residuals: {len(cal)}')
        out = df.loc[target_mask,['Target_Date_Italy','Target_UTC','Target_Hour_Italy',
                                  'Target_Price_EUR_MWh']].copy()
        out['Rolling_LEAR_Prediction_EUR_MWh'] = pred
        out['Error_EUR_MWh'] = y[target_mask]-pred
        out['Retrained_On'] = str(last_fit.date())
        out['Training_Through'] = str(cutoff.date() if last_fit == day else (last_fit-pd.Timedelta(days=2)).date())
        out['Calibration_Observations'] = len(cal)
        for level in LEVELS:
            alpha = 1-level/100
            lower_offset, upper_offset = np.quantile(cal,[alpha/2,1-alpha/2])
            out[f'Lower_{level}_EUR_MWh'] = pred+lower_offset
            out[f'Upper_{level}_EUR_MWh'] = pred+upper_offset
        out.to_csv(output_file,mode='a',header=not output_file.exists(),index=False)
        # Results from today are only eligible after day+2; date filter enforces this.
        residuals = pd.concat([residuals,pd.DataFrame({'Date':day,'Residual':out['Error_EUR_MWh'].to_numpy()})],ignore_index=True)
        print(f'{day.date()}: {len(out)} forecasts, calibration n={len(cal)}, '
              f'elapsed {time.perf_counter()-day_t0:.1f}s',flush=True)
    print(f'DONE. Total elapsed {time.perf_counter()-total_t0:.1f}s. Saved: {output_file}',flush=True)
    print('Note: historical rolling simulation; forward testing requires later untouched observations.')

if __name__ == '__main__':
    main()
