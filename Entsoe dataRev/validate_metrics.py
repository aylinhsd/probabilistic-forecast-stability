#!/usr/bin/env python3
"""Independent recomputation of saved evaluation metrics from original prediction CSVs.

Run on Buck from any directory:
 python validate_metrics_from_predictions.py --config evaluation_config.json

Requires pandas and numpy. Does not import evaluate_all_models.py.
Checks overall and monthly rows, quantile scores, interval statistics,
Christoffersen input counts, common-hour alignment and actual-price agreement.
Does NOT prove model-training leakage freedom or test assumptions.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def near(a, b, atol=1e-7):
    try:
        a, b = float(a), float(b)
    except (ValueError, TypeError):
        return False
    return bool(np.isclose(a, b, rtol=1e-8, atol=atol, equal_nan=True))


def score(y, p, quant, settings):
    e = y - p
    m = np.abs(y) >= float(settings['mape_min_abs_actual'])
    den = np.abs(y) + np.abs(p)
    sm = np.zeros(len(y), dtype=float)
    np.divide(200 * np.abs(e), den, out=sm, where=den > 0)
    s = dict(Hours=len(y), MAE_EUR_MWh=np.abs(e).mean(),
             RMSE_EUR_MWh=np.sqrt(np.square(e).mean()),
             Bias_Actual_Minus_Forecast_EUR_MWh=e.mean(),
             MAPE_Percent=(100 * (np.abs(e[m]) / np.abs(y[m])).mean()) if m.any() else np.nan,
             MAPE_Excluded=int((~m).sum()), MAPE_Eligible=int(m.sum()),
             sMAPE_Percent=sm.mean(), Quantile_Count=len(quant))
    if quant:
        levels = sorted(quant)
        losses = []
        for tau in levels:
            delta = y - quant[tau]
            loss = np.where(delta >= 0, tau * delta, (tau - 1) * delta)
            s[f'Pinball_q{tau:.3f}_EUR_MWh'] = loss.mean()
            s[f'Quantile_Coverage_q{tau:.3f}_Percent'] = 100 * (y <= quant[tau]).mean()
            losses.append(loss.mean())
        s['Mean_Pinball_EUR_MWh'] = np.mean(losses)
        dense = (len(levels) >= settings['crps_min_quantiles'] and levels[0] <= .02
                 and levels[-1] >= .98 and max(np.diff(levels)) <= .05)
        s['Approx_CRPS_EUR_MWh'] = 2 * np.trapezoid(losses, levels) if dense else np.nan
    else:
        s['Mean_Pinball_EUR_MWh'] = np.nan
        s['Approx_CRPS_EUR_MWh'] = np.nan
    for level in settings['interval_levels']:
        low, high = (1 - level) / 2, (1 + level) / 2
        lo = next((q for q in quant if abs(q-low) < 1e-7), None)
        hi = next((q for q in quant if abs(q-high) < 1e-7), None)
        if lo is None or hi is None:
            continue
        lower, upper = quant[lo], quant[hi]
        alpha = 1 - level
        covered = (y >= lower) & (y <= upper)
        prefix = f'PI{round(level*100)}_'
        s[prefix+'Coverage_Percent'] = 100 * covered.mean()
        s[prefix+'Coverage_Gap_pp'] = 100 * (covered.mean() - level)
        s[prefix+'Width_EUR_MWh'] = (upper - lower).mean()
        penalty = (upper - lower) + 2/alpha * np.where(y < lower, lower-y, 0) + 2/alpha * np.where(y > upper, y-upper, 0)
        s[prefix+'Winkler_EUR_MWh'] = penalty.mean()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='evaluation_config.json', help='Path to evaluation config JSON')
    ap.add_argument('--max-examples', type=int, default=15)
    args = ap.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    root = config_path.parent
    config = json.loads(config_path.read_text())
    settings = config['settings']
    result_dir = root / settings.get('output_folder', 'evaluation_results')
    metrics_file = result_dir / 'all_model_metrics.csv'
    christ_file = result_dir / 'christoffersen_tests.csv'
    if not metrics_file.exists() or not christ_file.exists():
        sys.exit(f'Missing {metrics_file} or {christ_file}. Run evaluate_all_models.py first.')
    saved = pd.read_csv(metrics_file)
    christ = pd.read_csv(christ_file)
    models = {}
    errors = []
    warnings = []
    for cfg in config['models']:
        if not cfg.get('enabled', True):
            continue
        name = cfg['name']
        f = root / cfg['file']
        if not f.exists():
            sys.exit(f'Missing prediction file: {f}')
        raw = pd.read_csv(f, low_memory=False)
        tcol = cfg.get('timestamp_column', 'Target_UTC')
        acol = cfg.get('actual_column', 'Target_Price_EUR_MWh')
        pcol = cfg['point_column']
        d = raw[[tcol, acol, pcol]].copy()
        d.columns = ['time', 'actual', 'point']
        d['time'] = pd.to_datetime(d.time, utc=True)
        if d.time.isna().any() or d.time.duplicated().any():
            errors.append(f'{name}: duplicate/missing timestamps')
        d['actual'] = pd.to_numeric(d.actual)
        d['point'] = pd.to_numeric(d.point)
        if not np.isfinite(d[['actual','point']].to_numpy()).all():
            errors.append(f'{name}: nonfinite actual/point')
        d['local_date'] = d.time.dt.tz_convert(settings['timezone']).dt.date
        end = pd.Timestamp(settings['test_end']).date() if settings.get('test_end') else None
        if end is not None:
            d = d.loc[d.local_date <= end].copy()
        d = d.sort_values('time').reset_index(drop=True)
        q = {}
        for key, col in cfg.get('quantile_columns', {}).items():
            # Independently align supplied quantiles by timestamp, not row number
            tmp = pd.DataFrame({'time': pd.to_datetime(raw[tcol], utc=True), 'value': pd.to_numeric(raw[col])})
            q[float(key)] = tmp.set_index('time').reindex(d.time)['value'].to_numpy(float)
        if cfg.get('calibrate_from_validation', False):
            start_val = pd.Timestamp(settings['validation_start']).date()
            start_test = pd.Timestamp(settings['test_start']).date()
            val = d[(d.local_date >= start_val) & (d.local_date < start_test)]
            if len(val) < settings['minimum_calibration_rows']:
                errors.append(f'{name}: insufficient validation rows: {len(val)}')
            residual = val.actual.to_numpy(float) - val.point.to_numpy(float)
            for tau in settings['calibration_quantiles']:
                tau = float(tau)
                q[tau] = d.point.to_numpy(float) + np.quantile(residual, tau)
        test_start = pd.Timestamp(settings['test_start']).date()
        mask = d.local_date >= test_start
        t = d.loc[mask].reset_index(drop=True)
        q = {tau: vals[mask.to_numpy()] for tau, vals in q.items()}
        models[name] = (t, q)
        print(f'Loaded {name}: {len(t):,} test rows, {len(q)} quantiles')
    common = sorted(set.intersection(*(set(d.time) for d, _ in models.values())))
    if not common:
        sys.exit('No shared timestamps')
    print(f'Common test timestamps: {len(common):,} ({common[0]} to {common[-1]})')
    counts_checked = 0
    metrics_checked = 0
    ref_actual = None
    for name, (d, q) in models.items():
        ix = pd.Index(d.time).get_indexer(common)
        aligned = d.iloc[ix].reset_index(drop=True)
        aq = {tau: values[ix] for tau, values in q.items()}
        if ref_actual is None:
            ref_actual = aligned.actual.to_numpy(float)
        elif not np.allclose(ref_actual, aligned.actual.to_numpy(float), atol=1e-7, rtol=0):
            errors.append(f'{name}: actual prices differ from other models at shared timestamps')
        time_diffs = aligned.time.diff().dropna()
        gap_count = int((time_diffs != pd.Timedelta(hours=1)).sum())
        if gap_count:
            warnings.append(f'{name}: {gap_count} non-hourly gaps in common test timestamps')
        groups = [('Overall', np.arange(len(aligned)))]
        monthly = aligned.time.dt.tz_convert(settings['timezone']).dt.strftime('%Y-%m')
        groups += [(month, np.flatnonzero(monthly.to_numpy() == month)) for month in sorted(monthly.unique())]
        for period, positions in groups:
            y = aligned.actual.to_numpy(float)[positions]
            p = aligned.point.to_numpy(float)[positions]
            qs = {tau: values[positions] for tau, values in aq.items()}
            independent = score(y, p, qs, settings)
            matching = saved[(saved.Model == name) & (saved.Period == period)]
            if len(matching) != 1:
                errors.append(f'{name}/{period}: expected one saved metrics row, found {len(matching)}')
                continue
            original = matching.iloc[0]
            for field, value in independent.items():
                if field not in original:
                    errors.append(f'{name}/{period}: missing metric {field}')
                elif not near(value, original[field]):
                    errors.append(f'{name}/{period}/{field}: saved={original[field]} independent={value}')
                metrics_checked += 1
            for level in settings['interval_levels']:
                lo_tau, hi_tau = (1-level)/2, (1+level)/2
                lo = next((tau for tau in qs if abs(tau-lo_tau) < 1e-7), None)
                hi = next((tau for tau in qs if abs(tau-hi_tau) < 1e-7), None)
                if lo is None or hi is None:
                    continue
                violation = ((y < qs[lo]) | (y > qs[hi])).astype(int)
                n00 = int(((violation[:-1] == 0) & (violation[1:] == 0)).sum())
                n01 = int(((violation[:-1] == 0) & (violation[1:] == 1)).sum())
                n10 = int(((violation[:-1] == 1) & (violation[1:] == 0)).sum())
                n11 = int(((violation[:-1] == 1) & (violation[1:] == 1)).sum())
                c = christ[(christ.Model == name) & (christ.Period == period) & (christ.Interval_Level_Percent == round(100*level))]
                if len(c) != 1:
                    errors.append(f'{name}/{period}/{level}: expected one Christoffersen row, found {len(c)}')
                    continue
                for field, value in {'Observations':len(y),'Violations':int(violation.sum()),
                                     'N00':n00,'N01':n01,'N10':n10,'N11':n11,
                                     'Observed_Coverage_Percent':100*(1-violation.mean())}.items():
                    if not near(c.iloc[0][field], value):
                        errors.append(f'{name}/{period}/{level}/{field}: saved={c.iloc[0][field]} independent={value}')
                    counts_checked += 1
    print(f'Independent metric comparisons: {metrics_checked:,}')
    print(f'Independent Christoffersen count comparisons: {counts_checked:,}')
    for msg in warnings:
        print('WARNING:', msg)
    if errors:
        print(f'FAIL: {len(errors)} discrepancies')
        for msg in errors[:args.max_examples]:
            print('  -', msg)
        if len(errors) > args.max_examples:
            print(f'  ... {len(errors)-args.max_examples} additional discrepancies')
        return 1
    print('PASS: all checked saved scores and violation counts match independent recomputation')
    print('LIMITATION: This does not verify how models were trained or rule out data leakage.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
