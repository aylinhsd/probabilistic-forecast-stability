"""Configurable, chronological electricity-price forecast evaluation.
Run: python3 evaluate_all_models.py --config evaluation_config.json
Dependencies: pandas, numpy, matplotlib. Optional scipy not required.

Point-only forecasts are optionally converted to probabilistic forecasts using
VALIDATION residual quantiles (explicitly marked as post-hoc calibrated).
Quantile forecasts are scored as supplied. Never calibrates on test observations.
CRPS is a quantile-integral approximation for dense quantile grids only.
"""
import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# PART 1 — Point-forecast accuracy. All metrics compare actual y with prediction p.
# MAPE excludes |actual| below the configured threshold and reports how many.
def metric_block(y, p, threshold):
    err = y - p
    denom = np.abs(y)
    eligible = denom >= threshold
    mape = float(100 * np.mean(np.abs(err[eligible]) / denom[eligible])) if eligible.any() else np.nan
    smape_den = np.abs(y) + np.abs(p)
    smape = float(100 * np.mean(np.divide(2 * np.abs(err), smape_den, out=np.zeros_like(err), where=smape_den > 0)))
    return dict(MAE_EUR_MWh=float(np.mean(np.abs(err))), RMSE_EUR_MWh=float(np.sqrt(np.mean(err ** 2))),
                Bias_Actual_Minus_Forecast_EUR_MWh=float(np.mean(err)), MAPE_Percent=mape,
                MAPE_Excluded=int((~eligible).sum()), MAPE_Eligible=int(eligible.sum()),
                sMAPE_Percent=smape)


# PART 2 — Quantile score: asymmetric penalty for errors at quantile tau.
# Example: tau=0.90 penalizes underprediction more than overprediction.
def pinball(y, q, tau):
    e = y - q
    return np.maximum(tau * e, (tau - 1) * e)


# PART 3 — Interval quality: empirical coverage, sharpness (width), Winkler.
# Smaller Winkler is better; narrow intervals are penalized when actual falls outside.
def interval_metrics(y, lower, upper, level):
    alpha = 1 - level
    if alpha <= 0 or alpha >= 1:
        raise ValueError('Interval level must be between 0 and 1')
    width = upper - lower
    score = width + (2 / alpha) * np.maximum(lower - y, 0) + (2 / alpha) * np.maximum(y - upper, 0)
    covered = (y >= lower) & (y <= upper)
    return dict(Coverage_Percent=float(100 * covered.mean()), Coverage_Gap_pp=float(100 * (covered.mean() - level)),
                Width_EUR_MWh=float(width.mean()), Winkler_EUR_MWh=float(score.mean()))


# PART 4 — Christoffersen tests for interval misses (1) vs hits (0).
# LRuc: correct unconditional miss rate; LRind: independence of consecutive misses;
# LRcc: combined test. Reject when p-value < significance.
def christoffersen_tests(violations, nominal_level, significance=0.05):
    """Christoffersen (1998) LRuc, LRind, LRcc for a binary violation series.

    H0 UC: unconditional violation probability = 1 - nominal_level.
    H0 IND: first-order violation transitions are independent.
    H0 CC: both UC and IND hold. Asymptotic chi-square p-values.
    """
    v = np.asarray(violations, dtype=int)
    if len(v) == 0 or not np.isin(v, [0, 1]).all():
        raise ValueError('Christoffersen requires nonempty binary violations')
    if not 0 < nominal_level < 1 or not 0 < significance < 1:
        raise ValueError('Invalid nominal level or significance')

    n = len(v)
    failures = int(v.sum())
    expected_failure_rate = 1 - nominal_level
    empirical_failure_rate = failures / n

    def bernoulli_loglik(successes, trials, prob):
        if trials == 0:
            return 0.0
        if prob == 0:
            return 0.0 if successes == 0 else -np.inf
        if prob == 1:
            return 0.0 if successes == trials else -np.inf
        return successes * np.log(prob) + (trials - successes) * np.log1p(-prob)

    lr_uc = max(0.0, 2 * (
        bernoulli_loglik(failures, n, empirical_failure_rate)
        - bernoulli_loglik(failures, n, expected_failure_rate)
    ))
    p_uc = float(math.erfc(math.sqrt(lr_uc / 2)))  # chi-square(1) survival

    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))
    transitions = n - 1
    result = dict(
        Observations=n, Violations=failures,
        Observed_Coverage_Percent=100 * (1 - empirical_failure_rate),
        Nominal_Coverage_Percent=100 * nominal_level,
        N00=n00, N01=n01, N10=n10, N11=n11,
        LRuc=lr_uc, LRuc_p=p_uc,
        LRuc_Reject=bool(p_uc < significance),
        LRind=np.nan, LRind_p=np.nan, LRind_Reject=None,
        LRcc=np.nan, LRcc_p=np.nan, LRcc_Reject=None,
        Note=''
    )
    if transitions == 0 or n00 + n01 == 0 or n10 + n11 == 0:
        result['Note'] = 'Independence not identifiable: too few transitions or missing prior state'
        return result
    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11)
    pooled = (n01 + n11) / transitions
    ll_independent = bernoulli_loglik(n01 + n11, transitions, pooled)
    ll_markov = (bernoulli_loglik(n01, n00 + n01, p01)
                 + bernoulli_loglik(n11, n10 + n11, p11))
    lr_ind = max(0.0, 2 * (ll_markov - ll_independent))
    lr_cc = lr_uc + lr_ind
    p_ind = float(math.erfc(math.sqrt(lr_ind / 2)))  # chi-square(1)
    p_cc = float(math.exp(-lr_cc / 2))  # chi-square(2)
    result.update(LRind=lr_ind, LRind_p=p_ind,
                  LRind_Reject=bool(p_ind < significance),
                  LRcc=lr_cc, LRcc_p=p_cc,
                  LRcc_Reject=bool(p_cc < significance))
    return result


# PART 5 — Build the interval violation sequence for each requested level.
# Test hourly transitions only when there are no gaps in the UTC time series.
def christoffersen_rows(d, qs, settings, model, period):
    y = d.Actual.to_numpy(float)
    out = []
    for level in settings['interval_levels']:
        lo_tau, hi_tau = (1 - level) / 2, (1 + level) / 2
        lo = next((k for k in qs if abs(k - lo_tau) < 1e-7), None)
        hi = next((k for k in qs if abs(k - hi_tau) < 1e-7), None)
        if lo is None or hi is None:
            continue
        # Ensure chronological order and use only consecutive hourly UTC pairs.
        order = np.argsort(d.Timestamp.to_numpy())
        t = d.Timestamp.iloc[order]
        v = ((y[order] < qs[lo][order]) | (y[order] > qs[hi][order])).astype(int)
        r = christoffersen_tests(v, level, settings.get('christoffersen_significance', 0.05))
        # Independence LR assumes adjacent observations represent adjacent hours.
        # Avoid silently treating data gaps as consecutive observations.
        gaps = int((t.diff().dropna() != pd.Timedelta(hours=1)).sum())
        if gaps:
            r['LRind'] = r['LRind_p'] = r['LRcc'] = r['LRcc_p'] = np.nan
            r['LRind_Reject'] = r['LRcc_Reject'] = None
            r['Note'] = f'Independence/CC unavailable: {gaps} non-hourly gaps'
        out.append({'Model': model, 'Period': period,
                    'Interval_Level_Percent': round(100 * level), **r})
    return out


# PART 6 — Read one model CSV, validate timestamps, and standardize columns.
# Either read native quantile columns OR derive post-hoc quantiles from
# VALIDATION residuals; never use test residuals to construct quantiles.
def load_model(root, cfg, settings):
    path = root / cfg['file']
    if not path.exists():
        raise FileNotFoundError(f"{cfg['name']}: missing file {path}")
    raw = pd.read_csv(path, low_memory=False)
    tc = cfg.get('timestamp_column', 'Target_UTC')
    ac = cfg.get('actual_column', 'Target_Price_EUR_MWh')
    pc = cfg['point_column']
    needed = [tc, ac, pc]
    if any(col not in raw for col in needed):
        raise ValueError(f"{cfg['name']}: missing columns {set(needed) - set(raw.columns)}")
    d = raw[[tc, ac, pc]].copy()
    d.columns = ['Timestamp', 'Actual', 'Point']
    d['Timestamp'] = pd.to_datetime(d['Timestamp'], utc=True, errors='raise')
    if d['Timestamp'].isna().any() or d['Timestamp'].duplicated().any():
        raise ValueError(f"{cfg['name']}: missing or duplicate timestamps")
    if d[['Actual', 'Point']].isna().any().any():
        raise ValueError(f"{cfg['name']}: missing actuals or point predictions")
    d['Actual'] = pd.to_numeric(d['Actual'], errors='raise')
    d['Point'] = pd.to_numeric(d['Point'], errors='raise')
    if not np.isfinite(d[['Actual', 'Point']].to_numpy()).all():
        raise ValueError(f"{cfg['name']}: nonfinite actual or prediction")
    d['Date_Italy'] = d['Timestamp'].dt.tz_convert(settings['timezone']).dt.date
    validation_start = pd.Timestamp(settings['validation_start']).date()
    test_start = pd.Timestamp(settings['test_start']).date()
    test_end = pd.Timestamp(settings['test_end']).date() if settings.get('test_end') else None
    d['Period'] = np.where(d.Date_Italy < validation_start, 'Train',
                           np.where(d.Date_Italy < test_start, 'Validation', 'Test'))
    if test_end is not None:
        d = d[d.Date_Italy <= test_end].copy()
    d = d.sort_values('Timestamp').reset_index(drop=True)
    # Explicit quantile mapping: {"0.05":"Q05", ...}; optional
    mapping = cfg.get('quantile_columns', {})
    quantiles = {}
    for k, col in mapping.items():
        tau = float(k)
        if not 0 < tau < 1:
            raise ValueError(f'{cfg["name"]}: invalid quantile {tau}')
        if col not in raw:
            raise ValueError(f'{cfg["name"]}: missing quantile column {col}')
        quantiles[tau] = pd.Series(pd.to_numeric(raw[col], errors='raise').to_numpy(), index=pd.to_datetime(raw[tc], utc=True)).reindex(d.Timestamp).to_numpy()
    if cfg.get('calibrate_from_validation', False):
        if quantiles:
            raise ValueError(f'{cfg["name"]}: choose supplied quantiles OR calibration, not both')
        cal = d[d.Period == 'Validation']
        if len(cal) < settings['minimum_calibration_rows']:
            raise ValueError(f'{cfg["name"]}: insufficient validation residuals ({len(cal)})')
        residuals = cal.Actual.to_numpy() - cal.Point.to_numpy()
        for tau in settings['calibration_quantiles']:
            tau = float(tau)
            offset = float(np.quantile(residuals, tau))
            quantiles[tau] = d.Point.to_numpy() + offset
    for tau, values in quantiles.items():
        if not np.isfinite(values).all():
            raise ValueError(f'{cfg["name"]}: nonfinite quantiles at {tau}')
    for tau1, tau2 in zip(sorted(quantiles)[:-1], sorted(quantiles)[1:]):
        if (quantiles[tau1] > quantiles[tau2] + 1e-8).any():
            raise ValueError(f'{cfg["name"]}: crossing quantiles {tau1}, {tau2}')
    return d, quantiles


# PART 7 — Evaluate a single model on one period (overall or one month).
# Approximate CRPS uses numerical integration of pinball losses over quantiles.
# It is NOT exact CRPS and omits the extreme tails outside the supplied grid.
def evaluate_one(d, qs, settings, model, period):
    y = d.Actual.to_numpy(float)
    p = d.Point.to_numpy(float)
    result = {'Model': model, 'Period': period, 'Hours': len(d), **metric_block(y, p, settings['mape_min_abs_actual'])}
    result['Quantile_Count'] = len(qs)
    if qs:
        losses = {tau: pinball(y, arr, tau) for tau, arr in qs.items()}
        result['Mean_Pinball_EUR_MWh'] = float(np.mean([v.mean() for v in losses.values()]))
        for tau, loss in sorted(losses.items()):
            result[f'Pinball_q{tau:.3f}_EUR_MWh'] = float(loss.mean())
            result[f'Quantile_Coverage_q{tau:.3f}_Percent'] = float(100 * np.mean(y <= qs[tau]))
        # CRPS = 2 * integral_0^1 pinball(tau) d tau. Approximation only
        # if a sufficiently dense, nearly complete grid is available.
        taus = np.array(sorted(qs))
        dense = len(taus) >= settings['crps_min_quantiles'] and taus[0] <= 0.02 and taus[-1] >= 0.98 and np.max(np.diff(taus)) <= 0.05
        result['Approx_CRPS_EUR_MWh'] = float(2 * np.trapezoid(np.array([losses[t].mean() for t in taus]), taus)) if dense else np.nan
        result['CRPS_Status'] = 'Quantile integral approximation (tails truncated)' if dense else 'Unavailable: quantile grid too sparse'
    else:
        result['Mean_Pinball_EUR_MWh'] = np.nan
        result['Approx_CRPS_EUR_MWh'] = np.nan
        result['CRPS_Status'] = 'Unavailable: no quantile forecasts'
    for level in settings['interval_levels']:
        lo_tau, hi_tau = round((1 - level) / 2, 8), round((1 + level) / 2, 8)
        match_lo = next((k for k in qs if abs(k - lo_tau) < 1e-7), None)
        match_hi = next((k for k in qs if abs(k - hi_tau) < 1e-7), None)
        if match_lo is not None and match_hi is not None:
            z = interval_metrics(y, qs[match_lo], qs[match_hi], level)
            result.update({f'PI{round(level*100)}_{key}': value for key, value in z.items()})
    return result


# PART 8 — Exploratory pairwise comparison of models on matched hours.
# Compare daily averaged absolute-error losses with a HAC variance estimate.
def paired_dm(d1, d2, max_lag):
    """Approximate two-sided DM: daily mean absolute-error loss difference,
    Newey-West long-run variance, normal approximation. Exploratory test.
    """
    a = d1[['Timestamp', 'Date_Italy', 'Actual', 'Point']].merge(
        d2[['Timestamp', 'Actual', 'Point']], on='Timestamp', how='inner', suffixes=('_a', '_b'), validate='one_to_one')
    if len(a) < 100:
        return None
    if not np.allclose(a.Actual_a, a.Actual_b, atol=1e-7, rtol=0):
        raise ValueError('Paired DM: actual prices differ')
    a['diff'] = np.abs(a.Actual_a - a.Point_a) - np.abs(a.Actual_b - a.Point_b)
    z = a.groupby('Date_Italy')['diff'].mean().to_numpy()
    n = len(z)
    if n < 30:
        return None
    x = z - z.mean()
    lag = min(max_lag, n - 1)
    lrv = float(np.dot(x, x) / n)
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * float(np.dot(x[k:], x[:-k]) / n)
    if lrv <= 0:
        return None
    stat = float(z.mean() / np.sqrt(lrv / n))
    p = math.erfc(abs(stat) / math.sqrt(2))
    return {'Matched_Hours': len(a), 'Matched_Days': n, 'Mean_Daily_MAE_Difference_A_Minus_B': float(z.mean()),
            'DM_HAC_z': stat, 'Approx_Two_Sided_p': p, 'HAC_Lag_Days': lag,
            'Note': 'Exploratory daily-loss HAC normal approximation; not a finite-sample exact DM test'}


# PART 9 — Orchestrator: load config, align models, score, save CSVs/plots.
# All enabled models are compared on the INTERSECTION of their test hours.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='evaluation_config.json')
    args = ap.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    root = config_path.parent
    settings = config['settings']
    out = root / settings.get('output_folder', 'evaluation_results')
    out.mkdir(parents=True, exist_ok=True)
    models = {}
    for cfg in config['models']:
        if not cfg.get('enabled', True):
            continue
        name = cfg['name']
        if name in models:
            raise ValueError(f'Duplicate model name: {name}')
        models[name] = load_model(root, cfg, settings)
        print(f'Loaded {name}: {len(models[name][0]):,} rows, {len(models[name][1])} quantiles')
    if not models:
        raise ValueError('No enabled models')
    test_start = pd.Timestamp(settings['test_start']).date()
    test_sets = {}
    for name, (d, qs) in models.items():
        t = d[(d.Period == 'Test') & (d.Date_Italy >= test_start)].copy().reset_index(drop=True)
        if t.empty:
            raise ValueError(f'{name}: no test observations')
        # restrict quantiles to the test rows, preserving aligned row indices
        positions = d.index[(d.Period == 'Test') & (d.Date_Italy >= test_start)].to_numpy()
        q_test = {tau: arr[positions] for tau, arr in qs.items()}
        test_sets[name] = (t, q_test)
    # Comparable overall results: intersection of ALL models' test timestamps.
    common = set(test_sets[next(iter(test_sets))][0].Timestamp)
    for t, _ in test_sets.values():
        common &= set(t.Timestamp)
    if not common:
        raise ValueError('No common test timestamps across models')
    common = pd.DatetimeIndex(sorted(common))
    rows = []
    christoffersen_output = []
    common_test = {}
    for name, (t, qs) in test_sets.items():
        indexed = t.set_index('Timestamp')
        aligned = indexed.reindex(common).rename_axis('Timestamp').reset_index()
        # Original test dataframe and q arrays are sorted by timestamp
        qmap = {tau: pd.Series(arr, index=t.Timestamp).reindex(common).to_numpy() for tau, arr in qs.items()}
        common_test[name] = (aligned, qmap)
        rows.append(evaluate_one(aligned, qmap, settings, name, 'Overall'))
        christoffersen_output.extend(christoffersen_rows(aligned, qmap, settings, name, 'Overall'))
        for month, indices in aligned.groupby(aligned.Timestamp.dt.tz_convert(settings['timezone']).dt.strftime('%Y-%m')).groups.items():
            idx = np.asarray(list(indices), dtype=int)
            subset = aligned.iloc[idx].reset_index(drop=True)
            month_q = {tau: arr[idx] for tau, arr in qmap.items()}
            rows.append(evaluate_one(subset, month_q, settings, name, month))
            if settings.get("christoffersen_monthly", True):
                christoffersen_output.extend(christoffersen_rows(subset, month_q, settings, name, month))
    pd.DataFrame(christoffersen_output).to_csv(out / 'christoffersen_tests.csv', index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / 'all_model_metrics.csv', index=False)
    overall = metrics[metrics.Period == 'Overall'].copy()
    overall.to_csv(out / 'overall_model_summary.csv', index=False)
    # DM on identical timestamps for all enabled models
    dm_rows = []
    names = list(common_test)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            dm = paired_dm(common_test[a][0], common_test[b][0], settings.get('dm_hac_lag_days', 7))
            if dm:
                dm_rows.append({'Model_A': a, 'Model_B': b, **dm})
    pd.DataFrame(dm_rows).to_csv(out / 'paired_dm_mae.csv', index=False)
    # PART 10 — Visualization: percentage labels and absolute accuracy bars.
    # Other plots below cover interval coverage, monthly stability, and a week.
    # Graph: accuracy and percentage metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, col, ylabel in zip(axes.ravel(),
        ['MAE_EUR_MWh', 'RMSE_EUR_MWh', 'MAPE_Percent', 'sMAPE_Percent'],
        ['MAE (EUR/MWh)', 'RMSE (EUR/MWh)', 'MAPE (%)', 'sMAPE (%)']):
        vals = overall[col].to_numpy(dtype=float)
        bars = ax.bar(overall.Model, vals)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.annotate(f'{val:.2f}' + ('%' if 'Percent' in col else ''),
                            (bar.get_x() + bar.get_width()/2, bar.get_height()),
                            ha='center', va='bottom', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0, top=max(vals[np.isfinite(vals)]) * 1.25 if np.isfinite(vals).any() and max(vals[np.isfinite(vals)]) > 0 else 1)
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', alpha=0.2)
    fig.suptitle(f'Common test hours: {len(common):,} | {common.min().date()} to {common.max().date()}')
    fig.tight_layout()
    fig.savefig(out / '01_accuracy_percentages.png', dpi=180)
    plt.close(fig)
    for level in settings['interval_levels']:
        col = f'PI{round(level*100)}_Coverage_Percent'
        if col not in overall:
            continue
        part = overall[overall[col].notna()]
        if part.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(part.Model, part[col])
        for bar, val in zip(bars, part[col]):
            ax.annotate(f'{val:.1f}%', (bar.get_x() + bar.get_width()/2, val), ha='center', va='bottom')
        ax.axhline(level*100, color='black', ls='--', label=f'Target {level*100:.0f}%')
        ax.set_ylim(0, 110)
        ax.set_ylabel('Observed coverage (%)')
        ax.set_title(f'{level*100:.0f}% interval coverage — common test period')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'02_coverage_{round(level*100)}.png', dpi=180)
        plt.close(fig)
    # Monthly stability chart for available 80% intervals
    monthly = metrics[metrics.Period != 'Overall'].copy()
    if 'PI80_Coverage_Percent' in monthly:
        fig, ax = plt.subplots(figsize=(14, 6))
        for name, part in monthly.groupby('Model'):
            if part['PI80_Coverage_Percent'].notna().any():
                ax.plot(part.Period, part.PI80_Coverage_Percent, marker='o', ms=3, label=name)
        ax.axhline(80, ls='--', color='black', label='Target 80%')
        ax.set_ylim(0, 105)
        ax.set_ylabel('Observed coverage (%)')
        ax.set_title('Monthly 80% interval coverage')
        ax.tick_params(axis='x', rotation=60)
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / '03_monthly_80_coverage.png', dpi=180)
        plt.close(fig)
    # Reusable monthly plots for accuracy and probabilistic scores.
    monthly_plots = [
        ('MAE_EUR_MWh', '04_monthly_mae.png', 'MAE (EUR/MWh)'),
        ('MAPE_Percent', '05_monthly_mape.png', 'MAPE (%)'),
        ('Approx_CRPS_EUR_MWh', '06_monthly_approx_crps.png', 'Approximate CRPS (EUR/MWh)'),
        ('PI80_Winkler_EUR_MWh', '07_monthly_winkler_80.png', '80% Winkler score (EUR/MWh)'),
    ]
    for col, filename, ylabel in monthly_plots:
        if col not in monthly.columns or not monthly[col].notna().any():
            continue
        fig, ax = plt.subplots(figsize=(14, 6))
        for name, part in monthly.groupby('Model'):
            part = part.sort_values('Period')
            if part[col].notna().any():
                ax.plot(part.Period, part[col], marker='o', ms=3, label=name)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Monthly model comparison — {ylabel}')
        ax.tick_params(axis='x', rotation=60)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)
    # Optional detailed week: all point models + chosen model's 80% interval.
    plot_cfg = settings.get('example_week', {})
    if plot_cfg.get('start') and plot_cfg.get('end'):
        model_for_interval = plot_cfg.get('interval_model', names[0])
        start, end = pd.Timestamp(plot_cfg['start']).date(), pd.Timestamp(plot_cfg['end']).date()
        reference = common_test[names[0]][0]
        selected = (reference.Date_Italy >= start) & (reference.Date_Italy <= end)
        if selected.any():
            fig, ax = plt.subplots(figsize=(16, 7))
            x = reference.loc[selected, 'Timestamp'].dt.tz_convert(settings['timezone'])
            ax.plot(x, reference.loc[selected, 'Actual'], color='black', lw=1.7, label='Actual day-ahead price')
            for name in names:
                points = common_test[name][0].loc[selected, 'Point']
                ax.plot(x, points, lw=1.2, label=name)
            if model_for_interval in common_test:
                t, qs = common_test[model_for_interval]
                lower = next((qs[k] for k in qs if abs(k - 0.1) < 1e-7), None)
                upper = next((qs[k] for k in qs if abs(k - 0.9) < 1e-7), None)
                if lower is not None and upper is not None:
                    ax.fill_between(x, lower[selected.to_numpy()], upper[selected.to_numpy()], alpha=0.16,
                                    label=f'{model_for_interval}: 80% interval')
            ax.set_ylabel('Price (EUR/MWh)')
            ax.set_title(f'All models and actual day-ahead prices | {start} to {end}')
            ax.legend()
            ax.grid(alpha=0.2)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(out / '08_week_all_models_80_interval.png', dpi=180)
            plt.close(fig)
    print('\nCOMMON TEST HOURS:', len(common))
    cols = ['Model', 'Hours', 'MAE_EUR_MWh', 'RMSE_EUR_MWh', 'MAPE_Percent', 'sMAPE_Percent',
            'Mean_Pinball_EUR_MWh', 'Approx_CRPS_EUR_MWh', 'PI80_Coverage_Percent', 'PI80_Winkler_EUR_MWh']
    print(overall[[c for c in cols if c in overall]].round(3).to_string(index=False))
    print('\nResults:', out)
    print('Note: validation-calibrated intervals are post-hoc baselines; test outcomes are not used for calibration.')

if __name__ == '__main__':
    main()
