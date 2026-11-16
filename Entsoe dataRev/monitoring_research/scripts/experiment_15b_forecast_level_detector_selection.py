#!/usr/bin/env python3
"""Experiment 15b: choose detectors under a common notification budget.

Aim in plain language: damage saved hourly forecasts in controlled ways, then
see which monitor catches the problem promptly without sending too many alerts.

Run from any directory:
    python3 monitoring_research/scripts/experiment_15b_forecast_level_detector_selection.py
    python3 monitoring_research/scripts/experiment_15b_forecast_level_detector_selection.py --self-check

Uses numpy, pandas, matplotlib, scipy; river is optional for ADWIN. All paths
are relative to this script. Outputs go to monitoring_research/{tables,figures,reports}.
No training or original experiment outputs are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'experiment_15b_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

try:
    from river.drift import ADWIN
except ImportError:
    ADWIN = None

ROOT = Path(__file__).resolve().parents[2]
MON = ROOT / 'monitoring_research'
PREFIX = 'experiment_15b'
SIGNALS = ['absolute_crps', 'relative_crps_loss', 'undercoverage',
           'overcoverage', 'volatility_adjusted_crps']
METRICS = SIGNALS + ['coverage', 'width80', 'volatility']
METHODS = ['Fixed P95', 'Tuned quantile', 'CUSUM', 'Page-Hinkley', 'EWMA']
if ADWIN is not None:
    METHODS.append('ADWIN upward-confirmed')


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repetitions', type=int, default=300)
    p.add_argument('--calibration-repetitions', type=int, default=200)
    p.add_argument('--seed', type=int, default=15026)
    p.add_argument('--budget', type=float, default=1.0, help='Baseline notifications per 100 days, per signal')
    p.add_argument('--cooldown', type=int, default=7, help='Minimum days between notifications')
    p.add_argument('--block-days', type=int, default=7)
    p.add_argument('--self-check', action='store_true', help='Run invariant checks only')
    a = p.parse_args()
    if min(a.repetitions, a.calibration_repetitions, a.cooldown, a.block_days) < 1 or a.budget <= 0:
        p.error('Repetitions, block days, cooldown and budget must be positive')
    return a


def crps(y, q, tau):
    e = y[:, None] - q
    return 2 * np.trapezoid(np.maximum(tau * e, (tau - 1) * e), x=tau, axis=1)


def load_hourly():
    """Validate all timestamps/quantiles, retaining complete 23/24/25-hour days."""
    path = ROOT / 'ENTSOE_Probabilistic_LEAR_v3_predictions.csv'
    d = pd.read_csv(path).copy()
    utc = pd.to_datetime(d.Target_UTC, utc=True, errors='raise')
    dates = utc.dt.tz_convert('Europe/Rome').dt.tz_localize(None).dt.normalize()
    if utc.duplicated().any() or utc.isna().any() or not dates.eq(pd.to_datetime(d.Target_Date_Italy)).all():
        raise ValueError('Invalid/duplicate UTC timestamps or inconsistent Italian dates')
    qcols = sorted([c for c in d if c.startswith('Q_')], key=lambda c: float(c[2:]))
    tau = np.array([float(c[2:]) for c in qcols])
    if not np.all(np.diff(tau) > 0) or not all(np.isclose(tau, v).any() for v in (.1, .5, .9)):
        raise ValueError('Invalid probability grid or required quantiles missing')
    # Benchmark matches Experiment 13: point LEAR plus fixed pre-test residual quantiles.
    p = pd.read_csv(ROOT / 'ENTSOE_LEAR_v3_predictions.csv', usecols=[
        'Target_UTC', 'Target_Price_EUR_MWh', 'LEAR_v3_Prediction_EUR_MWh', 'Split'])
    p['utc'] = pd.to_datetime(p.Target_UTC, utc=True)
    if p.utc.duplicated().any():
        raise ValueError('Duplicate benchmark timestamps')
    v = p.loc[p.Split.eq('Validation')]
    if v.empty or v.utc.max() >= utc[d.Split.eq('Test')].min():
        raise ValueError('Benchmark residual calibration must precede native test')
    residual = v.Target_Price_EUR_MWh.to_numpy() - v.LEAR_v3_Prediction_EUR_MWh.to_numpy()
    if not np.isfinite(residual).all():
        raise ValueError('Nonfinite benchmark residuals')
    rq = np.quantile(residual, tau)
    d['utc'] = utc
    d['date'] = dates
    d = d.loc[d.Split.eq('Test')].sort_values('utc').reset_index(drop=True)
    p = p.set_index('utc').reindex(d.utc)
    y = d.Target_Price_EUR_MWh.to_numpy(float)
    q = d[qcols].to_numpy(float)
    pred = p.LEAR_v3_Prediction_EUR_MWh.to_numpy(float)
    if not np.allclose(y, p.Target_Price_EUR_MWh.to_numpy(float), atol=1e-5, rtol=0):
        raise ValueError('Actuals do not match benchmark at UTC timestamps')
    if not np.isfinite(q).all() or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Nonfinite prices or forecasts')
    if np.any(np.diff(q, axis=1) < -1e-8):
        raise ValueError('Crossing native quantiles')
    if not (d.utc.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError('Nonconsecutive hourly data')
    dates = pd.DatetimeIndex(d.date.unique())
    if not (dates.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError('Nonconsecutive local days')
    code = dates.get_indexer(d.date)
    counts = np.bincount(code)
    if not np.isin(counts, [23, 24, 25]).all():
        raise ValueError('Incomplete local calendar days')
    return dict(path=path, dates=dates, y=y, q=q, tau=tau, code=code, counts=counts,
                benchmark=crps(y, pred[:, None] + rq, tau),
                i10=int(np.flatnonzero(np.isclose(tau, .1))[0]),
                i50=int(np.flatnonzero(np.isclose(tau, .5))[0]),
                i90=int(np.flatnonzero(np.isclose(tau, .9))[0]))


def daily_metrics(data, bias=0., width_factor=1.):
    """Affine quantile transformation preserves ordering; actuals/benchmark stay paired."""
    if width_factor <= 0:
        raise ValueError('Width factor must be positive')
    med = data['q'][:, data['i50'], None]
    q = med + width_factor * (data['q'] - med) + bias
    y, code, counts = data['y'], data['code'], data['counts']
    def mean(x):
        return np.bincount(code, weights=x, minlength=len(counts)) / counts
    coverage = mean((y >= q[:, data['i10']]) & (y <= q[:, data['i90']]))
    width = mean(q[:, data['i90']] - q[:, data['i10']])
    c = mean(crps(y, q, data['tau']))
    bench = mean(data['benchmark'])
    vol = np.sqrt(np.maximum(0., mean(y*y) - mean(y)**2))
    # Last column's adjusted CRPS is filled using reference-only coefficients later.
    return np.column_stack([c, c / np.maximum(bench, 1e-9) - 1,
                            np.maximum(0, .8-coverage), np.maximum(0, coverage-.8),
                            np.zeros(len(c)), coverage, width, vol])


def make_folds(n):
    if n < 420:
        raise ValueError('At least 420 consecutive test days required for the declared three-fold design')
    folds = []
    for k, fit_end in enumerate((120, 180, 240), 1):
        cal_end, val_end = fit_end+60, fit_end+120
        test_end = val_end+60 if k < 3 else n
        folds.append(dict(fold=k, fit=(0, fit_end), calibration=(fit_end, cal_end),
                          validation=(cal_end, val_end), test=(val_end, test_end)))
    return folds


def block_indices(size, n, repetitions, block, rng):
    """Noncircular moving blocks, shared by every signal and every detector."""
    block = min(block, size)
    starts = rng.integers(0, size-block+1, size=(repetitions, math.ceil(n/block)))
    return (starts[:, :, None] + np.arange(block)).reshape(repetitions, -1)[:, :n]


def notifications(raw, cooldown):
    """A sustained alarm generates another notification every cooldown days.

    Unlike merging unlimited consecutive alarms, this cannot hide an always-on
    detector behind one long episode. State continues across event onset.
    """
    raw = np.asarray(raw, bool)
    out = np.zeros_like(raw)
    last = np.full(raw.shape[0], -cooldown)
    for t in range(raw.shape[1]):
        out[:, t] = raw[:, t] & ((t-last) >= cooldown)
        last[out[:, t]] = t
    return out


def run_detector(x, method, params, warmup):
    """x: trials x time, already standardized on the fitting period only."""
    reps, n = x.shape
    out = np.zeros_like(x, dtype=bool)
    if method in ('Fixed P95', 'Tuned quantile'):
        out = x > params['threshold']
    elif method == 'CUSUM':
        state = np.zeros(reps)
        for t in range(n):
            state = np.maximum(0., state + x[:, t] - params['k'])
            out[:, t] = state > params['h']
            state[out[:, t]] = 0.
    elif method == 'Page-Hinkley':
        total, minimum = np.zeros(reps), np.zeros(reps)
        for t in range(n):
            total += x[:, t] - params['delta']
            minimum = np.minimum(minimum, total)
            out[:, t] = total-minimum > params['threshold']
            total[out[:, t]] = 0.
            minimum[out[:, t]] = 0.
    elif method == 'EWMA':
        state = np.zeros(reps)
        alpha = params['alpha']
        threshold = params['L'] * np.sqrt(alpha/(2-alpha))
        for t in range(n):
            state = alpha*x[:, t] + (1-alpha)*state
            out[:, t] = state > threshold
            state[out[:, t]] = 0.
    elif method == 'ADWIN upward-confirmed':
        for r in range(reps):
            model = ADWIN(delta=params['delta'], clock=1)
            for t, value in enumerate(x[r]):
                model.update(float(value))
                # Explicit, predeclared directional confirmation. This is a
                # modified ADWIN, not generic two-sided drift detection.
                if model.drift_detected and t >= 14:
                    recent = x[r, t-6:t+1].mean()
                    prior = x[r, t-13:t-6].mean()
                    out[r, t] = recent > max(0., prior)
    else:
        raise ValueError(method)
    return out[:, warmup:]


def grid(method, fit_z):
    if method == 'Fixed P95':
        return [dict(threshold=float(np.quantile(fit_z, .95)), q=.95)]
    if method == 'Tuned quantile':
        values = [float(np.quantile(fit_z, q)) for q in (.95, .975, .99, .995, 1.)]
        values += [max(values)+v for v in (1., 2., 4., 8., 16., 32.)]
        return [dict(threshold=v) for v in sorted(set(values))]
    if method == 'CUSUM':
        return [dict(k=.5, h=float(v)) for v in np.geomspace(.5, 1500, 22)]
    if method == 'Page-Hinkley':
        return [dict(delta=.15, threshold=float(v)) for v in np.geomspace(.5, 1500, 22)]
    if method == 'EWMA':
        return [dict(alpha=.2, L=float(v)) for v in np.geomspace(.5, 60, 22)]
    return [dict(delta=v, clock=1) for v in (.1, .05, .02, .005, .001, .0001, .000001)]


def fit_detector(fit_values, cal_values, indices, method, budget, cooldown):
    mu = float(np.mean(fit_values))
    sd = max(float(np.std(fit_values)), 1e-8)
    fit_z = (fit_values-mu)/sd
    prefix = fit_z[-30:]
    x = np.column_stack([np.broadcast_to(prefix, (len(indices), len(prefix))),
                         (cal_values[indices]-mu)/sd])
    candidates = []
    for order, params in enumerate(grid(method, fit_z)):
        raw = run_detector(x, method, params, len(prefix))
        counts = notifications(raw, cooldown).sum(axis=1)
        rate = float(counts.mean()*100/raw.shape[1])
        candidates.append((rate, order, params, float(raw.mean()*100)))
    eligible = [c for c in candidates if c[0] <= budget+1e-12]
    # All grids are ordered from more to less sensitive. Do not use event tests
    # to tune thresholds. When none passes, keep the least noisy and flag it.
    chosen = min(eligible, key=lambda c: c[1]) if eligible else min(candidates, key=lambda c: c[0])
    return dict(mu=mu, sd=sd, warmup=prefix, params=chosen[2],
                calibration_rate=chosen[0], calibration_alarm_day_pct=chosen[3],
                calibration_budget_met=bool(eligible))


def scenario_specs():
    out = [dict(scenario='none', family='none', bias=0., factor=1., shape='none')]
    for sign, direction in ((1, 'positive'), (-1, 'negative')):
        for label, size in (('small', .5), ('medium', 1.), ('large', 2.)):
            out.append(dict(scenario=f'sudden_bias_{direction}_{label}', family='bias',
                            bias=sign*size, factor=1., shape='sudden'))
        for shape in ('gradual', 'temporary'):
            out.append(dict(scenario=f'{shape}_bias_{direction}_medium', family='bias',
                            bias=sign, factor=1., shape=shape))
    for family, factors in (('narrow', (.8, .5)), ('wide', (1.25, 1.75))):
        for factor in factors:
            out.append(dict(scenario=f'{family}_interval_x{factor:g}', family=family,
                            bias=0., factor=factor, shape='sudden'))
    return out


def wilson(hits, n):
    if not n:
        return np.nan, np.nan
    p, z = hits/n, 1.96
    den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0., center-half), min(1., center+half)


def rate_interval(counts, days):
    """Conditional trial-level normal interval; historical stability is not assumed."""
    rates = np.asarray(counts)*100/days
    mean = rates.mean()
    se = rates.std(ddof=1)/np.sqrt(len(rates)) if len(rates) > 1 else np.nan
    return float(mean), float(max(0., mean-1.96*se)), float(mean+1.96*se)


def score_notifications(notices, onset, shape, cooldown):
    horizon = 3 if shape == 'temporary' else 30
    hit = notices[:, onset:onset+horizon]
    any_hit = hit.any(axis=1)
    delay = np.where(any_hit, hit.argmax(axis=1), np.nan)
    result = dict(detected=any_hit, delay=delay,
                  detect7=notices[:, onset:onset+min(7, horizon)].any(axis=1),
                  detect14=notices[:, onset:onset+min(14, horizon)].any(axis=1),
                  detect30=any_hit,
                  # Restricted delay includes every miss at the horizon, avoiding
                  # deceptively short delays from a detector that rarely fires.
                  restricted_delay=np.where(any_hit, delay, horizon),
                  pre_notifications=notices[:, :onset].sum(axis=1))
    if shape == 'temporary':
        result['recovery_notifications'] = notices[:, onset+3:onset+33].sum(axis=1)
    else:
        result['recovery_notifications'] = np.full(len(notices), np.nan)
    return result


def summarize_trials(trials):
    keys = ['fold', 'stage', 'signal', 'method', 'scenario', 'family', 'shape']
    rows = []
    for key, d in trials.groupby(keys, sort=False):
        r = dict(zip(keys, key))
        n = len(d)
        r.update(trials=n, detection_rate=float(d.detected.mean()),
                 detection7=float(d.detect7.mean()), detection14=float(d.detect14.mean()),
                 detection30=float(d.detect30.mean()), missed_rate=float(1-d.detected.mean()),
                 median_delay_detected=float(d.delay.median()) if d.delay.notna().any() else np.nan,
                 mean_restricted_delay=float(d.restricted_delay.mean()),
                 mean_recovery_notifications=float(d.recovery_notifications.mean()) if d.recovery_notifications.notna().any() else np.nan)
        r['detection_ci_low'], r['detection_ci_high'] = wilson(int(d.detected.sum()), n)
        se = float(d.restricted_delay.std(ddof=1)/np.sqrt(n)) if n > 1 else np.nan
        horizon = 3 if r['shape'] == 'temporary' else 30
        r['restricted_delay_ci_low'] = max(0., r['mean_restricted_delay']-1.96*se)
        r['restricted_delay_ci_high'] = min(horizon, r['mean_restricted_delay']+1.96*se)
        rows.append(r)
    return pd.DataFrame(rows)


def make_figures(summary, null, selection, trials, examples, dates):
    folder = MON / 'figures'
    def save(name):
        plt.tight_layout()
        plt.savefig(folder / f'{PREFIX}_{name}.png', dpi=170, bbox_inches='tight')
        plt.close()
    # Every test result is shown, including methods that fail the budget.
    test = summary[summary.stage.eq('test') & summary.signal.eq('absolute_crps')]
    matrix = test.pivot_table(index='method', columns='scenario', values='detection_rate').reindex(METHODS)
    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
    ax.set(xticks=np.arange(len(matrix.columns)), xticklabels=matrix.columns,
           yticks=np.arange(len(matrix.index)), yticklabels=matrix.index,
           title='Experiment 15b — held-out synthetic CRPS detection (equal fold weighting)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, f'{matrix.iloc[i,j]:.0%}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, label='Notification during known event window')
    save('detection_heatmap')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n = null[null.stage.eq('test') & null.signal.eq('absolute_crps')]
    for method in METHODS:
        d = n[n.method.eq(method)]
        axes[0].plot(d.fold, d.notifications_per_100_days, marker='o', label=method)
    axes[0].axhline(float(null.budget.iloc[0]), color='black', ls='--', label='Budget')
    axes[0].set(title='Untouched empirical-null trials', xlabel='Chronological fold',
                ylabel='Notifications / 100 days', xticks=[1, 2, 3])
    axes[0].legend(fontsize=7)
    for method in METHODS:
        s = test[test.method.eq(method) & test.family.eq('bias')]
        d = n[n.method.eq(method)]
        axes[1].scatter(d.notifications_per_100_days.mean(), s.detection_rate.mean(), s=70, label=method)
    axes[1].axvline(float(null.budget.iloc[0]), color='black', ls='--')
    axes[1].set(title='CRPS bias scenarios: held-out trade-off', xlabel='Baseline notifications / 100 days',
                ylabel='Mean detection probability', ylim=(-.03, 1.03))
    axes[1].legend(fontsize=7)
    save('false_alert_tradeoff')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, signal, family in zip(axes, ['undercoverage', 'overcoverage'], ['narrow', 'wide']):
        s = summary[summary.stage.eq('test') & summary.signal.eq(signal) & summary.family.eq(family)]
        a = s.groupby('method').detection_rate.mean().reindex(METHODS)
        ax.bar(np.arange(len(a)), a, color='#2878a0')
        ax.set(xticks=np.arange(len(a)), xticklabels=a.index, ylim=(0, 1.08),
               title=f'{signal}: {family} intervals', ylabel='Mean detection probability')
        plt.setp(ax.get_xticklabels(), rotation=40, ha='right', fontsize=8)
    save('calibration_detection')
    e = pd.DataFrame(examples)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, col, label in zip(axes, ['absolute_crps', 'coverage', 'width80'],
                              ['Daily approximate CRPS', 'PI80 coverage', 'PI80 width']):
        for name, d in e.groupby('scenario', sort=False):
            ax.plot(d.day, d[col], label=name, lw=1, alpha=.85)
        ax.axvline(60, color='black', ls='--')
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[1].axhline(.8, color='gray', ls=':')
    axes[0].legend(fontsize=8)
    axes[0].set_title('Paired forecast-level examples: same actuals and sampled days')
    axes[-1].set_xlabel('Simulated day (injection starts at 60)')
    save('coherent_forecast_examples')
    r = trials[trials.stage.eq('test') & trials.signal.eq('absolute_crps') & trials.family.eq('bias')]
    fig, ax = plt.subplots(figsize=(10, 5))
    a = r.groupby(['method', 'volatility_band']).detected.mean().unstack().reindex(METHODS)
    a.plot.bar(ax=ax, color=['#2878a0', '#d79243'])
    ax.set(title='CRPS bias detection by realized event-window volatility', ylabel='Detection probability', ylim=(0, 1.08))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    save('volatility_robustness')


def markdown_table(frame, columns=None):
    if columns is not None:
        frame = frame[columns]
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return f'{x:.3f}' if np.isfinite(x) else '—'
        return str(x)
    lines = ['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']
    lines += ['| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(lines)


def write_report(a, splits, selection, summary, null, historical, checks, elapsed):
    test = summary[summary.stage.eq('test') & summary.signal.eq('absolute_crps') & summary.family.eq('bias')]
    comparison = test.groupby('method', sort=False).agg(
        detection_rate=('detection_rate', 'mean'), detection7=('detection7', 'mean'),
        restricted_delay=('mean_restricted_delay', 'mean')).reset_index()
    n = null[null.stage.eq('test') & null.signal.eq('absolute_crps')].groupby('method').agg(
        null_notifications_per_100=('notifications_per_100_days', 'mean'),
        test_folds_meeting_budget=('budget_met', 'sum')).reset_index()
    comparison = comparison.merge(n, on='method')
    compliant = comparison[comparison.test_folds_meeting_budget.eq(3)]
    if len(compliant):
        findings = '; '.join(f"{r.method}: {r.detection_rate:.1%} mean bias-event detection"
                             for r in compliant.itertuples())
        finding_text = f'CRPS methods meeting the null-notification budget in all three test folds: **{findings}**.'
    else:
        finding_text = 'No CRPS method met the null-notification budget in all three test folds.'
    latest = selection[selection.fold.eq(3) & selection.signal.eq('absolute_crps')].iloc[0]
    if latest.method != 'NONE':
        finding_text += (f" In the final fold, validation selected **{latest.method}**: test detection "
                         f"{latest.test_detection:.1%}, baseline notifications "
                         f"{latest.test_null_notifications_per_100:.2f}/100 days "
                         f"(budget {'met' if latest.test_budget_met else 'exceeded'}).")
    finding_text += ' Selection varies with period and signal; these results do not establish a universal winner.'
    selection.to_csv(MON / 'tables' / f'{PREFIX}_method_selection.csv', index=False)
    comparison.to_csv(MON / 'tables' / f'{PREFIX}_crps_method_comparison.csv', index=False)
    chosen = selection[selection.method.ne('NONE')]
    counts = chosen.groupby(['signal', 'method']).size().rename('folds_selected').reset_index()
    text = f'''# Experiment 15b — Forecast-level detector selection

## Aim in simple terms
Deliberately change saved hourly price forecasts, then ask: **which detector notices the problem promptly without sending too many unnecessary alerts?** We test incorrect forecast levels, slow drift, short disruptions and intervals that become too narrow or too wide.

## Main finding
{finding_text}

## What was run
- Original native Test forecasts only, with all 101 saved quantiles and complete Italian calendar days, including DST days. Actual prices and the benchmark are never changed.
- Three expanding chronological folds: fitting, threshold calibration, method selection (validation), then held-out testing. Test windows do not overlap. Later folds may use earlier dates as past information.
- {a.repetitions} paired trials per scenario, fold and validation/test stage; {a.calibration_repetitions} separate calibration trials. Moving blocks contain {a.block_days} complete days. The same sampled actuals and quantile curves are shared across methods and signals.
- Budget: **{a.budget:g} notification per 100 empirical-null days, per signal**. Notifications are at least {a.cooldown} days apart; continuing alarms produce another notification after that gap. State and cooldown continue across the known injection onset.
- Fixed P95 is an untuned comparator. Tuned quantile is its budget-calibrated counterpart. CUSUM uses k=0.5, Page-Hinkley delta=0.15 and EWMA alpha=0.2 on reference-standardized signals; only thresholds are calibrated. ADWIN is included when River is installed, with explicit upward direction confirmation and clock=1 so it checks daily (River's default clock=32 would check only every 32 days).
- Thresholds are selected on **no-injection calibration trials only**, taking the most sensitive grid value satisfying the notification budget. A method must also meet the independent validation-null budget to be eligible for selection.
- For each signal, validation chooses the method with highest average event detection, then lowest restricted delay, then fewer null notifications. That choice is frozen before test evaluation. All methods are still reported on the test set.

## Chronological splits
{markdown_table(splits)}

## Controlled changes
- Sudden forecast bias in both directions: 0.5, 1 and 2 times the fitting-period median PI80 half-width.
- Gradual bias in both directions: ramps to one half-width over 35 days.
- Temporary bias in both directions: one half-width for three days, then forecasts return exactly to their untouched counterparts.
- Interval contraction: width multipliers 0.8 and 0.5; expansion: 1.25 and 1.75, around the unchanged median.
- Bias or positive width scaling preserves quantile ordering. Approximate CRPS, coverage and width are recalculated from the transformed hourly curves **before** paired daily-block sampling. Gradual-ramp metrics use the appropriate daily ramp strength.
- Signals: absolute CRPS; CRPS / benchmark CRPS minus one; undercoverage max(0,0.8−coverage); overcoverage max(0,coverage−0.8); and CRPS residual after a fitting-only regression on log(1+realized daily volatility).
- The benchmark matches Experiment 13: point LEAR plus fixed validation-residual quantiles from before the native Test period. No held-out residuals are fitted.

## Main held-out CRPS comparison
Numbers below equally weight folds and the ten bias scenarios. Temporary-event detection uses a three-day window; persistent/gradual detection uses 30 days. Restricted delay assigns misses the corresponding window length; conditional median delays are in the detailed tables.

{markdown_table(comparison)}

**Read the budget column before comparing detection rates.** High detection from an over-budget method does not make it eligible. The table is descriptive, not a test-set optimization rule.

## Choices made on validation, evaluated on test
{markdown_table(selection)}

Selection frequency (a shortlist, not a universal winner):

{markdown_table(counts) if len(counts) else 'No method met the calibration and validation budgets.'}

For CRPS-based signals, selection averages bias scenarios; undercoverage selects on interval-contraction scenarios and overcoverage on expansion scenarios. This declared equal weighting is a research choice, not an operational cost function. `NONE` means no method passed the budget; do not force a winner. A frozen selection that exceeds the test-null budget is not operationally validated.

## Interpretation and limitations
- No historical dates have independently verified stable/failure labels. Therefore these are **empirical-null alerts and controlled perturbation detections**, not real-world false-positive or failure-detection estimates. Unchanged baseline periods may contain real changes.
- Budget matching is an empirical constraint, not exact equality or a statistical guarantee. Calibration/validation/test achieved rates and intervals are exported. Fixed P95 can fail the budget and is then excluded from selection.
- The budget is **per signal**. Combining five monitors needs separate joint notification calibration; it cannot inherit this budget.
- All trial detections count notifications, including chance coincidences. A paired untouched trial is evaluated on the identical sampled days. `paired_excess_detection` quantifies detection above these coincident baseline alerts.
- Median delay is conditional on detection. Misses, detection within 7/14/30 days and restricted delay are reported separately. For three-day events, all horizon columns are capped at the event's duration.
- Widening may improve undercoverage while reducing sharpness; additive bias can partly correct existing model bias. Achieved CRPS/coverage/width changes and fractions with worse CRPS are exported. Scenario names describe interventions, not guaranteed overall model damage.
- The 95% Wilson detection intervals and trial-based rate/delay intervals describe conditional Monte Carlo variation. Resampled days repeat a limited historical pool; trials are not independent real market failure episodes. No claim of statistical superiority is made from overlapping intervals.
- Seven-day blocks preserve some within-block dependence, but joins disrupt longer dependence and calendar structure. Fold and volatility summaries probe robustness; they do not reproduce every seasonal regime. No source history is removed as an inconvenient null period.
- All scores depend on observed prices. This is **after-outcome health monitoring**, not a forecast-time early-warning experiment. The same-day volatility residual is explanatory adjustment, not a prospective input.
- Upward-confirmed ADWIN is a modified comparator: a drift alert is accepted only if the latest seven-day mean exceeds both the fitting mean and the preceding seven-day mean. This gate can miss late changes and does not prove causal direction.
- This experiment evaluates predeclared parameter grids, not every possible tuning of each algorithm. Each trial initializes detector state from the final 30 fitting days; it does not estimate performance for every possible deployment initialization.

## Files
- `tables/{PREFIX}_splits.csv`, `_input_daily.csv`, `_calibration.csv`
- `tables/{PREFIX}_null_trials.csv`, `_null_summary.csv`
- `tables/{PREFIX}_event_trials.csv.gz` (compressed detailed trials), `_event_summary.csv`
- `tables/{PREFIX}_scenario_effects.csv`, `_volatility_summary.csv`
- `tables/{PREFIX}_method_selection.csv`, `_crps_method_comparison.csv`
- `tables/{PREFIX}_historical_alerts_unvalidated.csv`, `_example_daily_paths.csv`
- `figures/{PREFIX}_detection_heatmap.png`, `_false_alert_tradeoff.png`, `_calibration_detection.png`, `_coherent_forecast_examples.png`, `_volatility_robustness.png`
- `reports/{PREFIX}_run_metadata.json`

Checks: {', '.join(checks)}. Runtime: {elapsed:.1f} seconds.
'''
    (MON / 'reports' / f'{PREFIX}_report.md').write_text(text)
    return comparison


def self_checks():
    # Always-on alarms must not be hidden as one long episode.
    assert notifications(np.ones((1, 100), bool), 7).sum() == 15
    notice = np.zeros((2, 100), bool)
    notice[0, 62] = True
    s = score_notifications(notice, 60, 'sudden', 7)
    assert s['detected'].tolist() == [True, False]
    assert s['delay'][0] == 2 and np.isnan(s['delay'][1])
    assert s['restricted_delay'].tolist() == [2., 30.]
    # No recovery alert may count as detecting a three-day event.
    notice[1, 63] = True
    assert score_notifications(notice, 60, 'temporary', 7)['detected'].tolist() == [True, False]
    idx = block_indices(60, 150, 10, 7, np.random.default_rng(1))
    assert idx.shape == (10, 150) and idx.min() >= 0 and idx.max() < 60
    for fold in make_folds(470):
        assert fold['fit'][1] == fold['calibration'][0]
        assert fold['calibration'][1] == fold['validation'][0]
        assert fold['validation'][1] == fold['test'][0]
    print('Self-checks passed: notification budget, misses, delay, transient window, blocks, chronological splits.', flush=True)


def main():
    a = arguments()
    self_checks()
    if a.self_check:
        return
    started = time.monotonic()
    for folder in ('tables', 'figures', 'reports'):
        (MON / folder).mkdir(parents=True, exist_ok=True)
    data = load_hourly()
    baseline = daily_metrics(data)
    identity = daily_metrics(data, 0., 1.)
    assert np.allclose(identity, baseline)
    assert np.allclose(daily_metrics(data, 0., .5)[:, 6], baseline[:, 6]*.5)
    assert np.allclose(daily_metrics(data, 0., 1.75)[:, 6], baseline[:, 6]*1.75)
    assert np.all(daily_metrics(data, 0., .5)[:, 5] <= baseline[:, 5]+1e-12)
    assert np.all(daily_metrics(data, 0., 1.75)[:, 5] >= baseline[:, 5]-1e-12)
    daily = pd.DataFrame(baseline, columns=METRICS)
    daily.insert(0, 'date', data['dates'])
    daily['hours'] = data['counts']
    daily.drop(columns='volatility_adjusted_crps').to_csv(MON/'tables'/f'{PREFIX}_input_daily.csv', index=False)
    folds = make_folds(len(baseline))
    specs = scenario_specs()
    split_rows, calibration_rows, null_rows, event_frames = [], [], [], []
    effect_rows, historical_rows, examples = [], [], []
    settings = {}
    onset, n = 60, 150
    # One metric computation per transformation, shared across folds/trials.
    metric_cache = {(0., 1.): baseline}
    for fold in folds:
        f = fold['fold']
        print(f'Fold {f}/3: fitting and budget calibration', flush=True)
        for part in ('fit', 'calibration', 'validation', 'test'):
            lo, hi = fold[part]
            split_rows.append(dict(fold=f, period=part, start=str(data['dates'][lo].date()),
                                   end=str(data['dates'][hi-1].date()), days=hi-lo))
        fit_lo, fit_hi = fold['fit']
        fit = baseline[fit_lo:fit_hi]
        coef = np.polyfit(np.log1p(fit[:, 7]), fit[:, 0], 1)
        def adjust(values):
            values = values.copy()
            values[:, 4] = values[:, 0]-np.polyval(coef, np.log1p(values[:, 7]))
            return values
        base = adjust(baseline)
        half_width = float(np.median(fit[:, 6])/2)
        vol_cut = float(np.median(fit[:, 7]))
        cal_lo, cal_hi = fold['calibration']
        cal_indices = block_indices(cal_hi-cal_lo, 200, a.calibration_repetitions, a.block_days,
                                    np.random.default_rng(np.random.SeedSequence([a.seed, f, 0])))
        for si, signal in enumerate(SIGNALS):
            for method in METHODS:
                cfg = fit_detector(base[:fit_hi, si], base[cal_lo:cal_hi, si], cal_indices,
                                   method, a.budget, a.cooldown)
                settings[f, signal, method] = cfg
                calibration_rows.append(dict(fold=f, signal=signal, method=method,
                    params=json.dumps(cfg['params']), fitting_mean=cfg['mu'], fitting_sd=cfg['sd'],
                    volatility_regression_slope=float(coef[0]), volatility_regression_intercept=float(coef[1]),
                    bias_halfwidth=half_width, volatility_cut=vol_cut,
                    calibration_notifications_per_100_days=cfg['calibration_rate'],
                    calibration_alarm_day_pct=cfg['calibration_alarm_day_pct'],
                    calibration_budget_met=cfg['calibration_budget_met'], budget=a.budget))
        for stage_no, stage in enumerate(('validation', 'test'), 1):
            lo, hi = fold[stage]
            idx = block_indices(hi-lo, n, a.repetitions, a.block_days,
                                np.random.default_rng(np.random.SeedSequence([a.seed, f, stage_no]))) + lo
            untouched = base[idx]
            null_notices = {}
            print(f'Fold {f}/3: {stage}, {a.repetitions} paired trials x {len(specs)-1} event scenarios', flush=True)
            for spec in specs:
                level = np.zeros(n)
                if spec['shape'] == 'sudden':
                    level[onset:] = 1.
                elif spec['shape'] == 'temporary':
                    level[onset:onset+3] = 1.
                elif spec['shape'] == 'gradual':
                    level[onset:] = np.minimum(1., np.arange(1, n-onset+1)/35)
                values = np.empty_like(untouched)
                for strength in np.unique(level):
                    key = (round(float(spec['bias']*half_width*strength), 10),
                           round(float(1+(spec['factor']-1)*strength), 10))
                    if key not in metric_cache:
                        metric_cache[key] = daily_metrics(data, *key)
                    transformed = adjust(metric_cache[key])
                    positions = level == strength
                    values[:, positions] = transformed[idx[:, positions]]
                # Check actual scenario effects on paired sampled days.
                window = 3 if spec['shape'] == 'temporary' else 30
                delta = (values[:, onset:onset+window]-untouched[:, onset:onset+window]).mean(axis=1)
                band = np.where(untouched[:, onset:onset+window, 7].mean(axis=1) > vol_cut, 'high', 'low')
                if spec['family'] != 'none':
                    effect_rows.append(dict(fold=f, stage=stage, scenario=spec['scenario'], family=spec['family'],
                        mean_crps_change=float(delta[:, 0].mean()), fraction_crps_worsened=float((delta[:, 0]>0).mean()),
                        mean_coverage_change_pp=float(delta[:, 5].mean()*100), mean_width_change=float(delta[:, 6].mean()),
                        bias_scale_halfwidth=half_width))
                if f == 3 and stage == 'test' and spec['scenario'] in (
                        'none', 'sudden_bias_positive_medium', 'narrow_interval_x0.5', 'wide_interval_x1.75'):
                    for t in range(n):
                        examples.append(dict(scenario=spec['scenario'], day=t, source_date=str(data['dates'][idx[0,t]].date()),
                                             **dict(zip(METRICS, values[0,t]))))
                for si, signal in enumerate(SIGNALS):
                    for method in METHODS:
                        cfg = settings[f, signal, method]
                        x = np.column_stack([np.broadcast_to(cfg['warmup'], (a.repetitions, len(cfg['warmup']))),
                                             (values[:, :, si]-cfg['mu'])/cfg['sd']])
                        raw = run_detector(x, method, cfg['params'], len(cfg['warmup']))
                        notice = notifications(raw, a.cooldown)
                        if spec['family'] == 'none':
                            null_notices[signal, method] = notice
                            for trial in range(a.repetitions):
                                null_rows.append(dict(fold=f, stage=stage, signal=signal, method=method,
                                    trial=trial, notifications=int(notice[trial].sum()),
                                    alarm_days=int(raw[trial].sum()), days=n))
                            continue
                        score = score_notifications(notice, onset, spec['shape'], a.cooldown)
                        null_score = score_notifications(null_notices[signal, method], onset, spec['shape'], a.cooldown)
                        frame = pd.DataFrame(dict(fold=f, stage=stage, signal=signal, method=method,
                            scenario=spec['scenario'], family=spec['family'], shape=spec['shape'],
                            trial=np.arange(a.repetitions), volatility_band=band,
                            paired_null_detected=null_score['detected'],
                            paired_excess_detection=score['detected'].astype(int)-null_score['detected'].astype(int),
                            recovery_excess_notifications=score['recovery_notifications']-null_score['recovery_notifications'],
                            **score))
                        event_frames.append(frame)
            # Unmodified chronological deployment, warm-started from fit+cal history.
            # These alerts are explicitly unvalidated and never called false positives.
            for si, signal in enumerate(SIGNALS):
                for method in METHODS:
                    cfg = settings[f, signal, method]
                    x = ((base[:hi, si]-cfg['mu'])/cfg['sd'])[None, :]
                    raw = run_detector(x, method, cfg['params'], 0)
                    notify = notifications(raw, a.cooldown)
                    for day in range(lo, hi):
                        historical_rows.append(dict(fold=f, stage=stage, signal=signal, method=method,
                            date=str(data['dates'][day].date()), alarm=bool(raw[0,day]), notification=bool(notify[0,day])))
        # Bias scales differ by fold, so keep baseline only to limit memory.
        metric_cache = {(0., 1.): baseline}

    trials = pd.concat(event_frames, ignore_index=True)
    summary = summarize_trials(trials)
    extra = trials.groupby(['fold','stage','signal','method','scenario'], sort=False).agg(
        paired_null_detection_rate=('paired_null_detected','mean'),
        paired_excess_detection=('paired_excess_detection','mean'),
        recovery_excess_notifications=('recovery_excess_notifications','mean')).reset_index()
    summary = summary.merge(extra, on=['fold','stage','signal','method','scenario'], validate='one_to_one')
    null_trials = pd.DataFrame(null_rows)
    null_summary = []
    for key, d in null_trials.groupby(['fold','stage','signal','method'], sort=False):
        rate, low, high = rate_interval(d.notifications, n)
        # If all trials have zero events, show a nonzero Poisson exposure bound
        # instead of suggesting zero uncertainty; still conditional on this pool.
        if d.notifications.sum() == 0:
            high = float(.5*chi2.ppf(.95, 2)*100/(len(d)*n))
        null_summary.append(dict(zip(['fold','stage','signal','method'], key)) | dict(
            trials=len(d), notifications_per_100_days=rate, rate_ci_low=low, rate_ci_high=high,
            alarm_day_pct=float(d.alarm_days.sum()*100/d.days.sum()),
            budget=a.budget, budget_met=rate <= a.budget+1e-12))
    null = pd.DataFrame(null_summary)
    calibration = pd.DataFrame(calibration_rows)
    selection_rows = []
    for f in range(1, 4):
        for signal in SIGNALS:
            family = 'narrow' if signal == 'undercoverage' else 'wide' if signal == 'overcoverage' else 'bias'
            val = summary[summary.fold.eq(f) & summary.stage.eq('validation') & summary.signal.eq(signal) & summary.family.eq(family)]
            rankings = val.groupby('method').agg(validation_detection=('detection_rate','mean'),
                validation_restricted_delay=('mean_restricted_delay','mean')).reset_index()
            nv = null[null.fold.eq(f) & null.stage.eq('validation') & null.signal.eq(signal)]
            rankings = rankings.merge(nv[['method','notifications_per_100_days','budget_met']], on='method')
            ca = calibration[calibration.fold.eq(f) & calibration.signal.eq(signal)]
            rankings = rankings.merge(ca[['method','calibration_budget_met']], on='method')
            ok = rankings[rankings.budget_met & rankings.calibration_budget_met].sort_values(
                ['validation_detection','validation_restricted_delay','notifications_per_100_days','method'],
                ascending=[False,True,True,True])
            row = dict(fold=f, signal=signal, method='NONE', validation_detection=np.nan,
                       test_detection=np.nan, test_null_notifications_per_100=np.nan, test_budget_met=False)
            if len(ok):
                choice = ok.iloc[0]
                ev = summary[summary.fold.eq(f) & summary.stage.eq('test') & summary.signal.eq(signal) &
                             summary.family.eq(family) & summary.method.eq(choice.method)]
                nt = null[null.fold.eq(f) & null.stage.eq('test') & null.signal.eq(signal) & null.method.eq(choice.method)].iloc[0]
                row.update(method=choice.method, validation_detection=float(choice.validation_detection),
                           test_detection=float(ev.detection_rate.mean()),
                           test_null_notifications_per_100=float(nt.notifications_per_100_days), test_budget_met=bool(nt.budget_met))
            selection_rows.append(row)
    selection = pd.DataFrame(selection_rows)
    volatility = trials.groupby(['fold','stage','signal','method','family','volatility_band']).agg(
        trials=('detected','size'), detection_rate=('detected','mean'),
        mean_restricted_delay=('restricted_delay','mean')).reset_index()
    splits, historical = pd.DataFrame(split_rows), pd.DataFrame(historical_rows)
    tables = dict(splits=splits, calibration=calibration, null_trials=null_trials, null_summary=null,
                  event_trials=trials, event_summary=summary, scenario_effects=pd.DataFrame(effect_rows),
                  volatility_summary=volatility, historical_alerts_unvalidated=historical,
                  example_daily_paths=pd.DataFrame(examples))
    for name, frame in tables.items():
        suffix = '.csv.gz' if name == 'event_trials' else '.csv'
        frame.to_csv(MON/'tables'/f'{PREFIX}_{name}{suffix}', index=False)
    checks = ['UTC/local-date alignment', 'complete DST-aware days', 'no quantile crossings',
              'benchmark actual alignment', 'identity transformation', 'width scaling',
              'coverage nesting', 'notification cooldown', 'miss-aware delay', 'chronological splits']
    make_figures(summary, null, selection, trials, examples, data['dates'])
    elapsed = time.monotonic()-started
    comparison = write_report(a, splits, selection, summary, null, historical, checks, elapsed)
    metadata = dict(experiment=PREFIX, aim='Compare prompt detection at a common empirical-null notification budget',
        settings=vars(a), input=str(data['path'].relative_to(ROOT)),
        input_sha256=hashlib.sha256(data['path'].read_bytes()).hexdigest(),
        benchmark_input='ENTSOE_LEAR_v3_predictions.csv',
        benchmark_sha256=hashlib.sha256((ROOT/'ENTSOE_LEAR_v3_predictions.csv').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_days=len(baseline), input_hours=len(data['y']), quantiles=len(data['tau']),
        local_day_hour_counts={str(k):int(v) for k,v in pd.Series(data['counts']).value_counts().items()},
        methods=METHODS, signals=SIGNALS, event_scenarios=len(specs)-1,
        event_trial_rows=len(trials), checks=checks, elapsed_seconds=elapsed,
        interpretation='Conditional forecast-perturbation research; historical failure labels unavailable',
        versions=dict(numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__,
                      river=importlib.metadata.version('river') if ADWIN is not None else None))
    (MON/'reports'/f'{PREFIX}_run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('\nHeld-out CRPS comparison (read budget compliance alongside detection):', flush=True)
    print(comparison.to_string(index=False), flush=True)
    print(f'\nFinished in {elapsed:.1f}s. Report: {MON / "reports" / f"{PREFIX}_report.md"}', flush=True)


if __name__ == '__main__':
    main()
