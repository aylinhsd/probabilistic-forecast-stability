#!/usr/bin/env python3
"""Experiment 13: native probabilistic LEAR v3 versus existing benchmarks.

Place this script in monitoring_research/scripts/ under 'Entsoe dataRev'.
Inputs in the 'Entsoe dataRev' root:
  ENTSOE_Probabilistic_LEAR_v3_predictions.csv
  ENTSOE_LEAR_v3_predictions.csv
  ENTSOE_seasonal_naive_predictions.csv
Outputs in monitoring_research/{tables,figures,reports}/.
Dependencies: pandas numpy matplotlib scipy
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
MON = ROOT / 'monitoring_research'
TABLES, FIGURES, REPORTS = [MON / s for s in ('tables', 'figures', 'reports')]
NATIVE = ROOT / 'ENTSOE_Probabilistic_LEAR_v3_predictions.csv'
POINT = ROOT / 'ENTSOE_LEAR_v3_predictions.csv'
NAIVE = ROOT / 'ENTSOE_seasonal_naive_predictions.csv'


def load(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Place the CSV in the Entsoe dataRev root folder.')
    df = pd.read_csv(path, low_memory=False)
    df['utc'] = pd.to_datetime(df['Target_UTC'], utc=True, errors='raise')
    if df['utc'].isna().any() or df['utc'].duplicated().any():
        raise ValueError(f'Missing or duplicate UTC timestamps in {path.name}')
    return df


def approx_crps(y, q, tau):
    """2 * integrated quantile loss on the common finite quantile grid."""
    err = y[:, None] - q
    loss = np.maximum(tau[None, :] * err, (tau[None, :] - 1) * err)
    return 2 * np.trapezoid(loss, x=tau, axis=1)


def figure(path):
    plt.tight_layout()
    plt.savefig(FIGURES / path, dpi=190, bbox_inches='tight')
    plt.savefig((FIGURES / path).with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


def numerical_comparisons(groups, overall):
    """Plot existing pooled metrics without changing the evaluation procedure."""
    labels = groups['group'].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title, reference in [
        (axes[0], 'rMAE_native_vs_naive', 'Native / seasonal naive rMAE (lower is better)', 1),
        (axes[1], 'CRPSS_native_vs_residual', 'CRPS skill vs fixed residual (higher is better)', 0),
    ]:
        bars = ax.bar(labels, groups[metric], color=['#79b4ce', '#438ab0', '#235777'])
        ax.bar_label(bars, fmt='%.3f', padding=5)
        ax.axhline(reference, color='0.3', linestyle='--', label='Equal performance')
        ax.set(title=title, xlabel='Realized volatility group')
        ax.margins(y=.25)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(loc='best')
    fig.suptitle('Experiment 13: relative performance by volatility group\nRatios of pooled mean errors')
    figure('experiment_13_relative_performance_by_volatility_group.png')

    comparison = pd.concat([pd.DataFrame([dict(overall, group='Overall')]), groups], ignore_index=True)
    positions = np.arange(len(comparison))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    specifications = [
        ('MAE (€/MWh) — lower is better', [('MAE_native', 'Native'), ('MAE_naive', 'Seasonal naive')], 1),
        ('Approx. CRPS (€/MWh) — lower is better', [('CRPS_native', 'Native'), ('CRPS_residual', 'Fixed residual')], 1),
        ('PI80 coverage (%) — target 80%', [('PI80_native_coverage', 'Native'), ('PI80_residual_coverage', 'Fixed residual')], 100),
    ]
    for ax, (title, series, scale) in zip(axes, specifications):
        for i, (metric, label) in enumerate(series):
            bars = ax.bar(positions + (i - .5) * .36, comparison[metric] * scale,
                          width=.36, label=label, color=['#2878a0', '#d79243'][i])
            ax.bar_label(bars, fmt='%.2f', padding=4, fontsize=8)
        if scale == 100:
            ax.axhline(80, color='0.3', linestyle='--', linewidth=1, label='Nominal 80%')
            ax.set_ylim(0, 110)
        else:
            ax.margins(y=.2)
        ax.set(title=title, xticks=positions, xticklabels=comparison['group'])
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(loc='upper left', fontsize=8)
    fig.suptitle('Experiment 13: numerical comparison of MAE, CRPS and coverage\nPooled hourly results overall and within volatility groups')
    figure('experiment_13_numerical_metric_comparison.png')


def main():
    for folder in (TABLES, FIGURES, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)
    native = load(NATIVE)
    point = load(POINT)
    naive = load(NAIVE)
    native = native.loc[native['Split'].eq('Test')].copy()
    val = point.loc[point['Split'].eq('Validation')].copy()
    point_test = point.loc[point['Split'].eq('Test')].copy()
    if val.empty or native.empty or point_test.empty:
        raise ValueError('Native test, point test, or point validation rows missing.')

    qcols = sorted([c for c in native.columns if c.startswith('Q_')], key=lambda c: float(c[2:]))
    tau = np.array([float(c[2:]) for c in qcols])
    if 'Q_0.500' not in qcols or 'Q_0.100' not in qcols or 'Q_0.900' not in qcols:
        raise ValueError('Native forecast lacks required quantile columns.')
    if not np.all(np.diff(tau) > 0):
        raise ValueError('Quantile levels are not increasing.')

    # Calibration uses ONLY the point model's validation period (Jan–Jun 2025).
    # The test period begins July 2025. No test residuals are used for calibration.
    val_y = pd.to_numeric(val['Target_Price_EUR_MWh'], errors='raise').to_numpy(float)
    val_pred = pd.to_numeric(val['LEAR_v3_Prediction_EUR_MWh'], errors='raise').to_numpy(float)
    residual = val_y - val_pred
    if not np.isfinite(residual).all():
        raise ValueError('Non-finite validation residuals.')
    # Empirical quantiles: one fixed additive offset per probability level.
    residual_quantiles = np.quantile(residual, tau)

    # IMPORTANT: naive uses older split definitions; do NOT filter naive to Test.
    # Match timestamps first, then evaluate on the native Test dates.
    p = point_test[['utc', 'LEAR_v3_Prediction_EUR_MWh', 'Target_Price_EUR_MWh']].rename(
        columns={'Target_Price_EUR_MWh': 'point_actual'})
    s = naive[['utc', 'Predicted_Price_EUR_MWh', 'Target_Price_EUR_MWh']].rename(
        columns={'Target_Price_EUR_MWh': 'naive_actual'})
    d = native.merge(p, on='utc', how='left', validate='one_to_one')
    d = d.merge(s, on='utc', how='left', validate='one_to_one')
    for col in ['LEAR_v3_Prediction_EUR_MWh', 'Predicted_Price_EUR_MWh', 'point_actual', 'naive_actual']:
        if d[col].isna().any():
            raise ValueError(f'{d[col].isna().sum()} native test hours lack aligned {col}. Check file coverage.')
    y = pd.to_numeric(d['Target_Price_EUR_MWh'], errors='raise').to_numpy(float)
    if not np.allclose(y, d['point_actual'].to_numpy(float), atol=1e-5) or not np.allclose(
            y, d['naive_actual'].to_numpy(float), atol=1e-5):
        raise ValueError('Actual prices disagree across input files at matched UTC timestamps.')

    native_q = d[qcols].to_numpy(float)
    point_forecast = d['LEAR_v3_Prediction_EUR_MWh'].to_numpy(float)
    naive_forecast = d['Predicted_Price_EUR_MWh'].to_numpy(float)
    if not np.isfinite(native_q).all() or not np.isfinite(point_forecast).all() or not np.isfinite(naive_forecast).all():
        raise ValueError('Nonfinite forecasts in aligned test data.')
    if (np.diff(native_q, axis=1) < -1e-8).any():
        raise ValueError('Native quantile crossings found.')
    residual_q = point_forecast[:, None] + residual_quantiles[None, :]
    d['CRPS_native'] = approx_crps(y, native_q, tau)
    d['CRPS_residual'] = approx_crps(y, residual_q, tau)
    d['AE_native'] = np.abs(y - d['Q_0.500'].to_numpy(float))
    d['AE_point_LEAR'] = np.abs(y - point_forecast)
    d['AE_naive'] = np.abs(y - naive_forecast)
    d['PI80_native_width'] = d['Q_0.900'].to_numpy(float) - d['Q_0.100'].to_numpy(float)
    d['PI80_residual_width'] = residual_quantiles[np.where(np.isclose(tau, .9))[0][0]] - residual_quantiles[np.where(np.isclose(tau, .1))[0][0]]
    d['PI80_native_covered'] = ((y >= d['Q_0.100'].to_numpy(float)) & (y <= d['Q_0.900'].to_numpy(float))).astype(float)
    d['PI80_residual_covered'] = ((y >= residual_q[:, np.where(np.isclose(tau, .1))[0][0]]) &
                                  (y <= residual_q[:, np.where(np.isclose(tau, .9))[0][0]])).astype(float)
    d['date'] = d['utc'].dt.tz_convert('Europe/Rome').dt.date.astype(str)
    daily = d.groupby('date', sort=True).agg(
        hours=('utc', 'size'), volatility=('Target_Price_EUR_MWh', 'std'),
        MAE_native=('AE_native', 'mean'), MAE_point_LEAR=('AE_point_LEAR', 'mean'),
        MAE_naive=('AE_naive', 'mean'), CRPS_native=('CRPS_native', 'mean'),
        CRPS_residual=('CRPS_residual', 'mean'),
        PI80_native_coverage=('PI80_native_covered', 'mean'),
        PI80_residual_coverage=('PI80_residual_covered', 'mean'),
        PI80_native_width=('PI80_native_width', 'mean'),
        PI80_residual_width=('PI80_residual_width', 'mean')).reset_index()
    if (daily['MAE_naive'] <= 1e-10).any() or (daily['CRPS_residual'] <= 1e-10).any():
        raise ValueError('A daily benchmark error is zero; ratio requires special handling.')
    daily['rMAE_native_vs_naive'] = daily['MAE_native'] / daily['MAE_naive']
    daily['rMAE_point_vs_naive'] = daily['MAE_point_LEAR'] / daily['MAE_naive']
    daily['CRPSS_native_vs_residual'] = 1 - daily['CRPS_native'] / daily['CRPS_residual']
    daily['volatility_group'] = pd.qcut(daily['volatility'], q=3, labels=['Low', 'Medium', 'High'])
    daily.to_csv(TABLES / 'experiment_13_daily_metrics.csv', index=False)

    # Aggregate skill is a RATIO OF AGGREGATE ERRORS, not the mean of daily ratios.
    overall = {
        'hours': len(d), 'days': len(daily), 'MAE_native': d['AE_native'].mean(),
        'MAE_point_LEAR': d['AE_point_LEAR'].mean(), 'MAE_naive': d['AE_naive'].mean(),
        'CRPS_native': d['CRPS_native'].mean(), 'CRPS_residual': d['CRPS_residual'].mean(),
        'PI80_native_coverage': d['PI80_native_covered'].mean(),
        'PI80_residual_coverage': d['PI80_residual_covered'].mean(),
        'PI80_native_width': d['PI80_native_width'].mean(),
        'PI80_residual_width': d['PI80_residual_width'].mean(),
    }
    overall['rMAE_native_vs_naive'] = overall['MAE_native'] / overall['MAE_naive']
    overall['rMAE_point_vs_naive'] = overall['MAE_point_LEAR'] / overall['MAE_naive']
    overall['CRPSS_native_vs_residual'] = 1 - overall['CRPS_native'] / overall['CRPS_residual']
    pd.DataFrame([overall]).to_csv(TABLES / 'experiment_13_overall_summary.csv', index=False)

    rows = []
    for group, subset in d.merge(daily[['date', 'volatility_group']], on='date').groupby('volatility_group', observed=True):
        row = {'group': str(group), 'hours': len(subset), 'days': subset['date'].nunique(),
               'MAE_native': subset['AE_native'].mean(), 'MAE_naive': subset['AE_naive'].mean(),
               'CRPS_native': subset['CRPS_native'].mean(), 'CRPS_residual': subset['CRPS_residual'].mean(),
               'PI80_native_coverage': subset['PI80_native_covered'].mean(),
               'PI80_residual_coverage': subset['PI80_residual_covered'].mean(),
               'PI80_native_width': subset['PI80_native_width'].mean(),
               'PI80_residual_width': subset['PI80_residual_width'].mean()}
        row['rMAE_native_vs_naive'] = row['MAE_native'] / row['MAE_naive']
        row['CRPSS_native_vs_residual'] = 1 - row['CRPS_native'] / row['CRPS_residual']
        rows.append(row)
    groups = pd.DataFrame(rows)
    groups.to_csv(TABLES / 'experiment_13_volatility_group_summary.csv', index=False)
    numerical_comparisons(groups, overall)

    x = pd.to_datetime(daily['date'])
    for left, right, ylabel, filename, title in [
        ('MAE_native', 'MAE_naive', 'MAE (€/MWh)', 'experiment_13_daily_mae.png', 'Daily MAE: native vs seasonal naive'),
        ('CRPS_native', 'CRPS_residual', 'Approx. CRPS (€/MWh)', 'experiment_13_daily_crps.png', 'Daily CRPS: native vs fixed residual')]:
        plt.figure(figsize=(12, 4.5))
        plt.plot(x, daily[left], label=left, alpha=.85, linewidth=1)
        plt.plot(x, daily[right], label=right, alpha=.75, linewidth=1)
        plt.title(title); plt.ylabel(ylabel); plt.xlabel('Italian calendar date'); plt.legend(); plt.grid(alpha=.2)
        figure(filename)
    for col, ylabel, filename, zero in [
        ('rMAE_native_vs_naive', 'Daily rMAE', 'experiment_13_daily_rmae.png', 1),
        ('CRPSS_native_vs_residual', 'Daily CRPS skill score', 'experiment_13_daily_crpss.png', 0)]:
        plt.figure(figsize=(12, 4.5))
        plt.plot(x, daily[col], linewidth=1)
        plt.axhline(zero, color='black', linestyle='--', linewidth=1, label='Equal performance')
        plt.title(col.replace('_', ' ')); plt.ylabel(ylabel); plt.xlabel('Italian calendar date')
        plt.legend(); plt.grid(alpha=.2)
        figure(filename)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(daily['volatility'], daily['rMAE_native_vs_naive'], s=12, alpha=.45)
    axes[0].axhline(1, color='black', linestyle='--', linewidth=1)
    axes[0].set(xlabel='Observed daily price SD (€/MWh)', ylabel='Daily rMAE', title='Volatility vs rMAE')
    axes[1].scatter(daily['volatility'], daily['CRPSS_native_vs_residual'], s=12, alpha=.45)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set(xlabel='Observed daily price SD (€/MWh)', ylabel='Daily CRPSS', title='Volatility vs CRPSS')
    for ax, metric in zip(axes, ['rMAE_native_vs_naive', 'CRPSS_native_vs_residual']):
        vx = daily['volatility'].to_numpy()
        vy = daily[metric].to_numpy()
        slope, intercept = np.polyfit(vx, vy, 1)
        endpoints = np.array([vx.min(), vx.max()])
        ax.plot(endpoints, slope * endpoints + intercept, color='#ba512b', linewidth=2,
                label=f'Linear fit: y = {slope:.4f}x {intercept:+.3f}')
        ax.text(.97, .97,
                f'Pearson r = {pearsonr(vx, vy).statistic:+.3f}\n'
                f'Spearman ρ = {spearmanr(vx, vy).statistic:+.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=.85, edgecolor='0.85'))
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle(f'Experiment 13: volatility and daily relative performance ({len(daily)} days)')
    figure('experiment_13_volatility_relative_performance.png')

    report = f'''# Experiment 13 — Absolute versus benchmark-relative performance

## Research question
Does native Probabilistic LEAR v3 maintain its advantage over existing benchmarks during difficult market conditions?

## Inputs and alignment
- Native probabilistic LEAR v3: test split, {len(d):,} matched hourly timestamps, {len(daily)} Italian calendar dates.
- Seasonal naive: reused its previously generated point forecasts; joined on `Target_UTC`, **not** on its original split labels.
- Residual-based LEAR v3: reconstructed from original LEAR v3 point forecasts plus **fixed empirical quantiles of its validation residuals** (no test residuals used).
- Evaluation window: {daily['date'].iloc[0]} to {daily['date'].iloc[-1]}.
- Validation residual count: {len(residual)}.
- All compared observations share the same realized price, checked numerically.

## Methods
- Native point forecast: median `Q_0.500`; seasonal-naive and original point LEAR use their saved point predictions.
- Approximate CRPS: twice the trapezoidal integral of quantile/pinball loss over the common saved quantile grid ({tau.min():.3f}–{tau.max():.3f}); tails outside this grid are omitted for both models.
- Daily rMAE = daily native MAE / daily naive MAE. Daily CRPSS = 1 − daily native CRPS / daily residual CRPS.
- **Overall** and group ratios use ratios of pooled mean errors, not averages of daily ratios.
- PI80 coverage uses inclusive 10th/90th percentile bounds; width is upper minus lower.
- Volatility = sample SD of realized hourly prices per Italian calendar date. Groups are equal-frequency tertiles of daily volatility.

## Results — pooled test period
| Indicator | Result |
|---|---:|
| Native MAE | {overall['MAE_native']:.4f} €/MWh |
| Original point LEAR MAE | {overall['MAE_point_LEAR']:.4f} €/MWh |
| Seasonal naive MAE | {overall['MAE_naive']:.4f} €/MWh |
| Native / naive rMAE | {overall['rMAE_native_vs_naive']:.4f} |
| Native approximate CRPS | {overall['CRPS_native']:.4f} €/MWh |
| Fixed-residual approximate CRPS | {overall['CRPS_residual']:.4f} €/MWh |
| Native / residual CRPSS | {overall['CRPSS_native_vs_residual']:.4f} |
| Native PI80 coverage | {overall['PI80_native_coverage']:.2%} |
| Residual PI80 coverage | {overall['PI80_residual_coverage']:.2%} |
| Native PI80 width | {overall['PI80_native_width']:.4f} €/MWh |
| Residual PI80 width | {overall['PI80_residual_width']:.4f} €/MWh |

## Interpretation guidance
- rMAE below 1 means lower absolute error than seasonal naive; above 1 means worse.
- CRPSS above 0 means lower approximate CRPS than fixed-residual LEAR; below 0 means worse.
- Relative advantage is not proof of model health; absolute errors and coverage must also be monitored.
- Examine daily and volatility-group outputs before making claims about stability or deterioration.

## Numerical graphs
- [Relative performance by volatility group](../figures/experiment_13_relative_performance_by_volatility_group.png): labeled pooled rMAE and CRPSS, with equal-performance reference lines.
- [MAE, CRPS and coverage comparison](../figures/experiment_13_numerical_metric_comparison.png): labeled overall and volatility-group results for native forecasts and the corresponding benchmarks.
- [Updated scatterplots](../figures/experiment_13_volatility_relative_performance.png): daily rMAE and CRPSS against volatility, with straight least-squares trend lines, equations, Pearson r and Spearman rho.
- Each graph is also saved as a PDF with the same filename stem.
- Trend lines and correlations are descriptive; daily ratios are not pooled group skill scores.

## Limitations
- Residual calibration uses the original point model's validation period, which was also used during native model selection; this is an existing-method comparison, not a fully independent prospective benchmark study.
- The fixed-residual comparator is reconstructed from saved validation residuals; verify its width against the previously reported ~31.91 €/MWh. Differences may indicate different calibration conventions or files.
- The seasonal naive uses historical predictions from a differently partitioned experiment. Timestamp and target checks are essential.
- Volatility uses realized same-day prices, so it is **retrospective**, not available at forecast issue time.
- Daily ratios can be unstable when denominator errors are small; examine pooled ratios for main conclusions.
- CRPS is a finite-quantile approximation; do not compare directly with results calculated using a different integration convention.
- Correlations and volatility strata do not establish causation or detect confirmed model failure.
'''
    (REPORTS / 'experiment_13_report.md').write_text(report, encoding='utf-8')
    print('\nEXPERIMENT 13 — BENCHMARK-RELATIVE PERFORMANCE')
    print('Inputs:', NATIVE.name, POINT.name, NAIVE.name)
    print(f'Validation residuals: {len(residual):,}; matched test hours: {len(d):,}; dates: {len(daily)}')
    print(f'Test period: {daily.date.iloc[0]} to {daily.date.iloc[-1]}')
    print(f'Fixed residual PI80 width: {overall["PI80_residual_width"]:.4f} €/MWh')
    print('\nOVERALL RESULTS')
    for key, value in overall.items():
        print(f'{key:28s}: {value:.4f}' if isinstance(value, (float, np.floating)) else f'{key:28s}: {value}')
    print('\nVOLATILITY GROUP SUMMARY')
    print(groups.round(4).to_string(index=False))
    print('\nTables:', TABLES, '\nFigures:', FIGURES, '\nReport:', REPORTS / 'experiment_13_report.md')


if __name__ == '__main__':
    main()
