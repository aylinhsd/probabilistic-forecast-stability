#!/usr/bin/env python3
"""Experiment 12: observed market volatility vs native probabilistic LEAR v3 performance.

Project layout:
  Entsoe dataRev/ENTSOE_Probabilistic_LEAR_v3_predictions.csv
  Entsoe dataRev/monitoring_research/scripts/experiment_12_volatility_vs_performance.py
  Entsoe dataRev/monitoring_research/{tables,figures,reports}/

Run: python3 monitoring_research/scripts/experiment_12_volatility_vs_performance.py
Dependencies: pandas numpy scipy matplotlib
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MON = ROOT / 'monitoring_research'
TABLES, FIGURES, REPORTS = (MON / name for name in ('tables', 'figures', 'reports'))
INPUT = ROOT / 'ENTSOE_Probabilistic_LEAR_v3_predictions.csv'


def plot_volatility_correlations(daily):
    """Compare all four examined metrics with realized daily volatility."""
    panels = [
        ('CRPS_approx', 'Approximate CRPS', '€/MWh', '#2878a0'),
        ('MAE', 'Median MAE', '€/MWh', '#7057a3'),
        ('PI80_width', '80% prediction interval width', '€/MWh', '#bd772c'),
        ('PI80_coverage', '80% prediction interval coverage', '%', '#268477'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    x = daily['volatility_sd'].to_numpy()
    for ax, (metric, title, unit, color) in zip(axes.flat, panels):
        y = daily[metric].to_numpy() * (100 if metric == 'PI80_coverage' else 1)
        r = pearsonr(x, y).statistic
        rho = spearmanr(x, y).statistic
        ax.scatter(x, y, s=22, alpha=0.45, color=color, edgecolors='none')
        slope, intercept = np.polyfit(x, y, 1)
        limits = np.array([x.min(), x.max()])
        ax.plot(limits, slope * limits + intercept, color=color, linewidth=2,
                label='Linear fit')
        if metric == 'PI80_coverage':
            ax.axhline(80, color='0.4', linestyle='--', linewidth=1,
                       label='Nominal 80%')
            ax.set_ylim(-2, 103)
        ax.text(0.97, 0.97, f'Pearson r = {r:+.3f}\nSpearman ρ = {rho:+.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9,
                          edgecolor='0.85'))
        ax.set(title=title, ylabel=f'{title} ({unit})')
        ax.grid(alpha=0.18)
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', frameon=False, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel('Observed within-day price SD (€/MWh)')
    fig.suptitle('Experiment 12: volatility and forecast performance', fontsize=16)
    fig.text(0.5, 0.935,
             f'{len(daily):,} test days · Each point represents one Italian calendar date',
             ha='center', color='0.35')
    fig.text(0.5, 0.015,
             'Descriptive associations with realized volatility; linear fits do not imply causation.',
             ha='center', fontsize=9, color='0.35')
    fig.tight_layout(rect=(0, 0.035, 1, 0.92))
    for extension in ('png', 'pdf'):
        fig.savefig(FIGURES / f'experiment_12_volatility_correlations.{extension}', dpi=200)
    plt.close(fig)


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(f'Predictions CSV not found: {INPUT}\nPut it in the main Entsoe dataRev folder, or change INPUT above.')
    for folder in (TABLES, FIGURES, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT, low_memory=False)
    required = {'Target_Date_Italy', 'Target_UTC', 'Target_Price_EUR_MWh',
                'Split', 'Q_0.100', 'Q_0.500', 'Q_0.900'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required CSV columns: {sorted(missing)}')
    df = df.loc[df['Split'].astype(str).str.strip().str.lower().eq('test')].copy()
    if df.empty:
        raise ValueError('No test observations found in Split column.')
    if df['Target_UTC'].duplicated().any():
        raise ValueError('Duplicate Target_UTC values found in test rows.')

    qcols = sorted((col for col in df.columns if col.startswith('Q_')),
                   key=lambda col: float(col[2:]))
    tau = np.array([float(col[2:]) for col in qcols])
    if len(tau) < 3 or not np.all(np.diff(tau) > 0):
        raise ValueError('Invalid quantile grid.')
    numeric = ['Target_Price_EUR_MWh'] + qcols
    df[numeric] = df[numeric].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite price or quantile values in test data.')
    df = df.copy()  # Defragment after conversion of many quantile columns.
    y = df['Target_Price_EUR_MWh'].to_numpy(dtype=float)
    q = df[qcols].to_numpy(dtype=float)
    if (np.diff(q, axis=1) < -1e-8).any():
        raise ValueError('Quantile crossings detected; inspect forecast postprocessing.')

    error = y[:, None] - q
    pinball = np.maximum(tau[None, :] * error, (tau[None, :] - 1) * error)
    # Quantile-grid approximation, integrating only from min(tau) to max(tau).
    df['CRPS_approx'] = 2 * np.trapezoid(pinball, x=tau, axis=1)
    df['MAE_median'] = np.abs(y - df['Q_0.500'].to_numpy(dtype=float))
    df['PI80_width'] = df['Q_0.900'] - df['Q_0.100']
    df['PI80_covered'] = ((y >= df['Q_0.100'].to_numpy(dtype=float)) &
                          (y <= df['Q_0.900'].to_numpy(dtype=float))).astype(int)
    df['date'] = pd.to_datetime(df['Target_Date_Italy'], errors='raise').dt.normalize()

    daily = df.groupby('date', sort=True).agg(
        n_hours=('Target_Price_EUR_MWh', 'size'),
        volatility_sd=('Target_Price_EUR_MWh', 'std'),
        mean_price=('Target_Price_EUR_MWh', 'mean'),
        CRPS_approx=('CRPS_approx', 'mean'),
        MAE=('MAE_median', 'mean'),
        PI80_width=('PI80_width', 'mean'),
        PI80_coverage=('PI80_covered', 'mean'),
    ).reset_index()
    if daily[['volatility_sd', 'CRPS_approx', 'MAE', 'PI80_width', 'PI80_coverage']].isna().any().any():
        raise ValueError('Missing daily metric; check incomplete dates.')
    # Tertiles are descriptive, based on realized target-day volatility.
    daily['volatility_group'] = pd.qcut(
        daily['volatility_sd'].rank(method='first'), 3,
        labels=['Low', 'Medium', 'High'])
    summary = daily.groupby('volatility_group', observed=True).agg(
        days=('date', 'size'),
        mean_volatility=('volatility_sd', 'mean'),
        mean_CRPS=('CRPS_approx', 'mean'),
        mean_MAE=('MAE', 'mean'),
        mean_PI80_width=('PI80_width', 'mean'),
        mean_PI80_coverage=('PI80_coverage', 'mean'),
    )
    daily.to_csv(TABLES / 'experiment_12_daily_metrics.csv', index=False)
    summary.to_csv(TABLES / 'experiment_12_volatility_group_summary.csv')

    metrics = ['CRPS_approx', 'MAE', 'PI80_width', 'PI80_coverage']
    correlations = {}
    print('\nEXPERIMENT 12 — MARKET VOLATILITY VS PROBABILISTIC PERFORMANCE')
    print('Input:', INPUT)
    print(f'Test hours: {len(df):,}; local dates: {len(daily)}')
    print(f'Range: {daily.date.min().date()} to {daily.date.max().date()}')
    print('Hours per local date:', daily.n_hours.value_counts().sort_index().to_dict())
    print(f'Quantiles: {len(tau)} (from {tau[0]:.3f} to {tau[-1]:.3f})')
    print('\nCORRELATIONS WITH OBSERVED WITHIN-DAY PRICE SD')
    for metric in metrics:
        p = pearsonr(daily['volatility_sd'], daily[metric])
        s = spearmanr(daily['volatility_sd'], daily[metric])
        correlations[metric] = (p.statistic, p.pvalue, s.statistic, s.pvalue)
        print(f'{metric:16s}: Pearson {p.statistic:+.4f} (p={p.pvalue:.3g}); '
              f'Spearman {s.statistic:+.4f} (p={s.pvalue:.3g})')
    print('\nVOLATILITY GROUP SUMMARY (coverage shown as fraction)')
    print(summary.round(4).to_string())

    plot_volatility_correlations(daily)

    for metric, ylabel, filename in [
        ('CRPS_approx', 'Daily approximate CRPS (€/MWh)', 'experiment_12_volatility_crps.png'),
        ('PI80_width', 'Mean daily PI80 width (€/MWh)', 'experiment_12_volatility_width.png'),
        ('PI80_coverage', 'Daily PI80 coverage (%)', 'experiment_12_volatility_coverage.png'),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        values = daily[metric] * (100 if metric == 'PI80_coverage' else 1)
        ax.scatter(daily['volatility_sd'], values, alpha=0.45, s=18)
        ax.set(xlabel='Observed within-day price SD (€/MWh)', ylabel=ylabel,
               title=f'Experiment 12: volatility vs {metric}')
        if metric == 'PI80_coverage':
            ax.axhline(80, linestyle='--', linewidth=1, label='Nominal 80%')
            ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, label in zip(axes,
                                 ['mean_CRPS', 'mean_PI80_width', 'mean_PI80_coverage'],
                                 ['Approx. CRPS (€/MWh)', 'PI80 width (€/MWh)', 'PI80 coverage (%)']):
        values = summary[metric].to_numpy() * (100 if 'coverage' in metric else 1)
        ax.bar(summary.index.astype(str), values)
        ax.set(xlabel='Realized volatility group', ylabel=label)
        if 'coverage' in metric:
            ax.axhline(80, linestyle='--', linewidth=1)
        ax.grid(axis='y', alpha=0.2)
    fig.suptitle('Experiment 12: performance by realized volatility tertile')
    fig.tight_layout()
    fig.savefig(FIGURES / 'experiment_12_volatility_groups.png', dpi=180)
    plt.close(fig)

    lines = [
        '# Experiment 12 — Market volatility versus probabilistic LEAR v3 performance',
        '',
        '## Research question',
        'How does observed within-day electricity price volatility relate to the accuracy, '
        'sharpness, and calibration of native probabilistic LEAR v3 forecasts?',
        '',
        '## Data and procedure',
        f'- Source: `{INPUT.name}`; `Split == test` only.',
        f'- {len(df):,} test hours across {len(daily)} Italian calendar dates '
        f'({daily.date.min().date()} to {daily.date.max().date()}).',
        '- Volatility: sample standard deviation of realized hourly electricity prices within each Italian calendar date.',
        '- Approximate CRPS: twice the trapezoidal integral of pinball loss across available quantiles.',
        '- MAE: absolute error of the predicted median, averaged by date.',
        '- PI80 width: Q0.900 minus Q0.100, averaged by date.',
        '- PI80 coverage: share of hourly actual prices within [Q0.100, Q0.900] per date.',
        '- Days assigned to equally sized low/medium/high groups using realized volatility ranks.',
        '',
        '## Results — daily correlations',
        '| Metric | Pearson r | Pearson p | Spearman rho | Spearman p |',
        '|---|---:|---:|---:|---:|',
    ]
    for metric, (r, p, rho, sp) in correlations.items():
        lines.append(f'| {metric} | {r:+.4f} | {p:.3g} | {rho:+.4f} | {sp:.3g} |')
    lines += [
        '',
        '## Results — volatility groups',
        '| Group | Days | Mean volatility | Mean CRPS | Mean MAE | Mean PI80 width | PI80 coverage |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for group, row in summary.iterrows():
        lines.append(f'| {group} | {int(row.days)} | {row.mean_volatility:.2f} | '
                     f'{row.mean_CRPS:.2f} | {row.mean_MAE:.2f} | '
                     f'{row.mean_PI80_width:.2f} | {100 * row.mean_PI80_coverage:.2f}% |')
    lines += [
        '',
        '## Interpretation and limitations',
        '- Associations are descriptive, not evidence of causation or model failure.',
        '- Same-day realized volatility is known only after the target day; it cannot serve as a same-day forecast-issue early-warning feature.',
        '- Daily observations may be serially dependent, so conventional correlation p-values should not be treated as definitive inference.',
        '- Approximate CRPS integrates only over the supplied quantile range, not the entire distribution tails.',
        '- A daily average of coverage is noisy; the nominal 80% rate is not guaranteed on each individual day.',
        '- Group boundaries are defined using the full test sample; they are descriptive rather than prospectively calibrated thresholds.',
        '- This experiment does not establish whether deterioration is model-specific; benchmark-relative performance is the next question.',
        '',
        '## Saved outputs',
        '- `figures/experiment_12_volatility_correlations.png` (all four metrics, with Pearson and Spearman correlations)',
        '- `figures/experiment_12_volatility_correlations.pdf`',
        '- `tables/experiment_12_daily_metrics.csv`',
        '- `tables/experiment_12_volatility_group_summary.csv`',
        '- `figures/experiment_12_volatility_crps.png`',
        '- `figures/experiment_12_volatility_width.png`',
        '- `figures/experiment_12_volatility_coverage.png`',
        '- `figures/experiment_12_volatility_groups.png`',
        '- `reports/experiment_12_report.md` (this file)',
    ]
    report_path = REPORTS / 'experiment_12_report.md'
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\nTables:', TABLES)
    print('Figures:', FIGURES)
    print('Report:', report_path)


if __name__ == '__main__':
    main()
