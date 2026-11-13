#!/usr/bin/env python3
"""Experiment 14: compare fixed, relative and lagged-volatility-aware monitoring.

Place in Entsoe dataRev/monitoring_research/scripts/.
Requires experiment_13_daily_metrics.csv in monitoring_research/tables/.
Run Experiment 13 first. Outputs tables/*.csv, figures/*.png, reports/*.md.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MON = ROOT / 'monitoring_research'
TABLES, FIGURES, REPORTS = (MON / x for x in ('tables', 'figures', 'reports'))
INPUT = TABLES / 'experiment_13_daily_metrics.csv'
REFERENCE_END = '2026-02-28'
EVALUATION_START = '2026-03-01'
ALPHA = 0.95
VOL_WINDOW = 7


def savefig(name, pdf=False):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=200, bbox_inches='tight')
    if pdf:
        plt.savefig((FIGURES / name).with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


def alert_episodes(series):
    """Number of distinct consecutive-day alert runs."""
    s = series.astype(bool)
    return int((s & ~s.shift(1, fill_value=False)).sum())


def additional_figures(ref, ev, slope, intercept, residual_limit):
    """Visualize the fitted rule and evaluation categories without refitting."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for frame, label, color in [(ref, 'Reference', '#2878a0'),
                                (ev, 'Evaluation', '#d48a35')]:
        ax.scatter(frame.log_past7_volatility, frame.CRPS_native,
                   s=24, alpha=.5, color=color, label=f'{label} (n={len(frame)})')
    limits = np.array([min(ref.log_past7_volatility.min(), ev.log_past7_volatility.min()),
                       max(ref.log_past7_volatility.max(), ev.log_past7_volatility.max())])
    expected = intercept + slope * limits
    ax.plot(limits, expected, color='#235777', linewidth=2,
            label=f'Reference fit: y = {intercept:.3f} {slope:+.3f}x')
    ax.plot(limits, expected + residual_limit, color='#b64c3c', linestyle='--',
            linewidth=2, label=f'Alert boundary: fit + {residual_limit:.3f}')
    ax.set(xlabel='log(1 + previous-seven-day mean price SD)',
           ylabel='Daily approximate CRPS (€/MWh)',
           title='Experiment 14 — Reference-fitted market-aware monitoring rule')
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    savefig('experiment_14_market_regression.png', pdf=True)

    labels = ['Neither alerts', 'Fixed only', 'Market-aware only', 'Both alert']
    colors = ['#8a9ba8', '#cf893c', '#388453', '#aa514b']
    masks = [~ev.alert_fixed & ~ev.alert_market,
             ev.alert_fixed & ~ev.alert_market,
             ~ev.alert_fixed & ev.alert_market,
             ev.alert_fixed & ev.alert_market]
    counts = [int(mask.sum()) for mask in masks]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts, color=colors)
    ax.bar_label(bars, labels=[f'{n} days ({100*n/len(ev):.1f}%)' for n in counts], padding=5)
    ax.set_ylim(0, max(1, max(counts)) * 1.2)
    ax.set(ylabel='Evaluation days',
           title=f'Experiment 14 — Fixed vs market-aware disagreement ({len(ev)} days)')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    savefig('experiment_14_alert_disagreement.png', pdf=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    panels = [
        ('CRPS_native', 'Daily approximate CRPS (€/MWh)', 1, None),
        ('rMAE_native_vs_naive', 'Daily rMAE (lower is better)', 1, 1),
        ('CRPSS_native_vs_residual', 'Daily CRPSS (higher is better)', 1, 0),
        ('PI80_native_coverage', 'Daily PI80 coverage (%)', 100, 80),
    ]
    rng = np.random.default_rng(14)  # Reproducible visual jitter only.
    for ax, (metric, title, scale, reference) in zip(axes.flat, panels):
        for position, (mask, color) in enumerate(zip(masks, colors), start=1):
            values = ev.loc[mask, metric].to_numpy() * scale
            if not len(values):
                continue
            boxes = ax.boxplot([values], positions=[position], widths=.5,
                               patch_artist=True, showfliers=False,
                               medianprops=dict(color='black', linewidth=1.5))
            boxes['boxes'][0].set(facecolor=color, alpha=.3)
            ax.scatter(position + rng.uniform(-.16, .16, len(values)), values,
                       color=color, alpha=.4, s=15, edgecolors='none')
            ax.text(position, 1.02, f'Median {np.median(values):.2f}',
                    transform=ax.get_xaxis_transform(), ha='center', fontsize=9)
        if reference is not None:
            ax.axhline(reference, color='0.35', linestyle='--', linewidth=1,
                       label='Nominal 80%' if scale == 100 else 'Equal performance')
            ax.legend(loc='best', fontsize=8)
        ax.set(xticks=range(1, 5),
               xticklabels=[f'{label}\n(n={n})' for label, n in zip(labels, counts)],
               ylabel=title, xlim=(.5, 4.5))
        ax.set_title(title, pad=30)
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Experiment 14 — Performance by fixed / market-aware alert category', fontsize=15)
    fig.text(.5, .01,
             'Evaluation days only. Boxes: quartiles and median; whiskers: 1.5 × IQR. Points: individual days.\n'
             'Categories are defined by CRPS thresholds; differences are descriptive, not independent validation.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .96))
    # Save directly to retain room for the explanatory footer.
    for extension in ('png', 'pdf'):
        fig.savefig(FIGURES / f'experiment_14_disagreement_performance.{extension}',
                    dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    for folder in (TABLES, FIGURES, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)
    if not INPUT.is_file():
        raise FileNotFoundError(f'Missing {INPUT}\nRun experiment_13_benchmark_relative_performance.py first.')
    df = pd.read_csv(INPUT, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    needed = ['date', 'volatility', 'CRPS_native', 'rMAE_native_vs_naive',
              'CRPSS_native_vs_residual', 'PI80_native_coverage']
    missing = [x for x in needed if x not in df.columns]
    if missing:
        raise ValueError(f'Experiment 13 table missing columns: {missing}')
    if df.date.duplicated().any() or df[needed[1:]].isna().any().any():
        raise ValueError('Duplicate dates or missing monitoring metrics')
    if not (df.date.diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError('Daily dates are not consecutive; verify missing days before using lagged volatility')

    # Historical information only: for day t, mean realized volatility on t-7,...,t-1.
    df['past7_volatility'] = df['volatility'].shift(1).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).mean()
    df['log_past7_volatility'] = np.log1p(df['past7_volatility'])
    ref = df[(df.date <= REFERENCE_END) & df.log_past7_volatility.notna()].copy()
    ev_mask = df.date >= EVALUATION_START
    ev = df.loc[ev_mask].copy()
    if len(ref) < 100 or len(ev) < 30:
        raise ValueError(f'Insufficient reference/evaluation data: {len(ref)} / {len(ev)}')

    # Thresholds fixed from reference data only, applied without refitting in evaluation.
    fixed_crps = float(ref.CRPS_native.quantile(ALPHA))
    rmae_upper = float(ref.rMAE_native_vs_naive.quantile(ALPHA))
    crpss_lower = float(ref.CRPSS_native_vs_residual.quantile(1 - ALPHA))

    # Reference-only linear regression: CRPS ~ log(1 + previous-seven-day volatility).
    # The residual threshold is one-sided (95th percentile) on in-sample reference residuals.
    xref = ref.log_past7_volatility.to_numpy(float)
    yref = ref.CRPS_native.to_numpy(float)
    slope, intercept = np.polyfit(xref, yref, 1)
    ref_residual = yref - (slope * xref + intercept)
    residual_limit = float(np.quantile(ref_residual, ALPHA))
    df['market_expected_CRPS'] = slope * df.log_past7_volatility + intercept
    df['market_CRPS_threshold'] = df.market_expected_CRPS + residual_limit
    df['market_CRPS_excess'] = df.CRPS_native - df.market_expected_CRPS
    df['fixed_CRPS_threshold'] = fixed_crps
    df['rMAE_reference_threshold'] = rmae_upper
    df['CRPSS_reference_threshold'] = crpss_lower
    df['alert_fixed'] = df.CRPS_native > fixed_crps
    df['alert_market'] = df.CRPS_native > df.market_CRPS_threshold
    df['alert_rMAE'] = df.rMAE_native_vs_naive > rmae_upper
    df['alert_CRPSS'] = df.CRPSS_native_vs_residual < crpss_lower
    df['alert_relative_either'] = df.alert_rMAE | df.alert_CRPSS
    df['period'] = np.where(df.date <= REFERENCE_END, 'Reference', 'Evaluation')
    df.to_csv(TABLES / 'experiment_14_daily_monitoring.csv', index=False)

    ev = df.loc[ev_mask].copy()
    additional_figures(ref, ev, slope, intercept, residual_limit)
    names = [('Fixed CRPS', 'alert_fixed'), ('Market-aware CRPS', 'alert_market'),
             ('Relative rMAE', 'alert_rMAE'), ('Relative CRPSS', 'alert_CRPSS'),
             ('Either relative signal', 'alert_relative_either')]
    summary = pd.DataFrame([{
        'approach': name, 'alert_days': int(ev[col].sum()),
        'alert_pct': float(100 * ev[col].mean()), 'alert_episodes': alert_episodes(ev[col]),
        'evaluation_days': len(ev)
    } for name, col in names])
    summary.to_csv(TABLES / 'experiment_14_alert_summary.csv', index=False)
    overlap = pd.DataFrame(index=[name for name, _ in names], columns=[name for name, _ in names], dtype=int)
    for a, ca in names:
        for b, cb in names:
            overlap.loc[a, b] = int((ev[ca] & ev[cb]).sum())
    overlap.astype(int).to_csv(TABLES / 'experiment_14_alert_overlap.csv')

    # Figure 1: absolute CRPS with reference-only fixed and conditional thresholds.
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ev.date, ev.CRPS_native, label='Daily native CRPS', linewidth=1, color='#2878a0')
    ax.axhline(fixed_crps, linestyle='--', color='#b64c3c', label=f'Fixed reference P95 = {fixed_crps:.2f}')
    ax.plot(ev.date, ev.market_CRPS_threshold, color='#388453', linewidth=1.6,
            label='Market-aware threshold (lagged 7-day volatility)')
    ax.set(title='Experiment 14 — Daily CRPS and reference-derived monitoring thresholds',
           ylabel='Approximate CRPS (€/MWh)', xlabel='Italian calendar date')
    ax.grid(alpha=.2); ax.legend(loc='upper left')
    savefig('experiment_14_crps_threshold_comparison.png')

    # Figure 2: two benchmark-relative signals with equal-performance and reference limits.
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, col, limit, equal, title in [
        (axes[0], 'rMAE_native_vs_naive', rmae_upper, 1, 'Daily rMAE (larger = worse)'),
        (axes[1], 'CRPSS_native_vs_residual', crpss_lower, 0, 'Daily CRPSS (smaller = worse)')]:
        ax.plot(ev.date, ev[col], linewidth=.9, color='#2878a0')
        ax.axhline(equal, color='black', linestyle='--', label=f'Equal performance = {equal}')
        ax.axhline(limit, color='#b64c3c', linestyle=':', linewidth=1.8,
                   label=f'Reference warning threshold = {limit:.3f}')
        ax.set(ylabel=title); ax.grid(alpha=.2); ax.legend(loc='upper right')
    axes[-1].set_xlabel('Italian calendar date')
    fig.suptitle('Experiment 14 — Relative monitoring signals')
    savefig('experiment_14_relative_monitoring.png')

    # Figure 3: alert timeline (not validated detection of real failures).
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (name, col) in enumerate(names):
        dates = ev.loc[ev[col], 'date']
        ax.scatter(dates, np.full(len(dates), i), marker='|', s=100, linewidths=1.5)
    ax.set_yticks(range(len(names)), [n for n, _ in names])
    ax.set(title='Experiment 14 — Days flagged by each monitoring approach', xlabel='Italian calendar date')
    ax.grid(axis='x', alpha=.2)
    savefig('experiment_14_alert_timeline.png')

    # Figure 4: numerical counts.
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(summary.approach, summary.alert_days, color='#2878a0')
    ax.bar_label(bars, labels=[f'{n} days ({p:.1f}%)' for n, p in zip(summary.alert_days, summary.alert_pct)], padding=5)
    ax.set_xlim(0, max(1, summary.alert_days.max()) * 1.35)
    ax.set(xlabel=f'Alert days out of {len(ev)} evaluation days', title='Experiment 14 — Numerical alert comparison')
    ax.invert_yaxis(); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    savefig('experiment_14_alert_counts.png')

    # Figure 5: overlap in number of co-flagged days.
    fig, ax = plt.subplots(figsize=(9, 7))
    mat = overlap.astype(int).to_numpy()
    im = ax.imshow(mat, cmap='Blues', vmin=0)
    ax.set_xticks(range(len(names)), [n for n, _ in names], rotation=35, ha='right')
    ax.set_yticks(range(len(names)), [n for n, _ in names])
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(mat[i, j]), ha='center', va='center',
                    color='white' if mat[i, j] > mat.max() * .55 else 'black')
    fig.colorbar(im, ax=ax, label='Days flagged by both methods')
    ax.set_title('Experiment 14 — Alert overlap (evaluation days)')
    savefig('experiment_14_alert_overlap.png')

    rows = '\n'.join(f'| {r.approach} | {r.alert_days} | {r.alert_pct:.1f}% | {r.alert_episodes} |'
                     for r in summary.itertuples(index=False))
    report = f'''# Experiment 14 — Market-aware health monitoring

## Research question
Can reference-derived, market-aware thresholds distinguish unusual LEAR v3 errors from ordinary market difficulty better than fixed and benchmark-relative alerts?

## Inputs and chronology
- Input: `tables/experiment_13_daily_metrics.csv` generated by Experiment 13.
- Reference period: {ref.date.min().date()} to {ref.date.max().date()} ({len(ref)} days with a complete historical-volatility window).
- Evaluation period: {ev.date.min().date()} to {ev.date.max().date()} ({len(ev)} days).
- Thresholds and linear regression are estimated **only from reference data**, then frozen for evaluation.
- The explanatory volatility variable is the average observed daily price SD of the **previous seven complete Italian calendar days** (shifted by one day). Current-day realized volatility is never used to set its threshold.
- This is retrospective, **post-realization monitoring**, not forecast-time early warning: the day's CRPS, rMAE and CRPSS become available only after actual prices are observed.

## Methods
- **Fixed CRPS:** reference-period 95th percentile of daily native approximate CRPS = {fixed_crps:.4f} €/MWh; alert above it.
- **Relative rMAE:** reference 95th percentile = {rmae_upper:.4f}; alert above it.
- **Relative CRPSS:** reference 5th percentile = {crpss_lower:.4f}; alert below it.
- **Market-aware CRPS:** fit reference-only OLS `expected_CRPS = {intercept:.4f} + {slope:.4f} × log(1 + past7_volatility)`; alert if actual CRPS exceeds expected CRPS plus the reference 95th percentile of regression residuals ({residual_limit:.4f}).
- **Either relative signal:** union of the two relative alerts; shown separately to avoid hiding disagreement.
- The fixed and conditional thresholds are exploratory one-sided reference thresholds. The regression uses in-sample residuals, which may make calibration optimistic; no out-of-sample tuning was done.

## Results — evaluation period
| Monitoring approach | Alert days | Percent of evaluation days | Alert episodes |
|---|---:|---:|---:|
{rows}

## Outputs
- `tables/experiment_14_daily_monitoring.csv`: all daily indicators, thresholds and alert flags.
- `tables/experiment_14_alert_summary.csv`: counts and episode counts.
- `tables/experiment_14_alert_overlap.csv`: co-flagged day counts.
- `figures/experiment_14_crps_threshold_comparison.png`: fixed vs conditional CRPS thresholds.
- `figures/experiment_14_relative_monitoring.png`: relative scores with reference and equal-performance lines.
- `figures/experiment_14_alert_timeline.png`: days flagged by each approach.
- `figures/experiment_14_alert_counts.png`: labeled numerical counts.
- `figures/experiment_14_alert_overlap.png`: alert overlap heatmap.

## Additional diagnostic figures
- [Market-aware regression scatterplot](../figures/experiment_14_market_regression.png): reference and evaluation days, the straight reference-fitted regression line, and the frozen conditional alert boundary. The horizontal axis is log-transformed historical volatility, matching the fitted model.
- [Fixed versus market-aware disagreement](../figures/experiment_14_alert_disagreement.png): counts and percentages of evaluation days with neither alert, fixed only, market-aware only, or both alerts.
- [Performance by alert category](../figures/experiment_14_disagreement_performance.png): daily CRPS, rMAE, CRPSS, and PI80 coverage distributions, with numerical medians and category sizes. These are daily distributions, not pooled skill ratios.
- These three figures are also available as PDFs with the same filename stems.
- Alert categories are defined using CRPS, so differences in CRPS between categories are descriptive consequences of the monitoring rules, not independent evidence of detection quality. No failure labels are available.

## Interpretation and limitations
- Fewer alerts do **not** prove better monitoring. Compare overlap, chronology, and later externally defined outcomes.
- No independently verified model-failure labels exist. The sustained CRPS episode from Experiment 11 is an exploratory screening event, not ground truth.
- The volatility relationship is based on lagged volatility; it may not explain abrupt current-day spikes.
- Daily time-series dependence means a nominal reference 5% tail does not imply a 5% prospective false-alarm rate.
- The reference period and evaluation period may differ in ways beyond volatility; linear regression may be misspecified.
- The current-day realized prices are used to calculate performance metrics **after** the target day, not as an input to the market-aware threshold.
- Next, evaluate detector algorithms and prospective warning performance chronologically in Experiments 15–16.
'''
    (REPORTS / 'experiment_14_report.md').write_text(report, encoding='utf-8')

    print('\nEXPERIMENT 14 — MARKET-AWARE HEALTH MONITORING')
    print(f'Reference: {len(ref)} days ({ref.date.min().date()} to {ref.date.max().date()})')
    print(f'Evaluation: {len(ev)} days ({ev.date.min().date()} to {ev.date.max().date()})')
    print(f'Fixed CRPS P95: {fixed_crps:.4f}; rMAE P95: {rmae_upper:.4f}; CRPSS P05: {crpss_lower:.4f}')
    print(f'Market model: expected CRPS = {intercept:.4f} + {slope:.4f} * log1p(past7_volatility)')
    print(f'Market residual P95: {residual_limit:.4f}')
    print('\nALERT SUMMARY')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    print('\nTables:', TABLES, '\nFigures:', FIGURES, '\nReport:', REPORTS / 'experiment_14_report.md')


if __name__ == '__main__':
    main()
