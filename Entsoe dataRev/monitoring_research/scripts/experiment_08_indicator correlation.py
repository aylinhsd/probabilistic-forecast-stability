#!/usr/bin/env python3
"""Experiment 08: LEAR monitoring indicator correlation and redundancy.

Read-only input: market_regime_results/health_baseline/daily_health_baseline.csv
Outputs: monitoring_research/{tables,figures,reports}/experiment_08_*

This is exploratory correlation analysis, not an index-weight optimizer.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'market_regime_results/health_baseline/daily_health_baseline.csv'
BASE = ROOT / 'monitoring_research'
TABLES, FIGURES, REPORTS = (BASE / n for n in ('tables', 'figures', 'reports'))
for folder in (TABLES, FIGURES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

REF_START, REF_END = '2025-07-01', '2025-12-31'
EVAL_START, EVAL_END = '2026-01-01', '2026-10-13'
ROLL_WINDOW, ROLL_MIN = 30, 15

# Raw daily indicators. Absolute coverage deviation penalizes both over/undercoverage.
INDICATORS = {
    'crps': 'CRPS',
    'mae': 'MAE',
    'rmse': 'RMSE',
    'interval_score80': 'Interval score 80%',
    'coverage80_abs_deviation': '|Coverage80 - 0.80|',
    'pit_mean_deviation': '|PIT mean - 0.50|',
    'pit_variance_deviation': '|PIT variance - 1/12|',
}
FORECAST_CONTEXT = {'width80': 'Width 80%'}


def plot_matrix(matrix, title, output_stem):
    names = list(matrix.columns)
    size = max(7.5, len(names) * 1.05)
    fig, ax = plt.subplots(figsize=(size + 1.4, size))
    im = ax.imshow(matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), labels=names, rotation=45, ha='right')
    ax.set_yticks(range(len(names)), labels=names)
    for i in range(len(names)):
        for j in range(len(names)):
            value = matrix.iloc[i, j]
            ax.text(j, i, 'NA' if pd.isna(value) else f'{value:.2f}',
                    ha='center', va='center', fontsize=9,
                    color='white' if pd.notna(value) and abs(value) > 0.72 else 'black')
    ax.set_title(title, pad=14)
    fig.colorbar(im, ax=ax, label='Correlation coefficient', shrink=.8)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIGURES / f'{output_stem}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_matrix(frame, subset, granularity, method, columns):
    corr = frame.loc[subset, columns].corr(method=method, min_periods=20)
    corr.index = [INDICATORS.get(c, FORECAST_CONTEXT.get(c, c)) for c in corr.index]
    corr.columns = [INDICATORS.get(c, FORECAST_CONTEXT.get(c, c)) for c in corr.columns]
    stem = f'experiment_08_{granularity}_{subset.name}_{method}'
    corr.to_csv(TABLES / f'{stem}.csv', float_format='%.6f')
    plot_matrix(corr, f'{method.title()} correlation — {granularity}, {subset.name}', stem)
    return corr


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f'Missing input: {INPUT}')
    df = pd.read_csv(INPUT, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    if df['date'].duplicated().any():
        raise ValueError('Duplicate daily dates detected; inspect baseline before proceeding.')
    required = set(INDICATORS) - {'pit_variance_deviation'}
    required |= {'date', 'pit_var', 'coverage80', 'width80'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    df['pit_variance_deviation'] = (df['pit_var'] - 1.0 / 12.0).abs()
    # Recalculate absolute deviations to make their definition explicit.
    df['coverage80_abs_deviation'] = (df['coverage80'] - 0.80).abs()
    df['pit_mean_deviation'] = (df['pit_mean'] - 0.50).abs()

    cols = list(INDICATORS)
    # Rolling metrics are computed across the chronological sequence, without future data.
    # A rolling correlation can be inflated by overlapping windows and common trends.
    rolling = df[['date']].copy()
    for c in cols + ['width80']:
        rolling[c] = df[c].rolling(ROLL_WINDOW, min_periods=ROLL_MIN).mean()

    masks = {
        'calibration': df['date'].between(REF_START, REF_END),
        'evaluation': df['date'].between(EVAL_START, EVAL_END),
    }
    if not all(mask.sum() >= 30 for mask in masks.values()):
        raise ValueError('Insufficient observations in calibration/evaluation date ranges.')
    # Use explicit labels so correlations of error metrics and width are distinguishable.
    summaries = []
    pair_records = []
    for granularity, data in [('daily', df), ('rolling30', rolling)]:
        for period, mask in masks.items():
            idx = pd.Series(mask.to_numpy(), index=data.index, name=period)
            part = data.loc[idx, cols + ['width80']]
            summaries.append({
                'granularity': granularity, 'period': period,
                'days_total': int(mask.sum()),
                'days_complete': int(part[cols].dropna().shape[0]),
            })
            for method in ('pearson', 'spearman'):
                corr = save_matrix(data, idx, granularity, method, cols)
                for i, a in enumerate(corr.columns):
                    for b in corr.columns[i+1:]:
                        pair_records.append({
                            'granularity': granularity, 'period': period,
                            'method': method, 'indicator_a': a, 'indicator_b': b,
                            'correlation': corr.loc[a, b],
                        })
                # Width is a forecast-time contextual signal, not a realized health metric.
                width_corr = part.corr(method=method, min_periods=20)['width80'].drop('width80')
                for col, value in width_corr.items():
                    pair_records.append({
                        'granularity': granularity, 'period': period,
                        'method': method, 'indicator_a': 'Width 80% (forecast-time)',
                        'indicator_b': INDICATORS[col], 'correlation': value,
                    })

    pairs = pd.DataFrame(pair_records)
    pairs['absolute_correlation'] = pairs['correlation'].abs()
    pairs.to_csv(TABLES / 'experiment_08_indicator_pairs.csv', index=False, float_format='%.6f')
    pd.DataFrame(summaries).to_csv(TABLES / 'experiment_08_sample_sizes.csv', index=False)
    high = pairs[(pairs['absolute_correlation'] >= .85) &
                 (pairs['indicator_a'] != 'Width 80% (forecast-time)')].copy()
    high.sort_values(['granularity', 'period', 'method', 'absolute_correlation'],
                     ascending=[True, True, True, False]).to_csv(
        TABLES / 'experiment_08_high_correlation_pairs.csv', index=False, float_format='%.6f')

    # Compare each pair's Pearson correlation across periods; descriptive only.
    daily_pearson = pairs[(pairs.granularity == 'daily') & (pairs.method == 'pearson')]
    comparison = daily_pearson.pivot(index=['indicator_a', 'indicator_b'],
                                     columns='period', values='correlation').reset_index()
    comparison['evaluation_minus_calibration'] = comparison['evaluation'] - comparison['calibration']
    comparison.to_csv(TABLES / 'experiment_08_period_comparison.csv',
                      index=False, float_format='%.6f')

    def table_markdown(rows, max_rows=12):
        if rows.empty:
            return 'No pairs reached the descriptive |r| ≥ 0.85 screening level.'
        lines = ['| Period | Indicators | Pearson r |', '|---|---|---:|']
        for _, r in rows.head(max_rows).iterrows():
            lines.append(f"| {r['period']} | {r['indicator_a']} / {r['indicator_b']} | {r['correlation']:+.3f} |")
        return '\n'.join(lines)

    high_daily = high[(high.granularity == 'daily') & (high.method == 'pearson')]
    high_daily = high_daily.sort_values('absolute_correlation', ascending=False)
    report = f"""# Experiment 08 — Indicator Correlation and Redundancy Analysis

## Purpose
Examine whether realized forecast-quality indicators carry overlapping information before selecting components of the LEAR Health Index. This experiment does **not** choose final weights or optimize the index.

## Input and periods
- Input: `{INPUT.relative_to(ROOT)}`
- Calibration/reference: {REF_START} through {REF_END} ({masks['calibration'].sum()} days).
- Evaluation: {EVAL_START} through {EVAL_END} ({masks['evaluation'].sum()} days).
- Daily Pearson and Spearman correlations; also trailing {ROLL_WINDOW}-day averages with a minimum of {ROLL_MIN} days.
- Rolling windows use current and earlier dates only; early evaluation rolling values include late calibration days.

## Indicators
- CRPS, MAE, RMSE, 80% interval score: realized forecast performance.
- Absolute 80% coverage deviation: `abs(observed coverage - 0.80)`, penalizing both over- and undercoverage.
- Absolute PIT-mean deviation: `abs(mean PIT - 0.50)`; does **not** measure complete PIT uniformity.
- Absolute PIT-variance deviation: `abs(PIT variance - 1/12)`; another incomplete PIT diagnostic.
- 80% interval width is correlated with the health metrics separately, because width is available at forecast time and is **not** itself a realized-error metric.

## High correlations in raw daily Pearson analysis (descriptive |r| >= 0.85)
{table_markdown(high_daily)}

## Interpretation rules and limitations
1. Pearson measures linear association; Spearman measures rank-monotonic association. Neither establishes causality or that two metrics are interchangeable.
2. High correlations can arise from common market difficulty, outliers, seasonal changes, or overlapping rolling windows. Check raw daily results before interpreting rolling correlations.
3. The 2025 reference period lacks spring and is only six months long; cross-period changes are confounded with seasonal and market changes.
4. Daily absolute coverage deviations are noisy because there are usually only 24 hourly observations per day; conditional hourly coverage needs longer aggregation.
5. PIT-mean and PIT-variance deviations are incomplete calibration diagnostics. Experiment 05's distributional PIT findings remain important.
6. Width is available at forecast issue time; realized CRPS, MAE, RMSE, interval score, PIT and coverage are not. Keep Health and Early-Warning signals separate.
7. |r| >= 0.85 is a **screening convention**, not an automatic deletion rule or a statistical significance threshold. Serial dependence invalidates naive independent-day significance claims.

## Outputs
- `experiment_08_*_pearson.csv` and `experiment_08_*_spearman.csv`: matrices by period and granularity.
- `experiment_08_indicator_pairs.csv`: pairwise correlations, including width comparisons.
- `experiment_08_high_correlation_pairs.csv`: screening list.
- `experiment_08_period_comparison.csv`: changes in daily Pearson correlations.
- `experiment_08_sample_sizes.csv`: available observations.
- Matching PNG and PDF heatmaps.

## Decision before Experiment 09
Review the empirical matrices and select complementary signals. Do not treat this exploratory analysis as validated index construction or as a detector evaluation.
"""
    (REPORTS / 'experiment_08_report.md').write_text(report, encoding='utf-8')

    print('EXPERIMENT 08 — COMPLETE')
    print('Reference days:', int(masks['calibration'].sum()), '| Evaluation days:', int(masks['evaluation'].sum()))
    print('\nStrongest raw daily Pearson pairs (excluding width):')
    print(pairs[(pairs.granularity == 'daily') & (pairs.method == 'pearson') &
                (pairs.indicator_a != 'Width 80% (forecast-time)')]
          .sort_values('absolute_correlation', ascending=False)
          [['period', 'indicator_a', 'indicator_b', 'correlation']].head(14).to_string(index=False))
    print('\nSaved report:', REPORTS / 'experiment_08_report.md')
    print('Saved heatmaps: 8 PNG + 8 PDF; matrices and summary tables in', TABLES)

if __name__ == '__main__':
    main()
