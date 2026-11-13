#!/usr/bin/env python3
"""Experiment 14B: descriptive alert severity, context, and follow-up.
Run after Experiment 14. No independent failure labels are assumed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MON = ROOT / 'monitoring_research'
TABLES, FIGURES, REPORTS = (MON / s for s in ('tables', 'figures', 'reports'))
INPUT = TABLES / 'experiment_14_daily_monitoring.csv'
EVAL_START = pd.Timestamp('2026-03-01')
FOLLOWUP = 7
COVERAGE_TARGET = 0.80
COVERAGE_WINDOWS = (7, 30)

def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    for folder in (TABLES, FIGURES, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)
    if not INPUT.exists():
        raise FileNotFoundError(f'{INPUT} missing: run Experiment 14 first.')
    df = pd.read_csv(INPUT, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    required = ['date','CRPS_native','rMAE_native_vs_naive','CRPSS_native_vs_residual',
                'PI80_native_coverage','volatility','past7_volatility','market_expected_CRPS',
                'market_CRPS_threshold','fixed_CRPS_threshold','rMAE_reference_threshold',
                'CRPSS_reference_threshold','alert_fixed','alert_market','alert_rMAE','alert_CRPSS']
    missing = [c for c in required if c not in df]
    if missing: raise ValueError(f'Missing columns: {missing}')
    for c in ['alert_fixed','alert_market','alert_rMAE','alert_CRPSS']:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.lower().map({'true':True,'false':False})
        if df[c].isna().any(): raise ValueError(f'Invalid alert values: {c}')
        df[c] = df[c].astype(bool)
    if df.date.duplicated().any() or not (df.date.diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError('Dates must be unique and consecutive')
    # Coverage monitoring: compute rolling statistics across the full chronological
    # series so evaluation-day windows can use reference-period history.
    # Daily observations are approximately 24 hourly forecasts (23/25 on DST days).
    # These are UNWEIGHTED daily averages because this input is daily aggregated.
    # For exact pooled hourly coverage, use hourly covered counts / hourly totals.
    df['coverage_deviation_signed_pp'] = 100 * (df.PI80_native_coverage - COVERAGE_TARGET)
    df['coverage_deviation_abs_pp'] = df.coverage_deviation_signed_pp.abs()
    for w in COVERAGE_WINDOWS:
        r = df.PI80_native_coverage.rolling(window=w, min_periods=w)
        df[f'coverage_rolling{w}'] = r.mean()
        df[f'coverage_rolling{w}_deviation_signed_pp'] = 100 * (df[f'coverage_rolling{w}'] - COVERAGE_TARGET)
        df[f'coverage_rolling{w}_deviation_abs_pp'] = df[f'coverage_rolling{w}_deviation_signed_pp'].abs()
        df[f'coverage_daily_sd{w}_pp'] = 100 * r.std(ddof=1)
    # Calibrate upper-tail stability thresholds on reference days only.
    # Rolling windows use past/current realized coverage, so alerts are post-realization.
    reference_mask = df.date < EVAL_START
    coverage_limits = {}
    for w in COVERAGE_WINDOWS:
        for metric in (f'coverage_rolling{w}_deviation_abs_pp', f'coverage_daily_sd{w}_pp'):
            baseline = df.loc[reference_mask, metric].dropna()
            if len(baseline) < 30:
                raise ValueError(f'Insufficient reference history for {metric}: {len(baseline)}')
            limit = float(baseline.quantile(.95))
            coverage_limits[metric] = limit
            df[f'{metric}_reference_p95'] = limit
            df[f'{metric}_alert'] = df[metric] > limit
    # Under/over coverage directions are reported separately, not conflated.
    df['coverage_rolling30_under80'] = df['coverage_rolling30'] < COVERAGE_TARGET
    df['coverage_rolling30_over80'] = df['coverage_rolling30'] > COVERAGE_TARGET
    df['alert_relative_either'] = df.alert_rMAE | df.alert_CRPSS
    df['category'] = np.select(
        [df.alert_fixed & df.alert_market, df.alert_fixed & ~df.alert_market,
         ~df.alert_fixed & df.alert_market],
        ['Both alert','Fixed only','Market-aware only'], default='Neither alerts')
    df['fixed_exceedance'] = (df.CRPS_native - df.fixed_CRPS_threshold).clip(lower=0)
    df['fixed_exceedance_pct'] = 100*df.fixed_exceedance / df.fixed_CRPS_threshold
    df['market_residual'] = df.CRPS_native - df.market_expected_CRPS
    df['market_exceedance'] = (df.CRPS_native - df.market_CRPS_threshold).clip(lower=0)
    df['rMAE_exceedance'] = (df.rMAE_native_vs_naive - df.rMAE_reference_threshold).clip(lower=0)
    df['CRPSS_exceedance'] = (df.CRPSS_reference_threshold - df.CRPSS_native_vs_residual).clip(lower=0)
    # Ex post diagnostic, not usable to issue a forecast-time warning.
    ref = df[(df.date < EVAL_START) & df.past7_volatility.notna()]
    vol_p75 = float(ref.volatility.quantile(.75))
    df['high_realized_volatility_expost'] = df.volatility >= vol_p75
    # Forward-looking follow-up: ONLY for retrospective assessment, never alert construction.
    for c in ['CRPS_native','rMAE_native_vs_naive','CRPSS_native_vs_residual','PI80_native_coverage']:
        arr = df[c].to_numpy(dtype=float)
        next_mean, next_n = [], []
        for i in range(len(df)):
            chunk = arr[i+1:i+1+FOLLOWUP]
            next_mean.append(float(np.mean(chunk)) if len(chunk) == FOLLOWUP else np.nan)
            next_n.append(len(chunk))
        df[f'next7_mean_{c}'] = next_mean
    df['next7_fixed_alert_days'] = [int(df.alert_fixed.iloc[i+1:i+8].sum()) if i+8<=len(df) else np.nan for i in range(len(df))]
    df['next7_relative_alert_days'] = [int(df.alert_relative_either.iloc[i+1:i+8].sum()) if i+8<=len(df) else np.nan for i in range(len(df))]
    ev = df[df.date >= EVAL_START].copy()
    df.to_csv(TABLES/'experiment_14b_daily_diagnostics.csv',index=False)
    nine = ev[ev.category=='Fixed only'].copy()
    cols = ['date','CRPS_native','fixed_CRPS_threshold','fixed_exceedance',
            'fixed_exceedance_pct','market_expected_CRPS','market_CRPS_threshold',
            'market_residual','past7_volatility','volatility','high_realized_volatility_expost',
            'rMAE_native_vs_naive','CRPSS_native_vs_residual','PI80_native_coverage',
            'coverage_deviation_signed_pp','coverage_rolling7','coverage_rolling30',
            'coverage_rolling30_deviation_signed_pp','coverage_daily_sd30_pp',
            'coverage_rolling30_deviation_abs_pp_alert','coverage_daily_sd30_pp_alert',
            'alert_rMAE','alert_CRPSS','next7_mean_CRPS_native',
            'next7_mean_CRPSS_native_vs_residual','next7_mean_PI80_native_coverage',
            'next7_fixed_alert_days','next7_relative_alert_days']
    nine[cols].to_csv(TABLES/'experiment_14b_nine_fixed_only_days.csv',index=False)
    groups = ev.groupby('category',observed=True)
    summary = groups.agg(days=('date','size'),median_CRPS=('CRPS_native','median'),
                         median_fixed_exceedance=('fixed_exceedance','median'),
                         median_rMAE=('rMAE_native_vs_naive','median'),
                         median_CRPSS=('CRPSS_native_vs_residual','median'),
                         median_PI80_coverage=('PI80_native_coverage','median'),
                         median_past7_volatility=('past7_volatility','median'),
                         high_realized_volatility_pct=('high_realized_volatility_expost','mean'),
                         relative_alert_days=('alert_relative_either','sum'),
                         median_rolling30_coverage=('coverage_rolling30','median'),
                         median_rolling30_daily_sd_pp=('coverage_daily_sd30_pp','median'),
                         rolling30_deviation_alert_days=('coverage_rolling30_deviation_abs_pp_alert','sum'),
                         rolling30_variability_alert_days=('coverage_daily_sd30_pp_alert','sum')).reset_index()
    summary['high_realized_volatility_pct'] *= 100
    summary.to_csv(TABLES/'experiment_14b_category_summary.csv',index=False)
    # Plot 1: actual exceedance severity, date-wise.
    fig,ax=plt.subplots(figsize=(14,5))
    ax.scatter(ev.date,ev.fixed_exceedance,s=20,label='Fixed CRPS exceedance',alpha=.7)
    ax.scatter(ev.date,ev.market_exceedance,s=20,label='Market-aware CRPS exceedance',alpha=.7)
    ax.set(title='Experiment 14B — Magnitude beyond each CRPS alert threshold',
           xlabel='Italian calendar date',ylabel='Exceedance (€/MWh)')
    ax.grid(alpha=.2);ax.legend();save('experiment_14b_crps_alert_severity.png')
    # Plot 2: inspect nine removed days across the signals.
    fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
    if len(nine):
        labels=nine.date.dt.strftime('%Y-%m-%d')
        xx=np.arange(len(nine))
        axes[0].bar(xx,nine.fixed_exceedance,color='#db903e')
        axes[0].set(ylabel='Fixed CRPS exceedance (€/MWh)')
        axes[1].scatter(xx,nine.rMAE_native_vs_naive,color='#2b8b57',label='rMAE')
        axes[1].axhline(1,color='gray',ls='--',label='Equal MAE')
        axes[1].axhline(float(nine.rMAE_reference_threshold.iloc[0]),color='red',ls=':',label='rMAE alert limit')
        axes[1].set(ylabel='rMAE');axes[1].legend(fontsize=8)
        axes[2].scatter(xx,nine.CRPSS_native_vs_residual,color='#9158b8',label='CRPSS')
        axes[2].axhline(0,color='gray',ls='--',label='Equal CRPS')
        axes[2].axhline(float(nine.CRPSS_reference_threshold.iloc[0]),color='red',ls=':',label='CRPSS alert limit')
        axes[2].set(ylabel='CRPSS');axes[2].legend(fontsize=8)
        axes[-1].set_xticks(xx,labels,rotation=35,ha='right')
    fig.suptitle('Experiment 14B — Nine fixed-only alerts: severity and benchmark context')
    for ax in axes: ax.grid(alpha=.2)
    save('experiment_14b_nine_removed_alerts.png')
    # Plot 3: contextual retrospective market difficulty and CRPS.
    fig,ax=plt.subplots(figsize=(10,6))
    palette={'Neither alerts':'#8a9dab','Fixed only':'#e5a14c','Both alert':'#bb655e','Market-aware only':'#8e62c6'}
    for cat,part in groups:
        ax.scatter(part.past7_volatility,part.CRPS_native,s=24,alpha=.7,
                   label=f'{cat} (n={len(part)})',color=palette.get(cat))
    ax.set(xlabel='Mean realized daily price SD over previous seven days (€/MWh)',
           ylabel='Daily approximate CRPS (€/MWh)',
           title='Experiment 14B — Recent market volatility and forecasting error')
    ax.grid(alpha=.2);ax.legend();save('experiment_14b_volatility_vs_crps_by_alert.png')
    # Plot 4: future seven-day CRPS by category, retrospective descriptive only.
    fig,ax=plt.subplots(figsize=(10,5))
    order=['Neither alerts','Fixed only','Both alert']
    vals=[ev.loc[ev.category==cat,'next7_mean_CRPS_native'].dropna().to_numpy() for cat in order]
    ax.boxplot(
        vals,
        tick_labels=[f'{c}\n(n={len(v)})' for c, v in zip(order, vals)],
        showfliers=True
    )
    ax.set(title='Experiment 14B — CRPS in the seven days after each evaluated day',
           ylabel='Next-seven-day mean CRPS (€/MWh)')
    ax.grid(axis='y',alpha=.2);save('experiment_14b_followup_crps.png')
    # Plot 5: coverage trajectories. Rolling 7/30-day values are available only
    # after each day's observed prices have been recorded.
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ev.date, 100 * ev.PI80_native_coverage, '.', alpha=.23,
            markersize=4, label='Daily coverage')
    ax.plot(ev.date, 100 * ev.coverage_rolling7, linewidth=1.4, label='7-day rolling')
    ax.plot(ev.date, 100 * ev.coverage_rolling30, linewidth=2.1, label='30-day rolling')
    ax.axhline(80, color='black', linestyle='--', linewidth=1.2, label='Nominal 80%')
    ax.set(title='Experiment 14B — PI80 coverage and rolling calibration',
           xlabel='Italian calendar date', ylabel='Observed coverage (%)', ylim=(0, 105))
    ax.legend(ncol=4, fontsize=9); ax.grid(alpha=.2)
    save('experiment_14b_coverage_rolling.png')
    # Plot 6: signed calibration deviation and variability, with P95 reference lines.
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(ev.date, ev.coverage_rolling30_deviation_signed_pp, label='30-day signed deviation')
    axes[0].axhline(0, color='black', ls='--', lw=1, label='Target')
    t = coverage_limits['coverage_rolling30_deviation_abs_pp']
    axes[0].axhline(t, color='red', ls=':', label='Reference P95 absolute deviation')
    axes[0].axhline(-t, color='red', ls=':')
    axes[0].set(ylabel='Coverage deviation (percentage points)')
    axes[0].legend(fontsize=9); axes[0].grid(alpha=.2)
    axes[1].plot(ev.date, ev.coverage_daily_sd30_pp, label='30-day SD of daily coverage')
    axes[1].axhline(coverage_limits['coverage_daily_sd30_pp'], color='red', ls=':',
                    label='Reference P95 variability')
    axes[1].set(xlabel='Italian calendar date',
                ylabel='Daily coverage SD (percentage points)')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=.2)
    fig.suptitle('Experiment 14B — Calibration deviation versus temporal variability')
    save('experiment_14b_coverage_stability.png')
    # Plot 7: compare the nine fixed-only days with rolling coverage history.
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(ev.date, 100 * ev.coverage_rolling30, color='#3378a6',
            label='30-day rolling coverage')
    ax.axhline(80, ls='--', color='black', label='Nominal 80%')
    ax.scatter(nine.date, 100 * nine.coverage_rolling30, color='#d88635',
               s=65, zorder=4, label='Nine fixed-only days')
    ax.set(title='Experiment 14B — Rolling coverage at nine removed alert dates',
           xlabel='Italian calendar date', ylabel='30-day mean coverage (%)')
    ax.grid(alpha=.2); ax.legend()
    save('experiment_14b_nine_removed_coverage.png')
    coverage_summary = pd.DataFrame([
        {'metric': m, 'reference_p95': lim,
         'evaluation_alert_days': int(ev[f'{m}_alert'].sum()),
         'evaluation_valid_days': int(ev[m].notna().sum())}
        for m, lim in coverage_limits.items()
    ])
    coverage_summary.to_csv(TABLES/'experiment_14b_coverage_stability_summary.csv', index=False)
    lines=['# Experiment 14B — Alert severity and diagnostic evaluation','',
           '## Scope','Uses Experiment 14 daily metrics; no model retraining and no independent failure labels.',
           'All forward-seven-day summaries are **retrospective diagnostics**, never available at alert time.',
           'The current-day volatility column is also ex post and is not used for alert issuance.','',
           '## Definitions',
           '- Fixed severity = max(0, daily CRPS − fixed threshold), in €/MWh.',
           '- Market-aware severity = max(0, daily CRPS − market-aware threshold), in €/MWh.',
           '- Relative rMAE severity = max(0, rMAE − historical upper threshold).',
           '- Relative CRPSS severity = max(0, historical lower threshold − CRPSS).',
           '- High realized volatility = at or above reference-period 75th percentile of daily realized price SD; **ex-post diagnostic only**.',
           '- Seven-day follow-up excludes the alert day and requires all seven future days; last six evaluation dates have unavailable follow-up.',
           '- These are different severity scales and should not be directly compared.',
           '- PI80 nominal coverage is 80%; both systematic overcoverage and undercoverage matter.',
           '- Rolling 7/30-day coverage is the unweighted mean of daily coverage rates; DST dates may have 23/25 hours.',
           '- Rolling daily coverage variability is the sample SD of daily coverage percentages in the window.',
           '- Reference P95 thresholds are estimated only from available reference rolling-window observations.',
           '- High rolling variability and high absolute calibration deviation are different diagnostics.',
           '- No true/false-positive classification is made without independent deterioration labels.','',
           '## Category summary','','```\n'+summary.to_string(index=False,float_format=lambda x:f'{x:.3f}')+'\n```','',
           '## Nine fixed-only days','',
           f'Total: {len(nine)}. These are **not** automatically false alarms.',
           'See `tables/experiment_14b_nine_fixed_only_days.csv` for each date, CRPS severity, volatility, relative metrics, coverage, and follow-up.',
           '','## Figures',
           '- `experiment_14b_crps_alert_severity.png`',
           '- `experiment_14b_nine_removed_alerts.png`',
           '- `experiment_14b_volatility_vs_crps_by_alert.png`',
           '- `experiment_14b_followup_crps.png`',
           '- `experiment_14b_coverage_rolling.png`',
           '- `experiment_14b_coverage_stability.png`',
           '- `experiment_14b_nine_removed_coverage.png`','',
           '## Coverage stability thresholds and counts','',
           '```\n'+coverage_summary.to_string(index=False,float_format=lambda x:f'{x:.3f}')+'\n```','',
           '## Limitations','',
           '- Observed association is not causation; alert categories are partly defined by CRPS itself.',
           '- Seven-day follow-up is not independent ground truth, and overlapping windows create dependence.',
           '- Apparent persistence alone cannot establish model-specific deterioration.',
           '- Future evaluation should use prespecified, preferably independent events and compare alert episodes, lead time, missed events, and false alarms.']
    (REPORTS/'experiment_14b_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\nEXPERIMENT 14B — ALERT DIAGNOSTICS')
    print('Evaluation days:',len(ev),'| Fixed-only days:',len(nine))
    print('\nCATEGORY SUMMARY\n',summary.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nCOVERAGE STABILITY SUMMARY\n',coverage_summary.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nFIXED-ONLY DATES\n',nine[['date','CRPS_native','fixed_exceedance','rMAE_native_vs_naive','CRPSS_native_vs_residual','PI80_native_coverage','next7_mean_CRPS_native']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nTables:',TABLES,'\nFigures:',FIGURES,'\nReport:',REPORTS/'experiment_14b_report.md')

if __name__=='__main__':main()
