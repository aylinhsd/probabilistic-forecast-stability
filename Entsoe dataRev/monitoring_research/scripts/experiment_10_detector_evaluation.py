#!/usr/bin/env python3
"""Experiment 10: retrospective, non-causal evaluation of Experiment 09 alarms.

Run on Buck from the existing monitoring_research/scripts directory.
Inputs are read-only: Experiment 09 daily signals/settings and the daily baseline.
No detector re-fitting, model training, or evaluation-period parameter tuning.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'monitoring_research'
TABLES, FIGURES, REPORTS = (OUT / x for x in ('tables', 'figures', 'reports'))
for directory in (TABLES, FIGURES, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)
BASELINE = ROOT / 'market_regime_results/health_baseline/daily_health_baseline.csv'
SIGNALS = TABLES / 'experiment_09_daily_detector_signals.csv'
SETTINGS = TABLES / 'experiment_09_settings.csv'
START, END = pd.Timestamp('2026-01-01'), pd.Timestamp('2026-10-13')
REFERENCE_END = pd.Timestamp('2025-12-31')
COOLDOWN_DAYS = 7
HORIZONS = (7, 14, 30)


def notification_dates(alarm_dates, cooldown_days=COOLDOWN_DAYS):
    """Send an event on first alarm, suppress later alarms for 7 calendar days.

    Does not create an ongoing alarm state; not equivalent to an alert episode.
    """
    selected = []
    for date in sorted(pd.to_datetime(alarm_dates).unique()):
        date = pd.Timestamp(date)
        if not selected or (date - selected[-1]).days >= cooldown_days:
            selected.append(date)
    return selected


def episodes(flag):
    arr = np.asarray(flag, dtype=bool)
    return int(np.sum(arr & ~np.r_[False, arr[:-1]]))


def future_outcome(daily, date, horizon, threshold):
    """Future *daily* CRPS exceedance, excluding the alert date itself."""
    future = daily.loc[(daily.date > date) & (daily.date <= date + pd.Timedelta(days=horizon))]
    # Right-censor: never treat incomplete future horizons as negative outcomes.
    if len(future) < horizon:
        return np.nan
    return int((future.crps > threshold).any())


def main():
    for file in (BASELINE, SIGNALS, SETTINGS):
        if not file.exists():
            raise SystemExit(f'Missing input: {file}\nRun Experiment 09 first if its tables are missing.')
    d = pd.read_csv(BASELINE, parse_dates=['date']).sort_values('date')
    d = d.loc[(d.date >= pd.Timestamp('2025-07-01')) & (d.date <= END)].copy()
    if d.date.duplicated().any() or d[['date', 'crps', 'width80']].isna().any().any():
        raise ValueError('Baseline has duplicated dates or missing required values')
    if len(d) != 470 or len(d.loc[d.date >= START]) != 286:
        raise ValueError('Unexpected daily baseline coverage; expected 470 days, 286 evaluation days')
    if not d.date.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError('Missing dates in daily baseline; future-horizon analysis requires contiguous days')
    t = pd.read_csv(SIGNALS, parse_dates=['date'])
    required = {'date', 'period', 'metric', 'detector', 'sensitivity', 'alarm'}
    if not required.issubset(t.columns):
        raise ValueError(f'Experiment 09 signals missing: {sorted(required - set(t.columns))}')
    if t.duplicated(['date', 'metric', 'detector', 'sensitivity']).any():
        raise ValueError('Duplicate detector configuration/date rows')
    if not pd.api.types.is_bool_dtype(t.alarm):
        t['alarm'] = t.alarm.astype(str).str.lower().map({'true': True, 'false': False})
        if t.alarm.isna().any():
            raise ValueError('Unrecognized alarm values')
    eval_days = d.loc[d.date >= START, ['date', 'crps', 'width80']].copy()
    ref = d.loc[d.date <= REFERENCE_END]
    crps_p95 = float(ref.crps.quantile(.95))
    rolling = d.crps.rolling(30, min_periods=15).mean()
    rolling_ref_p95 = float(rolling.loc[d.date <= REFERENCE_END].dropna().quantile(.95))
    d['rolling_crps30'] = rolling
    d['rolling_above_ref_p95'] = d.rolling_crps30 > rolling_ref_p95
    # First entry into an elevated rolling state, not independent verified degradation.
    rising = d.rolling_above_ref_p95 & ~d.rolling_above_ref_p95.shift(fill_value=False)
    rolling_onsets = d.loc[rising & (d.date >= START), 'date'].tolist()
    print(f'Experiment 10 | CRPS daily reference p95={crps_p95:.4f}; rolling30 reference p95={rolling_ref_p95:.4f}')
    print('Rolling-CRPS screening onsets:', [x.date().isoformat() for x in rolling_onsets])

    rows, notifications, future_rows = [], [], []
    configs = t[['metric', 'detector', 'sensitivity']].drop_duplicates().sort_values(['metric','detector','sensitivity'])
    for c in configs.itertuples(index=False):
        sub = t.loc[(t.metric == c.metric) & (t.detector == c.detector) &
                    (t.sensitivity == c.sensitivity) & (t.date >= START) & (t.date <= END)].sort_values('date')
        if len(sub) != len(eval_days) or not sub.date.reset_index(drop=True).equals(eval_days.date.reset_index(drop=True)):
            raise ValueError(f'Incomplete/misaligned evaluation dates for {c}')
        raw = sub.loc[sub.alarm, 'date'].tolist()
        notices = notification_dates(raw)
        alert_days = len(raw)
        row = dict(metric=c.metric, detector=c.detector, sensitivity=c.sensitivity,
                   evaluation_days=len(sub), alarm_days=alert_days,
                   raw_alarm_episodes=episodes(sub.alarm),
                   notifications_7d_cooldown=len(notices),
                   notification_rate_per_30d=len(notices)/len(sub)*30,
                   first_alarm=raw[0].date().isoformat() if raw else '',
                   first_notification=notices[0].date().isoformat() if notices else '')
        rows.append(row)
        for date in notices:
            next_onset = next((o for o in rolling_onsets if o > date), None)
            notifications.append(dict(metric=c.metric, detector=c.detector, sensitivity=c.sensitivity,
                                      notification_date=date.date().isoformat(),
                                      days_until_next_rolling_onset=(next_onset-date).days if next_onset is not None else np.nan,
                                      # Only descriptive: onsets are screening events, not ground truth.
                                      next_rolling_onset=next_onset.date().isoformat() if next_onset is not None else ''))
            if c.metric == 'width80':
                for horizon in HORIZONS:
                    future_rows.append(dict(detector=c.detector, sensitivity=c.sensitivity,
                                            notification_date=date.date().isoformat(), horizon_days=horizon,
                                            future_daily_crps_p95=future_outcome(d, date, horizon, crps_p95)))

    summary = pd.DataFrame(rows)
    notify = pd.DataFrame(notifications, columns=['metric','detector','sensitivity','notification_date',
                                                   'days_until_next_rolling_onset','next_rolling_onset'])
    outcomes = pd.DataFrame(future_rows, columns=['detector','sensitivity','notification_date','horizon_days','future_daily_crps_p95'])
    summary.to_csv(TABLES / 'experiment_10_notification_summary.csv', index=False)
    notify.to_csv(TABLES / 'experiment_10_notification_dates.csv', index=False)
    outcomes.to_csv(TABLES / 'experiment_10_width_future_outcomes.csv', index=False)

    # Descriptive benchmark: proportion of *all eligible evaluation days* followed
    # by >=1 daily CRPS exceedance, using exactly the same horizon and censoring.
    benchmark = []
    for h in HORIZONS:
        vals = [future_outcome(d, day, h, crps_p95) for day in eval_days.date]
        valid = pd.Series(vals).dropna()
        benchmark.append(dict(horizon_days=h, eligible_days=len(valid),
                              base_future_exceedance_rate=float(valid.mean()) if len(valid) else np.nan))
    benchmark = pd.DataFrame(benchmark)
    benchmark.to_csv(TABLES / 'experiment_10_future_base_rates.csv', index=False)
    if len(outcomes):
        group = outcomes.groupby(['detector','sensitivity','horizon_days'], as_index=False).agg(
            total_notifications=('notification_date','size'),
            eligible_notifications=('future_daily_crps_p95','count'),
            followed_by_future_exceedance=('future_daily_crps_p95','sum'),
            fraction_followed_by_exceedance=('future_daily_crps_p95','mean'))
        group = group.merge(benchmark, on='horizon_days', how='left')
        group['difference_from_all_day_base_rate'] = (group.fraction_followed_by_exceedance - group.base_future_exceedance_rate)
    else:
        group = pd.DataFrame(columns=['detector','sensitivity','horizon_days','total_notifications',
                                      'eligible_notifications','followed_by_future_exceedance',
                                      'fraction_followed_by_exceedance','base_future_exceedance_rate',
                                      'difference_from_all_day_base_rate'])
    group.to_csv(TABLES / 'experiment_10_width_future_summary.csv', index=False)

    # Figure 1: alarm counts versus notifications after common cooldown.
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    for ax, metric in zip(axes, ['crps', 'width80']):
        sub = summary.loc[summary.metric == metric].copy()
        sub['label'] = sub.detector + ' / ' + sub.sensitivity
        xx = np.arange(len(sub))
        ax.bar(xx - .2, sub.alarm_days, width=.4, label='Raw alarm days')
        ax.bar(xx + .2, sub.notifications_7d_cooldown, width=.4, label='Notifications (7-day cooldown)')
        ax.set_xticks(xx, sub.label, rotation=40, ha='right', fontsize=8)
        ax.set_ylabel('Count over evaluation period')
        ax.set_title(f'{metric.upper()}: raw alarms vs common notification policy')
        ax.legend(fontsize=8)
    for suffix in ('.png', '.pdf'):
        fig.savefig(FIGURES / f'experiment_10_alert_burden{suffix}', dpi=170)
    plt.close(fig)

    # Figure 2: width notifications relative to daily CRPS; no implication of causality.
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(d.date, d.crps, lw=1, color='0.4', label='Daily realized CRPS')
    ax.axhline(crps_p95, color='tab:orange', ls=':', label='Reference daily CRPS p95')
    ax.axvline(START, color='0.6', ls='--', label='Evaluation starts')
    palette = {'PageHinkley':'tab:blue', 'CUSUM':'tab:green', 'ADWIN':'tab:purple', 'FixedP95':'tab:red'}
    for detector in ['PageHinkley','CUSUM','ADWIN','FixedP95']:
        s = notify.loc[(notify.metric == 'width80') & (notify.detector == detector) &
                       (notify.sensitivity == ('reference' if detector == 'FixedP95' else 'medium'))]
        for dt in pd.to_datetime(s.notification_date):
            ax.axvline(dt, color=palette[detector], alpha=.23, lw=1)
        if len(s):
            ax.plot([], [], color=palette[detector], lw=2, label=f'{detector} width notifications')
    for i, onset in enumerate(rolling_onsets):
        ax.axvline(onset, color='black', ls='--', lw=1.5, label='Rolling CRPS screening onset' if i == 0 else None)
    ax.set_title('Width80 notifications versus subsequent CRPS (medium settings)')
    ax.set_ylabel('Daily CRPS')
    ax.set_xlabel('Italy-local delivery date')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    for suffix in ('.png', '.pdf'):
        fig.savefig(FIGURES / f'experiment_10_width_notifications_vs_crps{suffix}', dpi=170)
    plt.close(fig)

    report = f'''# Experiment 10 — Detector Evaluation and Early-Warning Assessment

## Objective
Compare the Experiment 09 detectors under a common **notification policy**, and retrospectively describe whether forecast-width notifications precede elevated future daily CRPS. No LEAR retraining or new detector fitting.

## Inputs and periods
- Daily baseline: `market_regime_results/health_baseline/daily_health_baseline.csv`.
- Existing Experiment 09 daily detector signals and settings.
- Reference: July 1–December 31, 2025 (184 days); evaluation: January 1–October 13, 2026 (286 days).
- Daily CRPS reference p95 = **{crps_p95:.4f}**; 30-day rolling CRPS reference p95 = **{rolling_ref_p95:.4f}**.

## Methods
1. Retain Experiment 09 alarm events exactly as saved, including their exploratory sensitivity settings. **Do not optimize settings on evaluation outcomes.**
2. Count raw alarm days and contiguous runs; note that event-based detectors and persistent fixed-threshold alarms are structurally different.
3. Apply a **common 7-calendar-day notification cooldown** to each detector/metric/configuration: keep first alarm, then suppress subsequent notifications until at least seven days after the previous notification. This is an illustrative operational rule, **not a calibrated optimal policy**. The underlying detector still runs continuously and is never reset by this notification policy.
4. Identify entries into the 30-day rolling CRPS above its reference p95. These are **screening onsets, not independent ground-truth degradation events**. Do not interpret their timing as a validated detector delay.
5. For **width80** notifications, examine whether *any* daily realized CRPS exceeds the reference **daily** p95 in the next 7, 14, or 30 calendar days, **excluding the notification date**. Require the entire future horizon; otherwise censor the observation. Compare with the same future-outcome prevalence among **all eligible evaluation days**.

## Critical limitations
- Alarm days, detector events, and notifications are different quantities. Seven-day cooldown reduces repeated notifications but does not establish statistical false-alarm control.
- **Future CRPS exceedance is an exploratory proxy, not independently labeled degradation**. It is also a relatively common event during 2026; high event-following fractions may arise from background prevalence.
- Notifications can cluster; their future windows overlap and observations are not independent. This analysis does **not** estimate calibrated predictive precision, recall, significance, or a causal warning lead time.
- Width80 can reflect legitimate market uncertainty; CRPS also responds to changes in market difficulty. Six months of reference cannot remove seasonal confounding.
- Daily width is a daily aggregate; this analysis does not establish hour-level forecast issue-time lead times.
- Sensitivity labels from Experiment 09 are historical script labels: `low` is the most responsive configuration and `high` is the most conservative. They should not be read as verified statistical sensitivity levels.
- The Page-Hinkley and ADWIN implementations have different internal reset behavior from the custom one-sided CUSUM and fixed threshold; this comparison does not calibrate equal false-alarm rates.

## Saved outputs
- `experiment_10_notification_summary.csv`: alarm-day and cooldown notification counts.
- `experiment_10_notification_dates.csv`: notification dates and distance to next rolling-CRPS screening onset.
- `experiment_10_width_future_outcomes.csv`: individual width notifications and complete-horizon outcomes.
- `experiment_10_width_future_summary.csv`: grouped future-outcome rates, with all-day baseline.
- `experiment_10_future_base_rates.csv`: reference future-outcome prevalence among all eligible evaluation days.
- `experiment_10_alert_burden.png/.pdf`: raw alarm days versus common notification policy.
- `experiment_10_width_notifications_vs_crps.png/.pdf`: notification dates relative to realized CRPS.

## Results and interpretation
Review the generated CSV tables and figures before drawing conclusions or selecting a detector. In particular, compare each width detector's future-exceedance fraction with the **all-day baseline rate** and consider the number of eligible notifications. Do not treat the first warning before March as a proven lead time.
'''
    (REPORTS / 'experiment_10_report.md').write_text(report, encoding='utf-8')
    print('\nNotification summary:')
    print(summary.to_string(index=False))
    print('\nWidth notifications followed by elevated daily CRPS (medium configurations):')
    print(group.loc[(group.sensitivity == 'medium') | (group.sensitivity == 'reference')].round(3).to_string(index=False))
    print('\nEXPERIMENT 10 — COMPLETE')
    print('Report:', REPORTS / 'experiment_10_report.md')

if __name__ == '__main__':
    main()
