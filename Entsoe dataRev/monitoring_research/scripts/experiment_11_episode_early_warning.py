#!/usr/bin/env python3
"""Experiment 11: sustained rolling-CRPS screening onsets and width warning lead times.

Read-only with respect to source data and previous experiments. Saves outputs under
monitoring_research/{tables,figures,reports}/experiment_11_*.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MON = ROOT / 'monitoring_research'
TABLES, FIGURES, REPORTS = (MON / n for n in ('tables', 'figures', 'reports'))
for folder in (TABLES, FIGURES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)
BASE = ROOT / 'market_regime_results/health_baseline/daily_health_baseline.csv'
NOTIF = TABLES / 'experiment_10_notification_dates.csv'
SIGNALS = TABLES / 'experiment_09_daily_detector_signals.csv'
REFERENCE_END = pd.Timestamp('2025-12-31')
EVALUATION_START = pd.Timestamp('2026-01-01')
HORIZONS = [7, 14, 30, 60]
PERSISTENCE = [1, 3, 5, 7]
PRIMARY_PERSISTENCE = 3


def onset_table(df, threshold, persistence):
    """Find runs >= persistence. Onset is first day of run, confirmed later.

    A run begins at the first consecutive observed calendar day above threshold.
    A run that crosses missing dates is split. Re-arming requires a below-threshold day.
    """
    x = df[['date', 'health_rolling_crps']].dropna().sort_values('date').copy()
    x['above'] = x['health_rolling_crps'] > threshold
    x['new_run'] = (~x['above'].shift(fill_value=False)) | (x['date'].diff().dt.days.ne(1))
    x['run_id'] = x['new_run'].cumsum()
    runs = []
    for _, group in x[x['above']].groupby('run_id'):
        if len(group) < persistence:
            continue
        onset = group['date'].iloc[0]
        if onset < EVALUATION_START:
            continue
        runs.append({'persistence_days': persistence, 'onset_date': onset,
                     'confirmation_date': group['date'].iloc[persistence - 1],
                     'run_end_date': group['date'].iloc[-1],
                     'run_length_days': len(group),
                     'rolling_crps_at_onset': group['health_rolling_crps'].iloc[0]})
    return runs


def main():
    for path in [BASE, NOTIF, SIGNALS]:
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}')
    daily = pd.read_csv(BASE, parse_dates=['date']).sort_values('date')
    notifications = pd.read_csv(NOTIF, parse_dates=['notification_date'])
    signals = pd.read_csv(SIGNALS, parse_dates=['date'])
    for name in ['date', 'crps', 'health_rolling_crps']:
        if name not in daily:
            raise ValueError(f'Missing daily column: {name}')
    required = {'metric', 'detector', 'sensitivity', 'notification_date'}
    if not required.issubset(notifications):
        raise ValueError(f'Missing notification columns: {required - set(notifications)}')
    if not {'date','period','metric','value','detector','sensitivity','alarm'}.issubset(signals):
        raise ValueError('Unexpected Experiment 09 signal structure')
    if daily['date'].duplicated().any():
        raise ValueError('Duplicate dates in daily baseline')
    cal = daily.loc[daily.date <= REFERENCE_END, 'health_rolling_crps'].dropna()
    if cal.empty:
        raise ValueError('No reference rolling CRPS observations')
    threshold = float(cal.quantile(.95))
    eval_daily = daily[daily.date >= EVALUATION_START].copy()
    eval_end = eval_daily.date.max()
    onset_rows = []
    for n in PERSISTENCE:
        onset_rows.extend(onset_table(daily, threshold, n))
    onsets = pd.DataFrame(onset_rows, columns=['persistence_days','onset_date','confirmation_date',
                                              'run_end_date','run_length_days','rolling_crps_at_onset'])
    onsets.to_csv(TABLES / 'experiment_11_rolling_onsets.csv', index=False)
    primary = onsets.loc[onsets.persistence_days == PRIMARY_PERSISTENCE].sort_values('onset_date')
    width = notifications.loc[(notifications.metric == 'width80') &
                              (notifications.notification_date >= EVALUATION_START) &
                              (notifications.notification_date <= eval_end)].copy()
    width = width.sort_values(['detector','sensitivity','notification_date'])
    records = []
    detail = []
    for (detector, sensitivity), g in width.groupby(['detector','sensitivity'], sort=True):
        dates = sorted(g.notification_date.unique())
        for _, ep in primary.iterrows():
            onset = ep.onset_date
            for h in HORIZONS:
                pre = [pd.Timestamp(d) for d in dates if onset - pd.Timedelta(days=h) <= d < onset]
                lead = [(onset - d).days for d in pre]
                records.append({'detector':detector,'sensitivity':sensitivity,
                                'onset_date':onset,'confirmation_date':ep.confirmation_date,
                                'horizon_days':h,'notifications_before_onset':len(pre),
                                'has_advance_warning':bool(pre),
                                'earliest_warning_date':min(pre) if pre else pd.NaT,
                                'earliest_warning_lead_days':max(lead) if lead else np.nan,
                                'latest_warning_date':max(pre) if pre else pd.NaT,
                                'latest_warning_lead_days':min(lead) if lead else np.nan})
        for d in dates:
            future = primary.loc[primary.onset_date > d, 'onset_date']
            next_onset = future.min() if not future.empty else pd.NaT
            previous = primary.loc[primary.onset_date <= d, 'onset_date']
            prior_onset = previous.max() if not previous.empty else pd.NaT
            detail.append({'detector':detector,'sensitivity':sensitivity,'notification_date':d,
                           'next_screening_onset':next_onset,
                           'days_until_next_onset':(next_onset-d).days if pd.notna(next_onset) else np.nan,
                           'most_recent_screening_onset':prior_onset,
                           'after_first_screening_onset':bool(not primary.empty and d >= primary.onset_date.min())})
    by_episode = pd.DataFrame(records)
    by_episode.to_csv(TABLES / 'experiment_11_pre_onset_windows.csv', index=False)
    details = pd.DataFrame(detail)
    details.to_csv(TABLES / 'experiment_11_width_notification_classification.csv', index=False)
    summary = []
    for (detector, sensitivity), g in width.groupby(['detector','sensitivity'], sort=True):
        first = primary.onset_date.min() if not primary.empty else pd.NaT
        dates = g.notification_date
        before = int((dates < first).sum()) if pd.notna(first) else 0
        after = int((dates >= first).sum()) if pd.notna(first) else 0
        row = {'detector':detector,'sensitivity':sensitivity,'total_notifications':len(g),
               'notifications_before_first_onset':before,
               'notifications_on_or_after_first_onset':after,
               'first_notification':dates.min(),
               'primary_onset_count':len(primary)}
        for h in HORIZONS:
            sub = by_episode[(by_episode.detector == detector) &
                             (by_episode.sensitivity == sensitivity) &
                             (by_episode.horizon_days == h)]
            row[f'onsets_warned_{h}d'] = int(sub.has_advance_warning.sum()) if len(sub) else 0
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(TABLES / 'experiment_11_detector_lead_summary.csv', index=False)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(daily.date, daily.health_rolling_crps, color='0.25', lw=1.6, label='30-day rolling CRPS')
    ax.axhline(threshold, color='tab:orange', ls=':', lw=2, label=f'Reference rolling p95 = {threshold:.3f}')
    ax.axvline(EVALUATION_START, color='0.55', ls='--', label='Evaluation starts')
    for i, ep in enumerate(primary.itertuples()):
        ax.axvline(ep.onset_date, color='tab:red', lw=1.6, ls='--', label='Screening onset' if i == 0 else None)
        ax.axvline(ep.confirmation_date, color='tab:red', lw=1, ls=':', label='Persistence confirmed' if i == 0 else None)
    ax.set(title='Experiment 11 — Rolling CRPS screening onsets', xlabel='Italy-local delivery date', ylabel='30-day rolling CRPS')
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(FIGURES / f'experiment_11_rolling_onsets.{ext}', dpi=170)
    plt.close(fig)

    medium = width[(width.sensitivity == 'medium') | ((width.detector == 'FixedP95') & (width.sensitivity == 'reference'))]
    colors = {'PageHinkley':'tab:blue','CUSUM':'tab:green','ADWIN':'tab:purple','FixedP95':'tab:red'}
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(daily.date, daily.health_rolling_crps, color='0.3', lw=1.4, label='30-day rolling CRPS')
    ax.axhline(threshold, color='tab:orange', ls=':', label='Reference rolling p95')
    ax.axvline(EVALUATION_START, color='0.55', ls='--', label='Evaluation starts')
    for detector, group in medium.groupby('detector'):
        for j, date in enumerate(group.notification_date):
            ax.axvline(date, color=colors.get(detector, '0.6'), alpha=.22, lw=1.0,
                       label=f'{detector} width notification' if j == 0 else None)
    for i, ep in enumerate(primary.itertuples()):
        ax.axvline(ep.onset_date, color='black', lw=2, ls='--', label='Screening onset' if i == 0 else None)
    ax.set(title='Experiment 11 — Width notifications and sustained rolling CRPS (medium settings)',
           xlabel='Italy-local delivery date', ylabel='30-day rolling CRPS')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(FIGURES / f'experiment_11_width_pre_onset_timeline.{ext}', dpi=170)
    plt.close(fig)

    lines = [
        '# Experiment 11 — Sustained Deterioration and Early-Warning Lead-Time Analysis', '',
        '## Purpose',
        'Evaluate whether forecast-width notifications precede a sustained rolling-CRPS screening onset, rather than merely precede an isolated high-error day.', '',
        '## Data and method',
        f'- Reference: dates through {REFERENCE_END.date()}; evaluation: {EVALUATION_START.date()} to {eval_end.date()}.',
        '- Daily CRPS and the previously calculated 30-day rolling CRPS are read from the saved health baseline.',
        '- Width notification dates come from Experiment 10, including its common seven-day cooldown.',
        f'- Reference rolling-CRPS 95th percentile: **{threshold:.6f}** (computed from nonmissing reference rolling observations).',
        f'- Primary persistence requirement: **{PRIMARY_PERSISTENCE} consecutive calendar days** above the reference threshold.',
        '- Screening onset is dated to the first above-threshold day of a qualifying run; this is known only when the persistence requirement is confirmed. Thus onset dating is retrospective, not a real-time alarm date.',
        '- Sensitivity checks use persistence requirements of 1, 3, 5, and 7 days.',
        '- Pre-onset windows: 7, 14, 30, and 60 days; notifications on the onset day are **not** advance warnings.',
        '- Detector settings are inherited from Experiment 09; they are not tuned using evaluation outcomes.', '',
        '## Results',
        f'- Primary qualifying screening runs: **{len(primary)}**.',
    ]
    for ep in primary.itertuples():
        lines.append(f'- Onset {ep.onset_date.date()}, confirmation {ep.confirmation_date.date()}, end {ep.run_end_date.date()}, length {ep.run_length_days} days.')
    if primary.empty:
        lines.append('- No primary screening onset identified; pre-onset success rates are undefined.')
    lines += ['', '### Medium-setting width notifications', '',
              '| Detector | Notifications | Before first onset | On/after onset | Onsets warned within 7d | 14d | 30d |',
              '|---|---:|---:|---:|---:|---:|---:|']
    if not summary.empty:
        selected = summary[(summary.sensitivity == 'medium') | ((summary.detector == 'FixedP95') & (summary.sensitivity == 'reference'))]
        for r in selected.itertuples():
            lines.append(f'| {r.detector} | {r.total_notifications} | {r.notifications_before_first_onset} | {r.notifications_on_or_after_first_onset} | {r.onsets_warned_7d} | {r.onsets_warned_14d} | {r.onsets_warned_30d} |')
    lines += ['', '## Interpretation and limitations',
              '- This is a **screening definition**, not independently labeled model failure or concept drift.',
              '- The reference period is only six months, and seasonality/market-regime changes may affect the threshold.',
              '- A long run can be one episode; it does not supply multiple independent deterioration onsets.',
              '- A warning before an onset is temporal association, not proof that width predicts deterioration.',
              '- Notifications during an already ongoing deterioration are not counted as advance warnings.',
              '- We cannot estimate meaningful detection sensitivity or false-positive rates from one onset.',
              '- A full-day width aggregate is usable only once the forecasts contributing to that day have been issued; issue-time availability must be checked before operational deployment.',
              '- The 7-day cooldown is an illustrative policy, not a calibrated optimal setting.', '',
              '## Outputs',
              '- Tables: experiment_11_rolling_onsets.csv, experiment_11_pre_onset_windows.csv, experiment_11_width_notification_classification.csv, experiment_11_detector_lead_summary.csv.',
              '- Figures: experiment_11_rolling_onsets and experiment_11_width_pre_onset_timeline (PNG/PDF).', '']
    (REPORTS / 'experiment_11_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('EXPERIMENT 11 — COMPLETE')
    print(f'Reference rolling CRPS p95: {threshold:.6f}')
    print('\nScreening onsets by persistence:')
    if len(onsets):
        print(onsets[['persistence_days','onset_date','confirmation_date','run_length_days']].to_string(index=False))
    else:
        print('None')
    print('\nWidth detector lead summary:')
    if len(summary):
        print(summary.to_string(index=False))
    else:
        print('No width notifications')
    print(f'\nReport: {REPORTS / "experiment_11_report.md"}')

if __name__ == '__main__':
    main()
