#!/usr/bin/env python3
"""Experiment 09: chronological comparison of change detectors for LEAR.

Read-only inputs: daily_health_baseline.csv. Outputs are saved under monitoring_research.
Requires: numpy, pandas, matplotlib, river (pip install river).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from river import drift
except ImportError as exc:
    raise SystemExit('River is required: ~/LEAR/.venv/bin/python -m pip install river') from exc

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'market_regime_results/health_baseline/daily_health_baseline.csv'
OUT = ROOT / 'monitoring_research'
for folder in ('tables', 'figures', 'reports'):
    (OUT / folder).mkdir(parents=True, exist_ok=True)
REFERENCE_END = pd.Timestamp('2025-12-31')
EVAL_START = pd.Timestamp('2026-01-01')
EVAL_END = pd.Timestamp('2026-10-13')
METRICS = {'crps': 'Realized CRPS', 'width80': 'Forecast interval width 80%'}
# These are exploratory sensitivity settings, not optimized parameters.
CONFIGS = {
    'PageHinkley': [('low', {'delta': 0.005, 'threshold': 25.0, 'alpha': 0.999, 'min_instances': 30}),
                    ('medium', {'delta': 0.005, 'threshold': 50.0, 'alpha': 0.999, 'min_instances': 30}),
                    ('high', {'delta': 0.005, 'threshold': 100.0, 'alpha': 0.999, 'min_instances': 30})],
    'ADWIN': [('low', {'delta': 0.01}), ('medium', {'delta': 0.002}), ('high', {'delta': 0.0005})],
    'CUSUM': [('low', {'k_sigma': 0.25, 'h_sigma': 3.0}),
              ('medium', {'k_sigma': 0.5, 'h_sigma': 5.0}),
              ('high', {'k_sigma': 0.75, 'h_sigma': 8.0})],
}

class OneSidedCUSUM:
    """Upper-sided reference-standardized CUSUM, reset after alarm."""
    def __init__(self, mean, std, k_sigma, h_sigma):
        self.mean, self.std = float(mean), max(float(std), 1e-8)
        self.k, self.h = k_sigma, h_sigma
        self.stat = 0.0
    def update(self, x):
        self.stat = max(0.0, self.stat + (x - self.mean) / self.std - self.k)
        alarm = self.stat > self.h
        if alarm:
            self.stat = 0.0
        return alarm


def main():
    if not INPUT.exists():
        raise SystemExit(f'Missing input: {INPUT}')
    df = pd.read_csv(INPUT, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    df = df.loc[(df.date >= pd.Timestamp('2025-07-01')) & (df.date <= EVAL_END)].copy()
    if df.date.duplicated().any():
        raise ValueError('Duplicate dates in daily baseline')
    ref = df.loc[df.date <= REFERENCE_END]
    eva = df.loc[df.date >= EVAL_START]
    if len(ref) < 100 or len(eva) < 100:
        raise ValueError('Unexpected reference/evaluation sample sizes')
    print(f'Experiment 09 | reference {len(ref)} days | evaluation {len(eva)} days')
    alarms, settings, timelines = [], [], []
    for metric, metric_label in METRICS.items():
        if df[metric].isna().any():
            raise ValueError(f'Missing {metric} values')
        mu = float(ref[metric].mean())
        sd = float(ref[metric].std(ddof=1))
        q95 = float(ref[metric].quantile(.95))
        print(f'  {metric}: reference mean={mu:.4f}, sd={sd:.4f}, p95={q95:.4f}')
        # Fixed threshold is a daily p95 reference screening rule (not rolling threshold).
        for _, row in df.iterrows():
            timelines.append(dict(date=row.date, period='reference' if row.date <= REFERENCE_END else 'evaluation',
                                  metric=metric, value=float(row[metric]), detector='FixedP95',
                                  sensitivity='reference', alarm=bool(row[metric] > q95)))
        settings.append(dict(metric=metric, detector='FixedP95', sensitivity='reference',
                             parameters=json.dumps({'reference_p95': q95})))
        for family, variants in CONFIGS.items():
            for sensitivity, params in variants:
                settings.append(dict(metric=metric, detector=family, sensitivity=sensitivity,
                                     parameters=json.dumps(params, sort_keys=True)))
                if family == 'PageHinkley':
                    detector = drift.PageHinkley(**params)
                elif family == 'ADWIN':
                    detector = drift.ADWIN(**params)
                else:
                    detector = OneSidedCUSUM(mu, sd, **params)
                # All detectors receive the reference period sequentially as a warm-up.
                # Alarms are recorded there too, but only evaluation alarms are compared.
                for _, row in df.iterrows():
                    x = float(row[metric])
                    if family == 'CUSUM':
                        fired = detector.update(x)
                    else:
                        detector.update(x)
                        fired = bool(detector.drift_detected)
                    timelines.append(dict(date=row.date,
                                          period='reference' if row.date <= REFERENCE_END else 'evaluation',
                                          metric=metric, value=x, detector=family,
                                          sensitivity=sensitivity, alarm=fired))
    t = pd.DataFrame(timelines)
    t.to_csv(OUT / 'tables/experiment_09_daily_detector_signals.csv', index=False)
    pd.DataFrame(settings).to_csv(OUT / 'tables/experiment_09_settings.csv', index=False)
    for (metric, detector, sensitivity), sub in t.groupby(['metric', 'detector', 'sensitivity'], sort=True):
        ev = sub.loc[sub.period == 'evaluation'].sort_values('date')
        fired = ev.loc[ev.alarm, 'date']
        # Number of contiguous alert runs; detector reset behavior differs by method.
        flag = ev.alarm.to_numpy(dtype=bool)
        episodes = int(np.sum(flag & ~np.r_[False, flag[:-1]]))
        alarms.append(dict(metric=metric, detector=detector, sensitivity=sensitivity,
                           evaluation_days=len(ev), alarm_days=len(fired),
                           alert_episodes=episodes,
                           first_evaluation_alarm=fired.min().date().isoformat() if len(fired) else '',
                           last_evaluation_alarm=fired.max().date().isoformat() if len(fired) else ''))
    summary = pd.DataFrame(alarms).sort_values(['metric','detector','sensitivity'])
    summary.to_csv(OUT / 'tables/experiment_09_detector_summary.csv', index=False)
    # A single timeline per metric; medium setting for detectors, fixed daily p95 baseline.
    for metric, label in METRICS.items():
        fig, ax = plt.subplots(figsize=(13, 5))
        base = df[['date', metric]].copy()
        ax.plot(base.date, base[metric], color='0.3', linewidth=1.0, label=label)
        ax.axvline(EVAL_START, linestyle='--', color='0.55', label='Evaluation starts')
        ax.axhline(float(ref[metric].quantile(.95)), linestyle=':', color='tab:orange', label='Reference daily p95')
        for family, marker in [('PageHinkley','o'), ('CUSUM','^'), ('ADWIN','s')]:
            ss = t.loc[(t.metric == metric) & (t.detector == family) &
                       (t.sensitivity == 'medium') & (t.period == 'evaluation') & t.alarm]
            if len(ss):
                ax.scatter(ss.date, ss.value, marker=marker, s=50, label=f'{family} alarm', zorder=4)
        ax.set_title(f'Experiment 09 — {label}: detector comparison (medium settings)')
        ax.set_ylabel(label)
        ax.set_xlabel('Italy-local delivery date')
        ax.legend(loc='best', fontsize=8)
        fig.tight_layout()
        stem = OUT / 'figures' / f'experiment_09_{metric}_detector_timeline'
        fig.savefig(stem.with_suffix('.png'), dpi=170)
        fig.savefig(stem.with_suffix('.pdf'))
        plt.close(fig)
    report = f'''# Experiment 09 — Comparative Change-Detection Methods

## Purpose
Compare fixed daily reference thresholds, River Page-Hinkley, a one-sided standardized CUSUM, and River ADWIN on LEAR daily CRPS and daily 80% interval width. This is monitoring research, not model retraining.

## Data and chronology
- Reference: 2025-07-01 through 2025-12-31 ({len(ref)} days).
- Evaluation: 2026-01-01 through 2026-10-13 ({len(eva)} days).
- Input: `market_regime_results/health_baseline/daily_health_baseline.csv`.
- All detectors process dates chronologically. River detectors warm up on reference observations before evaluation; the CUSUM mean and standard deviation and fixed p95 thresholds are estimated from reference observations only.
- River Page-Hinkley uses its own running-mean formulation; CUSUM is a distinct reference-standardized, upper-sided algorithm. ADWIN is a two-sided adaptive-window detector.

## Detector settings
- Page-Hinkley: `delta=0.005`, `alpha=0.999`, `min_instances=30`; thresholds 25, 50, 100.
- ADWIN: delta 0.01, 0.002, 0.0005.
- Upper CUSUM: standardized reference mean, allowance k=0.25/0.5/0.75 standard deviations and alarm h=3/5/8; reset after each alarm.
- Fixed threshold: reference **daily** 95th percentile. It is not the 30-day rolling threshold from Experiment 01.
- All configurations are exploratory sensitivity settings, **not optimized or calibrated to a specified false-alarm rate**. Page-Hinkley and ADWIN may signal decreases as well as increases depending on implementation; only upper CUSUM is explicitly one-sided.

## Outputs
- `experiment_09_daily_detector_signals.csv`: every day and configuration, including warm-up alarms.
- `experiment_09_detector_summary.csv`: evaluation alarm-day counts, contiguous alert episodes, and first/last alarms.
- `experiment_09_settings.csv`: detector parameters.
- `experiment_09_crps_detector_timeline.png/.pdf` and `experiment_09_width80_detector_timeline.png/.pdf`.

## Interpretation cautions
- Daily CRPS requires realized outcomes; daily width uses forecast information, but the daily aggregate is fully available only after all forecasts included in that delivery date have been issued. This analysis does **not** establish exact intraday lead time.
- Different detectors have different reset behavior. Alarm counts are not directly comparable measures of false-positive rates.
- There is no independently verified ground-truth degradation onset. The March 11, 2026 rolling-CRPS threshold crossing from Experiment 01 is a prior monitoring alarm, not ground truth.
- Seasonal differences and market regimes may produce alarms without model degradation. Six months of reference data cannot fully establish annual seasonality.
- Do not infer statistical significance, validated early warning, or superiority from one evaluation period.

## Results
See `experiment_09_detector_summary.csv` and the two timelines. Conclusions should be written after inspecting the actual outputs.
'''
    (OUT / 'reports/experiment_09_report.md').write_text(report, encoding='utf-8')
    print('\nEvaluation detector summary:')
    print(summary.to_string(index=False))
    print('\nEXPERIMENT 09 — COMPLETE')
    print('Report:', OUT / 'reports/experiment_09_report.md')

if __name__ == '__main__':
    main()
