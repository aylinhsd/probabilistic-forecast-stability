#!/usr/bin/env python3
"""Plot native probabilistic LEAR v3 forecasts against actual day-ahead prices.

Run from anywhere; input CSV is located next to this script.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'ENTSOE_Probabilistic_LEAR_v3_predictions.csv'
OUT = BASE / 'market_regime_results' / 'probabilistic_lear_v3_vs_actual'
OUT.mkdir(parents=True, exist_ok=True)

PERIODS = [
    ('baseline', 'Baseline: July 2025 – February 2026', '2025-07-01', '2026-03-01'),
    ('march_april', 'March–April 2026', '2026-03-01', '2026-05-01'),
    ('may_june', 'May–June 2026', '2026-05-01', '2026-07-01'),
    ('july', 'July 2026', '2026-07-01', '2026-08-01'),
    ('aug_oct', 'August–October 13, 2026', '2026-08-01', '2026-10-14'),
]

required = ['Target_UTC', 'Target_Price_EUR_MWh', 'Q_0.100', 'Q_0.500', 'Q_0.900', 'Split']
df = pd.read_csv(SOURCE)
missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f'Missing columns: {missing}')
df = df.loc[df['Split'].eq('Test')].copy()
df['time'] = pd.to_datetime(df['Target_UTC'], utc=True).dt.tz_convert('Europe/Rome')
df = df.sort_values('time')
if df['time'].duplicated().any():
    raise ValueError('Duplicate timestamps found')

for key, title, start, stop in PERIODS:
    subset = df.loc[(df['time'] >= pd.Timestamp(start, tz='Europe/Rome')) &
                    (df['time'] < pd.Timestamp(stop, tz='Europe/Rome'))].copy()
    if subset.empty:
        print(f'No observations for {key}; skipped')
        continue
    for frequency, view in [('hourly', subset), ('daily', None)]:
        if frequency == 'daily':
            # Daily means of quantile bounds are summaries of hourly forecast bands,
            # NOT calibrated prediction intervals for the daily average.
            view = (subset.set_index('time')[['Target_Price_EUR_MWh','Q_0.100','Q_0.500','Q_0.900']]
                    .resample('D').mean().dropna().reset_index())
        t = view['time'].dt.tz_localize(None)
        y = view['Target_Price_EUR_MWh'].to_numpy(float)
        med = view['Q_0.500'].to_numpy(float)
        lo = view['Q_0.100'].to_numpy(float)
        hi = view['Q_0.900'].to_numpy(float)
        fig, ax = plt.subplots(figsize=(15, 5.4))
        ax.fill_between(t, lo, hi, color='#70A9E5', alpha=.30, label='80% forecast interval (hourly quantiles)')
        ax.plot(t, y, color='#222222', linewidth=1.05 if frequency == 'hourly' else 1.7,
                label='Actual price', zorder=3)
        ax.plot(t, med, color='#1765B3', linewidth=1.0 if frequency == 'hourly' else 1.7,
                label='Probabilistic LEAR v3 median (Q0.50)', zorder=2)
        first_day = pd.Timestamp(start)
        last_day = pd.Timestamp(stop)
        for day in pd.date_range(first_day, last_day, freq='D', inclusive='left'):
            if day.weekday() >= 5:
                ax.axvspan(day, day + pd.Timedelta(days=1), color='gray', alpha=.055, lw=0)
        ax.set_title(f'{title} | {frequency.capitalize()} actual vs probabilistic LEAR v3')
        ax.set_ylabel('Electricity price (€/MWh)')
        ax.set_xlabel('Date (Italy local time)')
        ax.grid(alpha=.2)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=11))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend(loc='upper left', fontsize=9)
        if frequency == 'daily':
            ax.text(.99, .02, 'Daily means of hourly forecasts and bounds; shading is not a daily-mean prediction interval',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='#555555')
        fig.tight_layout()
        dest = OUT / f'{key}_{frequency}.png'
        fig.savefig(dest, dpi=160)
        plt.close(fig)
        print(f'Saved {dest} ({len(view)} {frequency} observations)')
print('Finished.')
