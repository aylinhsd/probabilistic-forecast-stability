"""Create thesis-ready plots plus one reproducibly random week per test month.
Run beside ENTSOE_LEAR_probabilistic_v1_predictions.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'ENTSOE_LEAR_probabilistic_v1_predictions.csv'
OUT = ROOT / 'LEAR_probabilistic_figures'
OUT.mkdir(exist_ok=True)
LEVELS = [80, 85, 90]
ACTUAL = 'Target_Price_EUR_MWh'
POINT = 'LEAR_v3_Prediction_EUR_MWh'

if not SOURCE.exists():
    raise FileNotFoundError(f'Place {SOURCE.name} beside this script. Expected: {SOURCE}')
df = pd.read_csv(SOURCE, low_memory=False)
needed = ['Target_Date_Italy', 'Split', ACTUAL, POINT] + [f'{side}_{level}_EUR_MWh' for level in LEVELS for side in ('Lower', 'Upper')]
missing = [c for c in needed if c not in df]
if missing:
    raise ValueError(f'Missing columns: {missing}')
df['date'] = pd.to_datetime(df['Target_Date_Italy'])
if 'Target_UTC' in df:
    df['Target_UTC'] = pd.to_datetime(df['Target_UTC'], utc=True)
    df = df.sort_values('Target_UTC')
else:
    df = df.sort_values('date')
test = df.loc[df['Split'].eq('Test')].copy()
if test.empty:
    raise ValueError('No Test rows found')
test['month'] = test['date'].dt.to_period('M').dt.to_timestamp()
for level in LEVELS:
    lo, hi = test[f'Lower_{level}_EUR_MWh'], test[f'Upper_{level}_EUR_MWh']
    y = test[ACTUAL]
    a = 1 - level / 100
    test[f'coverage_{level}'] = ((lo <= y) & (y <= hi)).astype(float)
    test[f'width_{level}'] = hi - lo
    test[f'score_{level}'] = (hi - lo) + (2 / a) * (lo - y).clip(lower=0) + (2 / a) * (y - hi).clip(lower=0)
monthly = test.groupby('month').mean(numeric_only=True)


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=220, bbox_inches='tight')
    plt.close()
    print('Saved:', OUT / name)

# 1. Monthly coverage and nominal reference lines
fig, ax = plt.subplots(figsize=(12, 5.5))
for level in LEVELS:
    ax.plot(monthly.index, 100 * monthly[f'coverage_{level}'], marker='o', markersize=3, label=f'Observed {level}%')
    ax.axhline(level, linestyle='--', linewidth=0.9, alpha=0.55)
ax.set(title='Monthly prediction interval coverage — test period', ylabel='Coverage (%)', xlabel='Delivery month')
ax.legend(ncol=3)
ax.grid(alpha=0.2)
save('01_monthly_coverage.png')

# 2. Monthly widths
fig, ax = plt.subplots(figsize=(12, 5.5))
for level in LEVELS:
    ax.plot(monthly.index, monthly[f'width_{level}'], marker='o', markersize=3, label=f'{level}% interval')
ax.set(title='Monthly mean interval width — test period', ylabel='Width (EUR/MWh)', xlabel='Delivery month')
ax.legend()
ax.grid(alpha=0.2)
save('02_monthly_width.png')

# 3. Monthly interval score
fig, ax = plt.subplots(figsize=(12, 5.5))
for level in LEVELS:
    ax.plot(monthly.index, monthly[f'score_{level}'], marker='o', markersize=3, label=f'{level}% interval')
ax.set(title='Monthly mean interval score (lower is better)', ylabel='Winkler interval score (EUR/MWh)', xlabel='Delivery month')
ax.legend()
ax.grid(alpha=0.2)
save('03_monthly_interval_score.png')

# 4. Monthly coverage gaps
fig, ax = plt.subplots(figsize=(12, 5.5))
for level in LEVELS:
    ax.plot(monthly.index, 100 * monthly[f'coverage_{level}'] - level, marker='o', markersize=3, label=f'{level}% interval')
ax.axhline(0, color='black', linewidth=1, linestyle='--')
ax.set(title='Monthly calibration gap — test period', ylabel='Observed − nominal (percentage points)', xlabel='Delivery month')
ax.legend()
ax.grid(alpha=0.2)
save('04_monthly_coverage_gap.png')

# 5. A complete week of forecasts, using UTC target timestamps for correct DST ordering
start = pd.Timestamp('2025-07-07')
week = test.loc[(test['date'] >= start) & (test['date'] < start + pd.Timedelta(days=7))].copy()
if week.empty:
    week = test.head(168).copy()
week = week.reset_index(drop=True)
x = np.arange(len(week))
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.fill_between(x, week['Lower_90_EUR_MWh'].to_numpy(), week['Upper_90_EUR_MWh'].to_numpy(), alpha=0.18, label='90% interval')
ax.fill_between(x, week['Lower_80_EUR_MWh'].to_numpy(), week['Upper_80_EUR_MWh'].to_numpy(), alpha=0.28, label='80% interval')
ax.plot(x, week[ACTUAL].to_numpy(), linewidth=1.2, label='Actual price')
ax.plot(x, week[POINT].to_numpy(), linewidth=1.1, label='LEAR v3 forecast')
ax.set(title=f'Actual prices, point forecasts and intervals — week of {week["date"].iloc[0].date()}', xlabel='Consecutive delivery hours', ylabel='Price (EUR/MWh)')
ax.legend(ncol=4)
ax.grid(alpha=0.2)
save('05_example_week.png')

# 6. Rolling 30-day coverage, trailing 720 hourly observations; DST affects day counts slightly
fig, ax = plt.subplots(figsize=(12, 5.5))
for level in LEVELS:
    roll = test[f'coverage_{level}'].rolling(window=720, min_periods=720).mean() * 100
    ax.plot(test['date'].to_numpy(), roll.to_numpy(), label=f'{level}% interval', linewidth=1.3)
    ax.axhline(level, linestyle='--', linewidth=0.8, alpha=0.55)
ax.set(title='Trailing 720-hour coverage (~30 days) — test period', ylabel='Coverage (%)', xlabel='Delivery date')
ax.legend(ncol=3)
ax.grid(alpha=0.2)
save('06_rolling_coverage.png')

# 7. One reproducibly random, complete 7-day week from EACH test month.
# Seed makes the random selection identical every time you run the script.
# The selected weeks never cross month boundaries.
if 'Target_UTC' not in test.columns:
    raise ValueError('Target_UTC is required for chronological weekly plots')

rng = np.random.default_rng(2026)
selected_weeks = []
for month, month_df in test.groupby(test['date'].dt.to_period('M'), sort=True):
    month_df = month_df.sort_values('Target_UTC').copy()
    dates = sorted(month_df['date'].dt.normalize().unique())
    available_dates = set(dates)
    candidates = []
    for start in dates:
        start = pd.Timestamp(start)
        expected = pd.date_range(start, periods=7, freq='D')
        if all(d in available_dates for d in expected):
            candidates.append(start)
    if not candidates:
        print(f'Skipping {month}: no complete 7-day period available')
        continue

    start = candidates[int(rng.integers(len(candidates)))]
    end = start + pd.Timedelta(days=7)
    week = month_df.loc[(month_df['date'] >= start) &
                        (month_df['date'] < end)].copy()
    if week.empty:
        continue

    # UTC timestamps retain the correct sequence across daylight-saving transitions.
    x = week['Target_UTC'].dt.tz_convert('Europe/Rome')
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.fill_between(x, week['Lower_90_EUR_MWh'].to_numpy(),
                    week['Upper_90_EUR_MWh'].to_numpy(),
                    alpha=0.18, label='90% prediction interval')
    ax.fill_between(x, week['Lower_80_EUR_MWh'].to_numpy(),
                    week['Upper_80_EUR_MWh'].to_numpy(),
                    alpha=0.28, label='80% prediction interval')
    ax.plot(x, week[ACTUAL].to_numpy(), linewidth=1.2, label='Actual price')
    ax.plot(x, week[POINT].to_numpy(), linewidth=1.1, label='LEAR v3 forecast')
    ax.set(title=f'{month}: randomly selected week {start.date()} to {(end - pd.Timedelta(days=1)).date()}',
           xlabel='Delivery time (Italy)', ylabel='Electricity price (EUR/MWh)')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1, tz=x.dt.tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b', tz=x.dt.tz))
    ax.legend(ncol=4)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    save(f'07_random_week_{month}.png')
    selected_weeks.append({'Month': str(month), 'Week_Start': start.date(),
                           'Week_End': (end - pd.Timedelta(days=1)).date(),
                           'Hourly_Rows': len(week)})

pd.DataFrame(selected_weeks).to_csv(OUT / 'random_weeks_selected.csv', index=False)
print(f'\nDONE: 6 overview figures + {len(selected_weeks)} random-week figures saved in {OUT}')
