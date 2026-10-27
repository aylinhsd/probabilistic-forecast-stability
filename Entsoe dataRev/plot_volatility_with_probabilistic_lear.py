from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'ENTSOE_Probabilistic_LEAR_v3_predictions.csv'
OUTPUT = ROOT / 'market_regime_results' / 'daily_price_volatility_three_periods_probabilistic_LEAR_v3.png'
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(SOURCE)
raw = raw.loc[raw['Split'].eq('Test')].copy()
raw['Target_UTC'] = pd.to_datetime(raw['Target_UTC'], utc=True)
raw = raw.sort_values('Target_UTC')
assert raw['Target_UTC'].is_unique, 'Duplicate target timestamps'
required = ['Target_Price_EUR_MWh', 'Q_0.050', 'Q_0.100', 'Q_0.250', 'Q_0.500', 'Q_0.750', 'Q_0.900', 'Q_0.950']
assert set(required).issubset(raw.columns), 'Required quantile columns missing'
raw['date'] = raw['Target_UTC'].dt.tz_convert('Europe/Rome').dt.tz_localize(None).dt.normalize()

# Daily mean of the hourly forecast quantiles, for visualization only.
# These are NOT calibrated quantiles of the daily mean price.
daily = raw.groupby('date').agg(
    volatility=('Target_Price_EUR_MWh', 'std'),
    actual=('Target_Price_EUR_MWh', 'mean'),
    q05=('Q_0.050', 'mean'), q10=('Q_0.100', 'mean'),
    q25=('Q_0.250', 'mean'), q50=('Q_0.500', 'mean'),
    q75=('Q_0.750', 'mean'), q90=('Q_0.900', 'mean'),
    q95=('Q_0.950', 'mean'),
    n=('Target_Price_EUR_MWh', 'size'),
).reset_index()
daily['rolling_sd'] = daily['volatility'].rolling(7, min_periods=1).mean()

periods = [
    ('Baseline', '2025-07-01', '2026-03-01', '#eaf3fa'),
    ('March–July', '2026-03-01', '2026-08-01', '#fff0ef'),
    ('August–October', '2026-08-01', '2026-10-14', '#f2eff9'),
]
holidays = {
    '2025-08-15': 'Assumption', '2025-11-01': 'All Saints',
    '2025-12-08': 'Immaculate Conception', '2025-12-25': 'Christmas',
    '2025-12-26': 'St Stephen', '2026-01-01': 'New Year',
    '2026-01-06': 'Epiphany', '2026-04-05': 'Easter',
    '2026-04-06': 'Easter Monday', '2026-04-25': 'Liberation Day',
    '2026-05-01': 'Labour Day', '2026-06-02': 'Republic Day',
    '2026-08-15': 'Assumption',
}

plt.rcParams.update({'font.size': 10})
fig, (ax_sd, ax_forecast) = plt.subplots(
    2, 1, figsize=(21, 12), sharex=True,
    gridspec_kw={'height_ratios': [1, 1.25], 'hspace': 0.25}
)
start, finish = daily['date'].min(), daily['date'].max()

for ax in (ax_sd, ax_forecast):
    for label, begin, end, color in periods:
        ax.axvspan(pd.Timestamp(begin), pd.Timestamp(end), color=color, alpha=0.72, zorder=0)
    # Shading weekends (Sat/Sun) over the period backgrounds.
    for day in pd.date_range(start, finish, freq='D'):
        if day.weekday() == 5:
            ax.axvspan(day, day + pd.Timedelta(days=2), color='#9ca3af', alpha=0.16, lw=0, zorder=0.5)
    for label, begin, end, color in periods[1:]:
        ax.axvline(pd.Timestamp(begin), color='#65748b', linewidth=1.15, zorder=2)
    for date_string in holidays:
        day = pd.Timestamp(date_string)
        if start <= day <= finish:
            ax.axvline(day, color='#db3a42', linewidth=0.8, alpha=0.85, zorder=2)
    ax.grid(axis='y', alpha=0.2)
    ax.set_xlim(start, finish + pd.Timedelta(days=1))

# Upper panel: faithful reproduction of the original observed volatility chart.
ax_sd.plot(daily['date'], daily['volatility'], color='#aab5c5', linewidth=1.05,
           alpha=0.95, label='Daily within-day price SD', zorder=3)
ax_sd.plot(daily['date'], daily['rolling_sd'], color='#1166a3', linewidth=2.4,
           label='7-day rolling mean of daily SD', zorder=4)
for label, begin, end, color in periods:
    p = daily.loc[(daily['date'] >= pd.Timestamp(begin)) & (daily['date'] < pd.Timestamp(end))]
    mean_sd = p['volatility'].mean()
    ax_sd.hlines(mean_sd, pd.Timestamp(begin), min(pd.Timestamp(end), finish),
                 color='#c7650c', linestyle='--', linewidth=2.1, zorder=4)
    midpoint = pd.Timestamp(begin) + (pd.Timestamp(end) - pd.Timestamp(begin)) / 2
    ax_sd.text(midpoint, 1.065, label, ha='center', va='bottom', fontweight='bold',
               transform=ax_sd.get_xaxis_transform(), fontsize=11)

# Names placed at top of volatility panel, staggered to limit overlap.
for i, (date_string, name) in enumerate(holidays.items()):
    day = pd.Timestamp(date_string)
    if start <= day <= finish:
        ax_sd.text(day, 0.94 - 0.13 * (i % 3), name, color='#ad2630',
                   fontsize=7.5, rotation=90, ha='right', va='top',
                   transform=ax_sd.get_xaxis_transform(), zorder=6)

ax_sd.set_ylabel('Within-day SD of actual hourly price (EUR/MWh)')
ax_sd.set_title('A. Observed electricity price volatility', fontsize=13, pad=36)
ax_sd.legend(handles=[
    Line2D([0], [0], color='#aab5c5', label='Daily within-day price SD'),
    Line2D([0], [0], color='#1166a3', lw=2.4, label='7-day rolling mean of daily SD'),
    Line2D([0], [0], color='#c7650c', lw=2, linestyle='--', label='Period average daily SD'),
    Patch(facecolor='#9ca3af', alpha=0.28, label='Weekends'),
    Line2D([0], [0], color='#db3a42', lw=1, label='Italian national holiday'),
], loc='upper left', ncol=3, fontsize=9)

# Lower panel: nested central 50%, 80%, 90% hourly forecast intervals,
# averaged by local day solely for visualization.
x = daily['date']
ax_forecast.fill_between(x, daily['q05'], daily['q95'], color='#c5dcf2', alpha=0.60,
                         label='90% interval (Q05–Q95)', zorder=2)
ax_forecast.fill_between(x, daily['q10'], daily['q90'], color='#7db3e1', alpha=0.55,
                         label='80% interval (Q10–Q90)', zorder=3)
ax_forecast.fill_between(x, daily['q25'], daily['q75'], color='#377bb7', alpha=0.45,
                         label='50% interval (Q25–Q75)', zorder=4)
ax_forecast.plot(x, daily['q50'], color='#125e9b', lw=1.7,
                 label='Probabilistic LEAR v3 median (Q50)', zorder=5)
ax_forecast.plot(x, daily['actual'], color='#151515', lw=1.4,
                 label='Observed daily mean price', zorder=6)
ax_forecast.set_ylabel('Electricity price (EUR/MWh)')
ax_forecast.set_title('B. Probabilistic LEAR v3 forecast distribution (daily display of hourly quantiles)', fontsize=13)
ax_forecast.legend(loc='upper left', ncol=2, fontsize=9)
ax_forecast.set_xlabel('Italian local delivery date')
ax_forecast.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax_forecast.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax_forecast.tick_params(axis='x', rotation=30)

fig.suptitle('Electricity price volatility and Probabilistic LEAR v3 — July 2025 to October 2026',
             fontsize=17, y=0.995)
fig.text(0.5, 0.016,
         'Daily volatility = sample SD of observed hourly prices in each Italian local day (not forecast error). '
         'Forecast bands are daily averages of hourly quantiles; they are not calibrated prediction intervals for daily mean price. '
         'Italian national holidays shown; local patron-saint holidays excluded.',
         ha='center', fontsize=9)
fig.subplots_adjust(left=0.07, right=0.985, top=0.89, bottom=0.095)
fig.savefig(OUTPUT, dpi=180, bbox_inches='tight')
plt.close(fig)
print('Saved:', OUTPUT)
print('Test rows:', len(raw), 'Daily rows:', len(daily))
for label, begin, end, color in periods:
    p = daily.loc[(daily['date'] >= pd.Timestamp(begin)) & (daily['date'] < pd.Timestamp(end))]
    print(f'{label}: {len(p)} days; mean daily SD={p["volatility"].mean():.3f} EUR/MWh')
