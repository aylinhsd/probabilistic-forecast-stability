"""Verify internal consistency of the ENTSO-E ML dataset (not publication timestamps)."""
from pathlib import Path
import pandas as pd
import numpy as np

folder = Path(__file__).resolve().parent
from pathlib import Path

path = Path(__file__).resolve().parent / "ENTSOE_ML_dataset_updated.csv"
df = pd.read_csv(path)
df['Target_UTC'] = pd.to_datetime(df['Target_UTC'], utc=True)
df['Issue_Time_UTC'] = pd.to_datetime(df['Issue_Time_UTC'], utc=True)
df['Input_Cutoff_UTC'] = pd.to_datetime(df['Input_Cutoff_UTC'], utc=True)
df['Delivery_Date_Italy'] = pd.to_datetime(df['Target_Date_Italy'])
price_lookup = df.set_index('Target_UTC')['Target_Price_EUR_MWh']
days = df.drop_duplicates('Delivery_Date_Italy').copy()
failures=[]
def check(name, ok):
    if ok: print('PASS:',name)
    else:
        print('FAIL:',name)
        failures.append(name)

# 1. Time-of-day issue schedule in Italian local time
issue_local = days['Issue_Time_UTC'].dt.tz_convert('Europe/Rome')
forecast_day = days['Delivery_Date_Italy'] - pd.Timedelta(days=1)
expected_time = np.where(forecast_day < pd.Timestamp('2015-02-10'), '08:30', '11:30')
check('Forecast issue times and February 10 transition', bool((issue_local.dt.strftime('%H:%M').to_numpy() == expected_time).all()))
check('Forecast issued on the previous Italian calendar day', bool((issue_local.dt.date.to_numpy() == forecast_day.dt.date.to_numpy()).all()))

# 2. Lag timestamps are before target-day boundary; verify all 168 values
cutoff_expected = days['Delivery_Date_Italy'].dt.tz_localize('Europe/Rome').dt.tz_convert('UTC')
check('Input boundary equals midnight at start of delivery day', bool((days['Input_Cutoff_UTC'] == cutoff_expected).all()))
lag_errors=0
for lag in range(1,169):
    column=f'Price_UTC_Lag_{lag}h'
    ts=days['Input_Cutoff_UTC'] - pd.Timedelta(hours=lag)
    expected=ts.map(price_lookup).to_numpy(dtype=float)
    observed=days[column].to_numpy(dtype=float)
    mismatches=np.isfinite(expected) & ~np.isclose(observed,expected,rtol=0,atol=1e-8,equal_nan=True)
    lag_errors+=int(mismatches.sum())
check('All verifiable hourly lag features match historical target prices',lag_errors==0)

# 3. Daily stats D-1 through D-7 independently recomputed from price targets
stats=df.groupby('Delivery_Date_Italy')['Target_Price_EUR_MWh'].agg(mean='mean',minimum='min',maximum='max',std='std')
stat_errors=0
for back in range(1,8):
    date=days['Delivery_Date_Italy']-pd.Timedelta(days=back)
    for stat in ['mean','minimum','maximum','std']:
        expected=date.map(stats[stat]).to_numpy(dtype=float)
        observed=days[f'{stat}_D-{back}'].to_numpy(dtype=float)
        stat_errors+=int((np.isfinite(expected) & ~np.isclose(observed,expected,rtol=0,atol=1e-8,equal_nan=True)).sum())
check('All verifiable daily statistic features match historical target prices',stat_errors==0)

# 4. For all delivery dates, earliest feature is prior to delivery-day midnight
check('No hourly lag refers to target delivery day',bool((days['Input_Cutoff_UTC']-pd.Timedelta(hours=1) < cutoff_expected).all()))

# 5. Display a real January 2015 example
example=days.loc[days['Delivery_Date_Italy']==pd.Timestamp('2015-01-13')].iloc[0]
lag13_ts=example['Input_Cutoff_UTC']-pd.Timedelta(hours=13)
print('\nJANUARY 13, 2015 EXAMPLE')
print('Forecast issued (Italy):',example['Issue_Time_UTC'].tz_convert('Europe/Rome'))
print('Lag 13 delivery time (Italy):',lag13_ts.tz_convert('Europe/Rome'))
print('Lag 13 value:',example['Price_UTC_Lag_13h'],'EUR/MWh')
print('Independent historical price:',price_lookup.loc[lag13_ts],'EUR/MWh')
print('January 12 average:',example['mean_D-1'],'EUR/MWh')
print('\nOVERALL:', 'ALL CHECKS PASSED' if not failures else 'FAILED: '+', '.join(failures))
if failures: raise SystemExit(1)
