from pathlib import Path
import pandas as pd
import numpy as np


folder = Path(__file__).resolve().parent
input_file = folder / 'ENTSOE_ML_dataset.csv'
output_predictions = folder / 'ENTSOE_seasonal_naive_predictions.csv'
output_report = folder / 'ENTSOE_seasonal_naive_report.xlsx'

columns = ['Target_Date_Italy', 'Target_UTC', 'Target_Price_EUR_MWh', 'Split']
df = pd.read_csv(input_file, usecols=columns, parse_dates=['Target_Date_Italy'])
df['Target_UTC'] = pd.to_datetime(df['Target_UTC'], utc=True)
df['Target_Price_EUR_MWh'] = pd.to_numeric(df['Target_Price_EUR_MWh'], errors='raise')
df = df.sort_values('Target_UTC').reset_index(drop=True)
if df[columns].isna().any().any() or df['Target_UTC'].duplicated().any():
    raise ValueError('Missing values or duplicate UTC timestamps in target data.')

# Derive the Italian clock hour and occurrence within each delivery day.
local_time = df['Target_UTC'].dt.tz_convert('Europe/Rome')
df['Hour_Italy'] = local_time.dt.hour
# 0 = first occurrence of that clock hour, 1 = repeated autumn 02:00.
df['Hour_Occurrence'] = df.groupby(['Target_Date_Italy', 'Hour_Italy']).cumcount()

# Historical lookup from the observed target prices, using D-7.
lookup = df.set_index(['Target_Date_Italy', 'Hour_Italy', 'Hour_Occurrence'])['Target_Price_EUR_MWh']
history_date = df['Target_Date_Italy'] - pd.Timedelta(days=7)
keys = pd.MultiIndex.from_arrays([history_date, df['Hour_Italy'], df['Hour_Occurrence']])
pred = lookup.reindex(keys).to_numpy().copy()
method = np.full(len(df), 'same_hour_same_occurrence', dtype=object)

# If target hour is the second 02:00 but D-7 has only one, reuse the first 02:00.
missing = pd.isna(pred)
keys_first = pd.MultiIndex.from_arrays([history_date, df['Hour_Italy'], np.zeros(len(df), dtype=int)])
fallback_first = lookup.reindex(keys_first).to_numpy()
use_first = missing & pd.notna(fallback_first)
pred[use_first] = fallback_first[use_first]
method[use_first] = 'reuse_first_occurrence'

# If D-7 is the spring DST day and lacks 02:00, use its 03:00 price.
missing = pd.isna(pred)
keys_next = pd.MultiIndex.from_arrays([history_date, (df['Hour_Italy'] + 1) % 24, np.zeros(len(df), dtype=int)])
fallback_next = lookup.reindex(keys_next).to_numpy()
use_next = missing & (df['Hour_Italy'].to_numpy() == 2) & pd.notna(fallback_next)
pred[use_next] = fallback_next[use_next]
method[use_next] = 'spring_missing_02_use_03'

method[pd.isna(pred)] = 'no_historical_day_in_ml_file'
df['Predicted_Price_EUR_MWh'] = pred
df['Match_Method'] = method

# Validate no leakage: D-7 is strictly earlier than the permitted cutoff at D-1 00:00.
cutoff_utc = (df['Target_Date_Italy'] - pd.Timedelta(days=1)).dt.tz_localize('Europe/Rome').dt.tz_convert('UTC')
history_end_utc = (history_date + pd.Timedelta(days=1)).dt.tz_localize('Europe/Rome').dt.tz_convert('UTC')
if not (history_end_utc < cutoff_utc).all():
    raise ValueError('Historical reference violates information cutoff.')

valid = df[df['Predicted_Price_EUR_MWh'].notna()].copy()
valid['Error'] = valid['Target_Price_EUR_MWh'] - valid['Predicted_Price_EUR_MWh']
valid['Absolute_Error'] = valid['Error'].abs()
valid['Squared_Error'] = valid['Error'] ** 2

# Each day's errors are summarized separately to examine stability.
daily = valid.groupby(['Split', 'Target_Date_Italy'], as_index=False).agg(
    Hours=('Error', 'size'), MAE=('Absolute_Error', 'mean'), MSE=('Squared_Error', 'mean')
)
daily['RMSE'] = np.sqrt(daily['MSE'])

summary = valid.groupby('Split', as_index=False).agg(
    Hours=('Error', 'size'), Days=('Target_Date_Italy', 'nunique'),
    MAE_EUR_MWh=('Absolute_Error', 'mean'), MSE=('Squared_Error', 'mean')
)
summary['RMSE_EUR_MWh'] = np.sqrt(summary.pop('MSE'))
summary = summary[['Split', 'Hours', 'Days', 'MAE_EUR_MWh', 'RMSE_EUR_MWh']]

annual = valid.assign(Year=valid['Target_Date_Italy'].dt.year).groupby('Year', as_index=False).agg(
    Hours=('Error', 'size'), MAE_EUR_MWh=('Absolute_Error', 'mean'), MSE=('Squared_Error', 'mean')
)
annual['RMSE_EUR_MWh'] = np.sqrt(annual.pop('MSE'))

# Preserve all 23/24/25-hour days whenever D-7 exists in the ML file.
all_counts = df.groupby('Target_Date_Italy').size()
valid_counts = valid.groupby('Target_Date_Italy').size()
partial_days = valid_counts[valid_counts != all_counts.reindex(valid_counts.index)]
if len(partial_days):
    raise ValueError(f'Incomplete prediction days: {partial_days.index.tolist()}')

export_columns = ['Target_Date_Italy', 'Target_UTC', 'Hour_Italy', 'Hour_Occurrence',
                  'Split', 'Target_Price_EUR_MWh', 'Predicted_Price_EUR_MWh',
                  'Error', 'Absolute_Error', 'Match_Method']
valid[export_columns].to_csv(output_predictions, index=False)
with pd.ExcelWriter(output_report, engine='openpyxl') as writer:
    summary.to_excel(writer, sheet_name='Split Metrics', index=False)
    annual.to_excel(writer, sheet_name='Annual Metrics', index=False)
    daily.drop(columns='MSE').to_excel(writer, sheet_name='Daily Metrics', index=False)
    df['Match_Method'].value_counts().rename_axis('Method').reset_index(name='Hours').to_excel(
        writer, sheet_name='DST and Coverage', index=False)
    df.loc[df['Predicted_Price_EUR_MWh'].isna(), ['Target_Date_Italy']].drop_duplicates().to_excel(
        writer, sheet_name='Excluded Days', index=False)

print('\nSEASONAL NAIVE BASELINE (D-7)')
print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print('\nDST and coverage:')
print(df['Match_Method'].value_counts().to_string())
print('\nExcluded target days:', df.loc[df['Predicted_Price_EUR_MWh'].isna(), 'Target_Date_Italy'].nunique())
print('Saved:', output_predictions)
print('Saved:', output_report)
