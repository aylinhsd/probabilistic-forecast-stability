
import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / "ENTSOE_ML_dataset.csv")

df["Target_UTC"] = pd.to_datetime(
    df["Target_UTC"], utc=True
)

df["Target_Date_Italy"] = pd.to_datetime(
    df["Target_Date_Italy"]
)

df = df.sort_values("Target_UTC")

print("\nTOTAL ROWS:", len(df))

print("\nMISSING PRICES:")
print(df["Target_Price_EUR_MWh"].isna().sum())

print("\nDUPLICATE UTC TIMESTAMPS:")
print(df["Target_UTC"].duplicated().sum())

print("\nTIME INTERVALS:")
print(df["Target_UTC"].diff().value_counts().head())

print("\nROWS PER ITALIAN DELIVERY DAY:")
print(
    df.groupby("Target_Date_Italy")
      .size()
      .value_counts()
      .sort_index()
)

print("\nFIRST 5 DAYS:")
print(
    df.groupby("Target_Date_Italy")
      .size()
      .head()
)

print("\nLAST 5 DAYS:")
print(
    df.groupby("Target_Date_Italy")
      .size()
      .tail()
)
