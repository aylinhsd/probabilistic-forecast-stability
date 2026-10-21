
import pandas as pd

from pathlib import Path

folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / "ENTSOE_ML_dataset.csv")

print("\nDATASET SHAPE:")
print(df.shape)

print("\nFIRST 25 COLUMNS:")
print(df.columns[:25].tolist())

print("\nLAST 15 COLUMNS:")
print(df.columns[-15:].tolist())

print("\nFIRST 3 ROWS:")
print(df.iloc[:3, :12].to_string(index=False))

print("\nLAST 3 ROWS:")
print(df.iloc[-3:, :12].to_string(index=False))
