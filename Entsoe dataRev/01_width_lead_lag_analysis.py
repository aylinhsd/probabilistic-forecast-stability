
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / "market_regime_results"
    / "stability_monitoring"
    / "daily_stability_monitoring.csv"
)

df = pd.read_csv(SOURCE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

required = ["date", "width80", "crps", "coverage80"]
if df[required].isna().any().any():
    raise ValueError("Missing values in required columns")

if df["date"].duplicated().any():
    raise ValueError("Duplicate dates")

print("Observations:", len(df))
print("First date:", df["date"].min().date())
print("Last date:", df["date"].max().date())

# Exploratory lead-lag correlations.
# width80 at day t is compared with outcomes at day t + h.
# No future outcomes are used to calculate width80.

print("\nLEAD-LAG CORRELATIONS")
print("-" * 62)
print(
    f"{'Horizon':<12}"
    f"{'Width vs CRPS':>18}"
    f"{'Width vs noncoverage':>24}"
)

for horizon in [0, 1, 3, 7, 14]:
    future_crps = df["crps"].shift(-horizon)
    future_noncoverage = 1 - df["coverage80"].shift(-horizon)

    corr_crps = df["width80"].corr(future_crps)
    corr_noncoverage = df["width80"].corr(
        future_noncoverage
    )

    print(
        f"{horizon:<12}"
        f"{corr_crps:>18.4f}"
        f"{corr_noncoverage:>24.4f}"
    )

print("\nAnalysis completed.")
