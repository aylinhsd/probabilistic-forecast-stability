
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "market_regime_results"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "ENTSOE_Probabilistic_LEAR_v3_predictions.csv")
df = df[df["Split"] == "Test"].copy()

df["Target_UTC"] = pd.to_datetime(df["Target_UTC"], utc=True)
df["date"] = df["Target_UTC"].dt.tz_convert("Europe/Rome").dt.date
df["width80"] = df["Q_0.900"] - df["Q_0.100"]

daily = df.groupby("date").agg(
    volatility=("Target_Price_EUR_MWh", "std"),
    width80=("width80", "mean")
).reset_index()

daily["date"] = pd.to_datetime(daily["date"])
daily["volatility_7d"] = daily["volatility"].rolling(7, min_periods=3).mean()
daily["width80_7d"] = daily["width80"].rolling(7, min_periods=3).mean()

periods = [
    ("Baseline", "2025-07-01", "2026-03-01"),
    ("March–April", "2026-03-01", "2026-05-01"),
    ("May–June", "2026-05-01", "2026-07-01"),
    ("July", "2026-07-01", "2026-08-01"),
    ("Aug–Oct", "2026-08-01", "2026-10-14")
]

holidays = [
    "2025-08-15", "2025-11-01", "2025-12-08",
    "2025-12-25", "2025-12-26", "2026-01-01",
    "2026-01-06", "2026-04-06", "2026-04-25",
    "2026-05-01", "2026-06-02", "2026-08-15"
]

fig, ax = plt.subplots(figsize=(17, 6))

ax.plot(
    daily["date"], daily["volatility"],
    color="gray", alpha=0.35, linewidth=0.9,
    label="Observed daily price volatility"
)
ax.plot(
    daily["date"], daily["volatility_7d"],
    color="tab:blue", linewidth=2,
    label="Observed volatility (7-day mean)"
)
ax.plot(
    daily["date"], daily["width80_7d"],
    color="tab:orange", linewidth=2,
    label="Probabilistic LEAR v3 PI80 width (7-day mean)"
)

for i, (name, start, end) in enumerate(periods):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if i % 2 == 0:
        ax.axvspan(start, end, color="gray", alpha=0.05)
    if i > 0:
        ax.axvline(start, color="black", linestyle=":", alpha=0.6)
    midpoint = start + (end - start) / 2
    ax.text(
        midpoint, 1.01, name,
        transform=ax.get_xaxis_transform(),
        ha="center", fontsize=9
    )

for day in holidays:
    date = pd.Timestamp(day)
    if daily["date"].min() <= date <= daily["date"].max():
        ax.axvline(date, color="red", linewidth=0.8, alpha=0.4)

ax.set_xlim(daily["date"].min(), daily["date"].max())
ax.set_ylabel("EUR/MWh")
ax.set_xlabel("Italian local date")
ax.set_title(
    "Observed Price Volatility and Probabilistic LEAR v3 Interval Width"
)
ax.grid(alpha=0.2)
ax.legend(loc="upper left")
fig.autofmt_xdate()
fig.tight_layout()

output = OUT / "daily_price_volatility_with_probabilistic_LEAR_v3.png"
fig.savefig(output, dpi=180)
plt.close(fig)

print("Saved:", output)
