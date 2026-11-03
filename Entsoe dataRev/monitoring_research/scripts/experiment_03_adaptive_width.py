
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "monitoring_research"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"

for folder in (TABLES, FIGURES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    ROOT / "market_regime_results" / "health_baseline"
    / "daily_health_baseline.csv",
    parse_dates=["date"]
).sort_values("date").reset_index(drop=True)

if df["date"].duplicated().any():
    raise ValueError("Duplicate delivery dates")

cal = df[df["date"].between("2025-07-01", "2025-12-31")]
ev = df[df["date"].between("2026-01-01", "2026-10-13")].copy()

if len(cal) != 184 or len(ev) != 286:
    raise ValueError("Unexpected calibration/evaluation dates")

# Fixed thresholds: calibrated on July-December 2025.
fixed_width = cal["width80"].quantile(0.95)
crps_limit = cal["health_rolling_crps"].quantile(0.95)

# Adaptive threshold: previous 90 delivery days only.
# shift(1) excludes the current day's width.
df["adaptive_width_threshold"] = (
    df["width80"].shift(1)
    .rolling(window=90, min_periods=90)
    .quantile(0.95)
)

ev = df[df["date"].between("2026-01-01", "2026-10-13")].copy()
ev["fixed_threshold"] = fixed_width
ev["fixed_alert"] = ev["width80"] > ev["fixed_threshold"]
ev["adaptive_alert"] = (
    ev["width80"] > ev["adaptive_width_threshold"]
)
ev["crps_alert"] = ev["health_rolling_crps"] > crps_limit

if ev["adaptive_width_threshold"].isna().any():
    raise ValueError("Adaptive threshold has missing values")

crps_onsets = ev.loc[
    ev["crps_alert"] &
    ~ev["crps_alert"].shift(1, fill_value=False),
    "date"
]

if crps_onsets.empty:
    raise ValueError("No CRPS onset found")

first_onset = crps_onsets.iloc[0]

def summarize(method):
    alert = ev[f"{method}_alert"]
    starts = alert & ~alert.shift(1, fill_value=False)
    before = ev["date"] < first_onset
    previous_week = ev["date"].between(
        first_onset - pd.Timedelta(days=7),
        first_onset - pd.Timedelta(days=1)
    )

    prior_dates = ev.loc[before & alert, "date"]
    last_prior = prior_dates.max() if not prior_dates.empty else pd.NaT

    return {
        "method": method,
        "total_alert_days": int(alert.sum()),
        "alert_episodes": int(starts.sum()),
        "alert_days_before_onset": int((alert & before).sum()),
        "alert_days_in_prior_7_days": int(
            (alert & previous_week).sum()
        ),
        "last_prior_alert": (
            last_prior.date().isoformat()
            if pd.notna(last_prior) else ""
        ),
        "lead_days_from_last_prior_alert": (
            int((first_onset - last_prior).days)
            if pd.notna(last_prior) else None
        ),
    }

summary = pd.DataFrame([
    summarize("fixed"),
    summarize("adaptive")
])

daily_file = TABLES / "experiment_03_daily.csv"
summary_file = TABLES / "experiment_03_summary.csv"

ev[[
    "date", "width80", "fixed_threshold",
    "adaptive_width_threshold", "fixed_alert",
    "adaptive_alert", "health_rolling_crps", "crps_alert"
]].to_csv(daily_file, index=False)

summary.to_csv(summary_file, index=False)

# Full evaluation-period figure.
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(ev["date"], ev["width80"], label="Daily width", alpha=0.7)
ax.plot(
    ev["date"], ev["fixed_threshold"],
    "--", label="Fixed threshold"
)
ax.plot(
    ev["date"], ev["adaptive_width_threshold"],
    label="90-day adaptive threshold"
)
ax.axvline(
    first_onset, color="red", linestyle=":",
    label="First CRPS alarm"
)
ax.set(
    title="Fixed vs adaptive width thresholds",
    xlabel="Delivery date",
    ylabel="80% interval width (EUR/MWh)"
)
ax.legend()
ax.grid(alpha=0.2)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(
    FIGURES / "experiment_03_thresholds.png",
    dpi=300, bbox_inches="tight"
)
fig.savefig(
    FIGURES / "experiment_03_thresholds.pdf",
    bbox_inches="tight"
)
plt.close(fig)

# Focused March figure.
march = ev[ev["date"].between(
    "2026-02-15", "2026-03-20"
)]

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(march["date"], march["width80"], label="Daily width")
ax.plot(
    march["date"], march["fixed_threshold"],
    "--", label="Fixed threshold"
)
ax.plot(
    march["date"], march["adaptive_width_threshold"],
    label="90-day adaptive threshold"
)
ax.axvline(
    first_onset, color="red", linestyle=":",
    label="CRPS alarm"
)
ax.set(
    title="Warning thresholds around March 2026",
    xlabel="Delivery date",
    ylabel="80% interval width (EUR/MWh)"
)
ax.legend()
ax.grid(alpha=0.2)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(
    FIGURES / "experiment_03_march.png",
    dpi=300, bbox_inches="tight"
)
fig.savefig(
    FIGURES / "experiment_03_march.pdf",
    bbox_inches="tight"
)
plt.close(fig)

report = f"""# Experiment 03 — Fixed vs Adaptive Width Monitoring

Calibration: July-December 2025
Evaluation: January 1-October 13, 2026

Fixed width threshold: {fixed_width:.4f} EUR/MWh
Adaptive threshold: 95th percentile of the preceding 90 days
CRPS health threshold: {crps_limit:.4f} EUR/MWh
First CRPS health alarm: {first_onset.date()}

## Results

{summary.to_string(index=False)}

## Methodological notes

- The adaptive threshold excludes the current delivery day.
- Both methods use the same evaluation period.
- CRPS health alarms require realized prices.
- Width is forecast-side information, but operational
  timing must be verified against forecast issue times.
- Fewer alerts do not automatically mean better detection.
- Only one CRPS onset is available for this comparison.
- The adaptive threshold may normalize persistent changes.
- This is a retrospective exploratory comparison.
"""

(REPORTS / "experiment_03_report.md").write_text(report)

print("=== EXPERIMENT 03 RESULTS ===")
print("First CRPS onset:", first_onset.date())
print(summary.to_string(index=False))
print("\nSaved tables, figures and report.")
