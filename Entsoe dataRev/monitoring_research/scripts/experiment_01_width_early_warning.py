from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "monitoring_research"

INPUT = (
    ROOT
    / "market_regime_results"
    / "health_baseline"
    / "daily_health_baseline.csv"
)

TABLES = RESEARCH / "tables"
FIGURES = RESEARCH / "figures"
REPORTS = RESEARCH / "reports"

for folder in [TABLES, FIGURES, REPORTS]:
    folder.mkdir(parents=True, exist_ok=True)

CAL_START = pd.Timestamp("2025-07-01")
CAL_END = pd.Timestamp("2025-12-31")

EVAL_START = pd.Timestamp("2026-01-01")
EVAL_END = pd.Timestamp("2026-10-13")

WINDOW_START = pd.Timestamp("2026-02-15")
WINDOW_END = pd.Timestamp("2026-03-20")

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(INPUT, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

required = [
    "date",
    "width80",
    "health_rolling_crps",
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

cal = df[
    df["date"].between(CAL_START, CAL_END)
].copy()

evaluation = df[
    df["date"].between(EVAL_START, EVAL_END)
].copy()

# --------------------------------------------------
# Calibrate thresholds
# --------------------------------------------------

width_threshold = cal["width80"].quantile(0.95)

crps_threshold = (
    cal["health_rolling_crps"]
    .dropna()
    .quantile(0.95)
)

evaluation["width_alert"] = (
    evaluation["width80"] > width_threshold
)

evaluation["crps_alert"] = (
    evaluation["health_rolling_crps"] > crps_threshold
)

# --------------------------------------------------
# Find first CRPS alarm
# --------------------------------------------------

crps_alarm_dates = evaluation.loc[
    evaluation["crps_alert"], "date"
]

if crps_alarm_dates.empty:
    raise RuntimeError("No CRPS alarm detected.")

first_crps_alarm = crps_alarm_dates.min()

# --------------------------------------------------
# Examine forecast-side warnings
# --------------------------------------------------

before_alarm = evaluation[
    evaluation["date"] < first_crps_alarm
].copy()

width_alarm_dates = before_alarm.loc[
    before_alarm["width_alert"], "date"
]

if width_alarm_dates.empty:
    first_width_warning = pd.NaT
    last_width_warning = pd.NaT
    lead_days = None
else:
    first_width_warning = width_alarm_dates.min()
    last_width_warning = width_alarm_dates.max()
    lead_days = (
        first_crps_alarm - last_width_warning
    ).days

# --------------------------------------------------
# Save daily analysis table
# --------------------------------------------------

window = evaluation[
    evaluation["date"].between(WINDOW_START, WINDOW_END)
].copy()

window[
    [
        "date",
        "width80",
        "width_alert",
        "health_rolling_crps",
        "crps_alert",
    ]
].to_csv(
    TABLES / "experiment_01_daily.csv",
    index=False,
)

# --------------------------------------------------
# Save summary table
# --------------------------------------------------

summary = pd.DataFrame(
    [
        {
            "calibration_start": CAL_START.date(),
            "calibration_end": CAL_END.date(),
            "width_threshold": width_threshold,
            "crps_threshold": crps_threshold,
            "first_crps_alarm": first_crps_alarm.date(),
            "first_width_warning_before_crps":
                first_width_warning.date()
                if pd.notna(first_width_warning)
                else "",
            "last_width_warning_before_crps":
                last_width_warning.date()
                if pd.notna(last_width_warning)
                else "",
            "lead_days_from_last_warning": lead_days,
            "width_alert_days_before_crps":
                int(before_alarm["width_alert"].sum()),
        }
    ]
)

summary.to_csv(
    TABLES / "experiment_01_summary.csv",
    index=False,
)

# --------------------------------------------------
# Create figure
# --------------------------------------------------

fig, axes = plt.subplots(
    2, 1,
    figsize=(11, 7),
    sharex=True,
)

axes[0].plot(
    window["date"],
    window["width80"],
    color="tab:blue",
    linewidth=2,
    label="80% prediction interval width",
)

axes[0].axhline(
    width_threshold,
    color="tab:orange",
    linestyle="--",
    label="Historical 95th percentile",
)

axes[0].axvline(
    first_crps_alarm,
    color="tab:red",
    linestyle=":",
    label="First CRPS health alarm",
)

axes[0].set_ylabel("Width (EUR/MWh)")
axes[0].set_title(
    "Forecast interval width before CRPS deterioration"
)
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].plot(
    window["date"],
    window["health_rolling_crps"],
    color="tab:purple",
    linewidth=2,
    label="Rolling CRPS",
)

axes[1].axhline(
    crps_threshold,
    color="tab:orange",
    linestyle="--",
    label="Historical 95th percentile",
)

axes[1].axvline(
    first_crps_alarm,
    color="tab:red",
    linestyle=":",
    label="First CRPS health alarm",
)

axes[1].set_ylabel("CRPS (EUR/MWh)")
axes[1].set_xlabel("Delivery date")
axes[1].legend()
axes[1].grid(alpha=0.25)

fig.autofmt_xdate()
fig.tight_layout()

fig.savefig(
    FIGURES / "experiment_01_width_vs_crps.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    FIGURES / "experiment_01_width_vs_crps.pdf",
    bbox_inches="tight",
)

plt.close(fig)

# --------------------------------------------------
# Save experiment report
# --------------------------------------------------

report = f"""# Experiment 01 — Width Early Warning

## Objective

Investigate whether the 80% prediction interval width
crosses a historical threshold before the first rolling
CRPS health alarm.

## Calibration

Period: {CAL_START.date()} to {CAL_END.date()}

Width threshold (daily 95th percentile):
{width_threshold:.4f} EUR/MWh

CRPS threshold (rolling 95th percentile):
{crps_threshold:.4f} EUR/MWh

## Results

First CRPS alarm:
{first_crps_alarm.date()}

First earlier width alert:
{first_width_warning.date() if pd.notna(first_width_warning) else 'None'}

Last earlier width alert:
{last_width_warning.date() if pd.notna(last_width_warning) else 'None'}

Lead time from last earlier width alert:
{lead_days if lead_days is not None else 'Not available'} days

Width alert days before CRPS alarm:
{int(before_alarm['width_alert'].sum())}

## Interpretation

This experiment examines temporal ordering only.
An earlier width threshold crossing is not sufficient
evidence of predictive early-warning performance.

The width threshold is calibrated on daily forecast
interval widths, while the CRPS threshold is calibrated
on rolling health values.

Forecast-time availability of width must be verified
using forecast issue timestamps for a strict
operational early-warning evaluation.

## Limitations

- Retrospective evaluation.
- Only one first CRPS alarm onset examined.
- Thresholds are exploratory historical quantiles.
- Daily width may reflect changing market regimes.
- Alarm precision and false-positive rates are not
  estimated by this experiment.
"""

(REPORTS / "experiment_01_report.md").write_text(report)

print("=== EXPERIMENT 01 RESULTS ===")
print(summary.to_string(index=False))

print("\nSaved files:")
print(TABLES / "experiment_01_daily.csv")
print(TABLES / "experiment_01_summary.csv")
print(FIGURES / "experiment_01_width_vs_crps.png")
print(FIGURES / "experiment_01_width_vs_crps.pdf")
print(REPORTS / "experiment_01_report.md")