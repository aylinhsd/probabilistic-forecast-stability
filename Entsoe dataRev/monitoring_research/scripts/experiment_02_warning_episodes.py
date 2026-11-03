from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "monitoring_research"

INPUT = (
    ROOT / "market_regime_results"
    / "health_baseline"
    / "daily_health_baseline.csv"
)

TABLES = RESEARCH / "tables"
FIGURES = RESEARCH / "figures"
REPORTS = RESEARCH / "reports"

for folder in (TABLES, FIGURES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

CAL_START = pd.Timestamp("2025-07-01")
CAL_END = pd.Timestamp("2025-12-31")
EVAL_START = pd.Timestamp("2026-01-01")
EVAL_END = pd.Timestamp("2026-10-13")

HORIZONS = [1, 3, 7, 14]

df = pd.read_csv(INPUT, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

required = ["date", "width80", "health_rolling_crps"]
missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Missing columns: {missing}")

if df["date"].duplicated().any():
    raise ValueError("Duplicate dates found.")

cal = df[df["date"].between(CAL_START, CAL_END)]
ev = df[df["date"].between(EVAL_START, EVAL_END)].copy()
ev = ev.reset_index(drop=True)

if len(cal) != 184 or len(ev) != 286:
    raise ValueError("Unexpected calibration or evaluation length.")

width_threshold = cal["width80"].quantile(0.95)
crps_threshold = cal["health_rolling_crps"].quantile(0.95)

ev["width_alert"] = ev["width80"] > width_threshold
ev["crps_alert"] = ev["health_rolling_crps"] > crps_threshold

# First day of each new CRPS alarm episode
ev["crps_onset"] = (
    ev["crps_alert"]
    & ~ev["crps_alert"].shift(1, fill_value=False)
)

# First day of each new width-alert episode
ev["width_onset"] = (
    ev["width_alert"]
    & ~ev["width_alert"].shift(1, fill_value=False)
)

# Group consecutive width-alert days into episodes
episode_number = ev["width_onset"].cumsum()
ev["width_episode_id"] = episode_number.where(
    ev["width_alert"]
)

health_onsets = ev.loc[ev["crps_onset"], "date"].tolist()

episodes = []

for episode_id, group in ev.loc[
    ev["width_alert"]
].groupby("width_episode_id"):

    start = group["date"].min()
    end = group["date"].max()

    record = {
        "episode_id": int(episode_id),
        "start": start,
        "end": end,
        "duration_days": len(group),
        "max_width": group["width80"].max(),
    }

    # Evaluate warning at episode onset, not at every alert day.
    for horizon in HORIZONS:
        future_onsets = [
            d for d in health_onsets
            if start < d <= start + pd.Timedelta(days=horizon)
        ]
        record[f"onset_within_{horizon}d"] = bool(future_onsets)

    episodes.append(record)

episode_df = pd.DataFrame(episodes)

episode_df.to_csv(
    TABLES / "experiment_02_width_episodes.csv",
    index=False
)

ev[
    [
        "date",
        "width80",
        "health_rolling_crps",
        "width_alert",
        "width_onset",
        "width_episode_id",
        "crps_alert",
        "crps_onset",
    ]
].to_csv(
    TABLES / "experiment_02_daily.csv",
    index=False
)

summary_rows = []

for horizon in HORIZONS:
    hits = int(
        episode_df[f"onset_within_{horizon}d"].sum()
    ) if not episode_df.empty else 0

    summary_rows.append({
        "horizon_days": horizon,
        "width_episodes": len(episode_df),
        "episodes_followed_by_crps_onset": hits,
        "episode_hit_fraction":
            hits / len(episode_df)
            if len(episode_df) else float("nan"),
        "crps_onsets": len(health_onsets),
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    TABLES / "experiment_02_summary.csv",
    index=False
)

# Figure: warning episodes and CRPS health onsets
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    ev["date"],
    ev["width80"],
    label="80% interval width",
    color="tab:blue",
    linewidth=1.5
)

ax.axhline(
    width_threshold,
    linestyle="--",
    color="tab:orange",
    label="Calibration width threshold"
)

for i, row in episode_df.iterrows():
    ax.axvspan(
        row["start"],
        row["end"] + pd.Timedelta(days=1),
        color="tab:blue",
        alpha=0.12,
        label="Width-alert episode" if i == 0 else None
    )

for i, onset in enumerate(health_onsets):
    ax.axvline(
        onset,
        color="tab:red",
        linestyle=":",
        linewidth=2,
        label="CRPS health onset" if i == 0 else None
    )

ax.set(
    title="Width-alert episodes and CRPS health-alarm onset",
    xlabel="Delivery date",
    ylabel="80% interval width (EUR/MWh)"
)
ax.legend()
ax.grid(alpha=0.2)

fig.autofmt_xdate()
fig.tight_layout()

fig.savefig(
    FIGURES / "experiment_02_warning_episodes.png",
    dpi=300,
    bbox_inches="tight"
)
fig.savefig(
    FIGURES / "experiment_02_warning_episodes.pdf",
    bbox_inches="tight"
)
plt.close(fig)

report = f"""# Experiment 02 — Width Warning Episodes

## Objective

Evaluate whether episodes of unusually wide forecast
intervals precede new rolling-CRPS health-alarm onsets.

## Calibration

Calibration: {CAL_START.date()} to {CAL_END.date()}
Evaluation: {EVAL_START.date()} to {EVAL_END.date()}

Width threshold: {width_threshold:.4f} EUR/MWh
CRPS threshold: {crps_threshold:.4f} EUR/MWh

## Event definition

A width episode is a consecutive sequence of days
with daily width above the calibration threshold.

A CRPS onset is a transition from below to above
the rolling-CRPS calibration threshold.

For each width episode, warning timing is measured
from its first day.

## Results

Width-alert episodes: {len(episode_df)}
CRPS health onsets: {len(health_onsets)}

{summary.to_string(index=False)}

## Limitations

- Only one CRPS onset may be present.
- Warning horizons are exploratory.
- Width episodes occurring during an already active
  health alarm are not excluded from this table.
- This is not an estimate of a prospective false-alarm rate.
- CRPS uses overlapping rolling windows.
- Same-day realized CRPS is unavailable at forecast issue.
- Forecast issue timestamps require separate verification.
"""

(REPORTS / "experiment_02_report.md").write_text(report)

print("=== EXPERIMENT 02 RESULTS ===")
print(f"Width threshold: {width_threshold:.4f}")
print(f"CRPS threshold: {crps_threshold:.4f}")
print(f"Width episodes: {len(episode_df)}")
print(f"CRPS onsets: {len(health_onsets)}")

print("\n=== WARNING HORIZON SUMMARY ===")
print(summary.to_string(index=False))

print("\n=== WIDTH EPISODES ===")
if not episode_df.empty:
    print(episode_df.to_string(index=False))

print("\nSaved results to monitoring_research/")