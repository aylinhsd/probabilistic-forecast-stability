#!/usr/bin/env python3
"""Experiment 15: Multi-Dimensional Degradation Detection and Detector Comparison.

INPUT: One row per day. Required columns (names configurable with arguments):
    date, crps, pi80_coverage, pi80_width
Optional: benchmark_crps, observed_volatility

IMPORTANT SCIENTIFIC LIMITATIONS
- No historical alarm is labeled as a verified model failure.
- Controlled tests below perturb *daily monitoring signals*, not coherent predictive
  distributions. They test detector mechanics and do NOT establish real model-failure
  detection accuracy. Follow-up forecast-level perturbation is necessary.
- Realized volatility and realized CRPS are available only after prices arrive:
  this is ex-post health monitoring, not prospective early warning.
- Interval width increasing is a change, not necessarily deterioration.
- Calibration deviation is TWO-SIDED: high as well as low coverage is concerning.

Usage:
    python3 monitoring_research/scripts/experiment_15_multidimensional_detector_comparison.py --demo
    python3 monitoring_research/scripts/experiment_15_multidimensional_detector_comparison.py --input daily.csv --reference-end 2025-12-31

Dependencies: numpy, pandas, matplotlib. Optional river for true ADWIN:
    pip install numpy pandas matplotlib river
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from river.drift import ADWIN
    HAS_ADWIN = True
except ImportError:
    HAS_ADWIN = False

EPS = 1e-9


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument('--input', type=Path, help='Optional path to daily monitoring CSV (default: Experiment 14 table)')
    g.add_argument('--demo', action='store_true', help='Run synthetic DEMONSTRATION only')
    p.add_argument('--project-root', type=Path, default=None,
                   help='monitoring_research root; inferred from script location by default')
    p.add_argument('--output', type=Path, default=None,
                   help='Deprecated compatibility option; use --project-root instead')
    p.add_argument('--reference-end', default=None, help='Last calibration date, YYYY-MM-DD; never uses subsequent data')
    p.add_argument('--date-column', default='date')
    p.add_argument('--crps-column', default='crps')
    p.add_argument('--coverage-column', default='pi80_coverage')
    p.add_argument('--width-column', default='pi80_width')
    p.add_argument('--benchmark-column', default='benchmark_crps')
    p.add_argument('--volatility-column', default='observed_volatility')
    p.add_argument('--coverage-scale', choices=['auto','fraction','percent'], default='auto')
    p.add_argument('--reference-fraction', type=float, default=0.40,
                   help='Used only if --reference-end omitted; first fraction of chronological days')
    p.add_argument('--repetitions', type=int, default=20, help='Randomized trials PER scenario')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--min-days', type=int, default=80)
    return p.parse_args()


def demo_daily(seed=2026):
    rng = np.random.default_rng(seed)
    n = 420
    dates = pd.date_range('2025-07-01', periods=n, freq='D')
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.55*x[i-1] + rng.normal(0,1)
    vol = np.maximum(4, 18 + 4*x + rng.normal(0,3,n))
    crps = np.maximum(0.5, 5.5 + 0.15*vol + 0.55*x + rng.normal(0,1.0,n))
    benchmark = np.maximum(0.5, 7.5 + 0.18*vol + 0.4*x + rng.normal(0,1.3,n))
    width = np.maximum(2, 22 + 0.55*vol + 2*x + rng.normal(0,4,n))
    # 24 hourly observations per day -> discrete coverages; plausible demo only.
    prob = np.clip(0.8 + 0.014*x + rng.normal(0,0.025,n),.05,.95)
    cover = rng.binomial(24, prob)/24
    return pd.DataFrame({'date':dates, 'crps':crps, 'pi80_coverage':cover,
                         'pi80_width':width, 'benchmark_crps':benchmark,
                         'observed_volatility':vol})


def load_data(a):
    if a.demo:
        return demo_daily(a.seed), 'SYNTHETIC DEMONSTRATION'
    data_path = a.input if a.input is not None else (locate_project_root(a) / 'tables' / 'experiment_14_daily_monitoring.csv')
    if not data_path.exists():
        raise FileNotFoundError(f'{data_path} missing: run Experiment 14 first (or supply --input).')
    df = pd.read_csv(data_path)
    # Direct support for the actual Experiment 14/14b column names.
    known = {'CRPS_native':'crps', 'PI80_native_coverage':'pi80_coverage',
             'PI80_native_width':'pi80_width', 'PI80_width':'pi80_width',
             'PI80_native_interval_width':'pi80_width', 'volatility':'observed_volatility'}
    df = df.rename(columns={k:v for k,v in known.items() if k in df.columns and v not in df.columns})
    # Existing Experiment 14 has already computed matched relative and market-adjusted metrics.
    if 'CRPSS_native_vs_residual' in df.columns:
        df['relative_crps_skill_loss'] = -pd.to_numeric(df['CRPSS_native_vs_residual'], errors='coerce')
    if 'rMAE_native_vs_naive' in df.columns:
        df['relative_rmae'] = pd.to_numeric(df['rMAE_native_vs_naive'], errors='coerce')
    if 'market_expected_CRPS' in df.columns and 'crps' in df.columns:
        df['market_crps_residual'] = pd.to_numeric(df['crps'], errors='coerce') - pd.to_numeric(df['market_expected_CRPS'], errors='coerce')
    fields = {a.date_column:'date', a.crps_column:'crps',
              a.coverage_column:'pi80_coverage', a.width_column:'pi80_width',
              a.benchmark_column:'benchmark_crps', a.volatility_column:'observed_volatility'}
    required = [a.date_column] + ([] if 'crps' in df else [a.crps_column]) + ([] if 'pi80_coverage' in df else [a.coverage_column])
    missing = [f for f in required if f not in df]
    if missing: raise ValueError(f'Missing CSV columns: {missing}. See --help.')
    df = df.rename(columns={k:v for k,v in fields.items() if k in df})
    return df, f'EXPERIMENT 14/14B DAILY INPUT: {data_path.name} (not independently verified)'


def clean_data(df,a):
    df=df.copy()
    df['date']=pd.to_datetime(df['date'],errors='raise').dt.normalize()
    if df['date'].duplicated().any():
        raise ValueError('Duplicate dates: supply an aggregated ONE-ROW-PER-DAY dataset.')
    df=df.sort_values('date').reset_index(drop=True)
    cols=[c for c in ('crps','pi80_coverage','pi80_width','benchmark_crps','observed_volatility', 'relative_crps_skill_loss','relative_rmae','market_crps_residual') if c in df]
    for c in cols: df[c]=pd.to_numeric(df[c],errors='coerce')
    if df[['crps','pi80_coverage']].isna().any().any():
        raise ValueError('Core indicators contain missing or nonnumeric values. Repair upstream.')
    if not np.isfinite(df[cols].to_numpy(dtype=float)[~np.isnan(df[cols].to_numpy(dtype=float))]).all():
        raise ValueError('Nonfinite values in monitoring data')
    cov=df.pi80_coverage
    if a.coverage_scale=='percent' or (a.coverage_scale=='auto' and cov.max()>1.0 and cov.max()<=100):
        df['pi80_coverage']=cov/100.0
    elif a.coverage_scale=='fraction' and cov.max()>1:
        raise ValueError('Fraction-scale coverage values must be <= 1')
    if not df.pi80_coverage.between(0,1).all(): raise ValueError('Coverage outside [0,1]')
    if (df.crps<0).any(): raise ValueError('CRPS cannot be negative')
    if 'pi80_width' in df and (df.pi80_width.dropna()<0).any(): raise ValueError('PI80 width cannot be negative')
    if 'benchmark_crps' in df and (df.benchmark_crps.dropna()<0).any(): raise ValueError('Benchmark CRPS negative')
    if 'observed_volatility' in df and (df.observed_volatility.dropna()<0).any(): raise ValueError('Volatility negative')
    if len(df)<a.min_days: raise ValueError(f'Need at least {a.min_days} days; got {len(df)}')
    return df


def reference_mask(df,a):
    if a.reference_end:
        mask = (df.date<=pd.Timestamp(a.reference_end)).to_numpy()
        if mask.sum()<30 or (~mask).sum()<20:
            raise ValueError('Reference must have >=30 days and later evaluation >=20 days')
        return mask
    # Match previous experiment reference/evaluation split when possible.
    if df.date.min() < pd.Timestamp('2026-03-01') and df.date.max() >= pd.Timestamp('2026-03-21'):
        mask=(df.date < pd.Timestamp('2026-03-01')).to_numpy()
        if mask.sum() >= 30 and (~mask).sum() >= 20: return mask
    if not 0.15<=a.reference_fraction<=0.75: raise ValueError('Reference fraction must be 0.15..0.75')
    cutoff=int(len(df)*a.reference_fraction)
    mask=np.arange(len(df))<cutoff
    return mask


def make_signals(df,ref):
    """Pre-specified signals: no scoring leakage from test into normalization."""
    rolling_cov=df.pi80_coverage.rolling(window=14,min_periods=1)
    rolling_mean=rolling_cov.mean()
    rolling_std=rolling_cov.std(ddof=0)
    s={
        'absolute_crps':df.crps.to_numpy(float),
        'coverage_signed':(df.pi80_coverage-0.8).to_numpy(float),
        'coverage_undercoverage':(0.8-df.pi80_coverage).to_numpy(float),
        'coverage_overcoverage':(df.pi80_coverage-0.8).to_numpy(float),
        'coverage_absolute_deviation':np.abs(df.pi80_coverage.to_numpy(float)-0.8),
        'rolling_coverage_deviation_14d':np.abs(rolling_mean.to_numpy(float)-0.8),
        'rolling_coverage_instability_14d':rolling_std.to_numpy(float),
    }
    if 'pi80_width' in df and df.pi80_width.notna().all():
        s['interval_width_change']=df.pi80_width.to_numpy(float)
    for field in ('relative_crps_skill_loss','relative_rmae','market_crps_residual'):
        if field in df.columns and df[field].notna().all():
            s[field]=df[field].to_numpy(float)
    if 'benchmark_crps' in df and df.benchmark_crps.notna().all():
        s['relative_crps_difference']=(df.crps-df.benchmark_crps).to_numpy(float)
    if 'observed_volatility' in df and df.observed_volatility.notna().all():
        # Reference-only linear difficulty adjustment. A simple *explanatory* baseline,
        # not causal adjustment and not available before future prices arrive.
        v=df.observed_volatility.to_numpy(float)
        for source,name in [('crps','volatility_adjusted_crps')] + ([('pi80_width','volatility_adjusted_width')] if 'pi80_width' in df and df.pi80_width.notna().all() else []):
            y=df[source].to_numpy(float)
            vref=v[ref]
            x=np.stack([np.ones(len(vref)),vref],axis=1)
            beta=np.linalg.lstsq(x,y[ref],rcond=None)[0]
            s[name]=y-(beta[0]+beta[1]*v)
    return s


def robust_scale(x):
    med=float(np.median(x))
    mad=float(np.median(np.abs(x-med)))
    val=1.4826*mad
    if val<EPS: val=float(np.std(x,ddof=1))
    return max(val,EPS)


def standardized(x,reference):
    mu=float(np.median(x[reference]))
    scale=robust_scale(x[reference])
    return (x-mu)/scale


def cusum(x,ref, k=0.5,h=5):
    """One-sided upward CUSUM with reset after alarm; no horizon peeking."""
    z=standardized(x,ref)
    acc=0.0; out=np.zeros(len(z),dtype=bool)
    for i,val in enumerate(z):
        acc=max(0,acc+val-k)
        if acc>h:
            out[i]=True;acc=0
    return out


def page_hinkley(x,ref, delta=0.15,threshold=8):
    """Upward PH on fixed reference standardized scale, reset after alarm."""
    z=standardized(x,ref)
    cumulative=0.; minimum=0.; out=np.zeros(len(z),dtype=bool)
    for i,val in enumerate(z):
        cumulative+=val-delta
        minimum=min(minimum,cumulative)
        if cumulative-minimum>threshold:
            out[i]=True;cumulative=0.;minimum=0.
    return out


def ewma(x,ref,alpha=0.2,L=3):
    """One-sided EWMA; fixed reference scale, resets after alarm."""
    z=standardized(x,ref)
    state=0.;out=np.zeros(len(z),dtype=bool)
    threshold=L*np.sqrt(alpha/(2-alpha))
    for i,val in enumerate(z):
        state=alpha*val+(1-alpha)*state
        if state>threshold:
            out[i]=True;state=0.
    return out


def fixed_threshold(x,ref,q=.95):
    # P95 from reference only, independent days after calibration.
    cutoff=float(np.quantile(x[ref],q))
    return x>cutoff


def adwin_detector(x,ref,delta=.002):
    if not HAS_ADWIN: return None
    # ADWIN detects changes in mean; standardized input is scale comparable.
    # River ADWIN can detect upward and downward shifts; here both are reported.
    # Unlike one-sided CUSUM/PH/EWMA, ADWIN alerts may include downward shifts.
    z=standardized(x,ref)
    model=ADWIN(delta=delta)
    out=np.zeros(len(z),dtype=bool)
    for i,val in enumerate(z):
        model.update(float(val))
        out[i]=bool(model.drift_detected)
    return out


def detectors(x,ref):
    return {'Fixed P95':fixed_threshold(x,ref),
            'CUSUM':cusum(x,ref),
            'Page-Hinkley':page_hinkley(x,ref),
            'EWMA':ewma(x,ref),
            'ADWIN':adwin_detector(x,ref)}


def reference_calibration(x,ref,method,target=0.05):
    """Tune one hyperparameter (threshold) on reference-only signals by grid search.

    This is a *heuristic* matched raw alarm-day-rate calibration, not a guarantee
    of equal event-level false alarms or nominal significance under dependence.
    """
    if method=='Fixed P95': return {'q':.95}
    if method=='CUSUM':
        grid=[{'k':k,'h':h} for k in (.25,.5,1.) for h in (2,3,4,5,7,10,14,20,30)]
        runner=lambda a:cusum(x,ref,**a)
    elif method=='Page-Hinkley':
        grid=[{'delta':d,'threshold':h} for d in (.05,.15,.3) for h in (2,4,6,8,12,16,24,35)]
        runner=lambda a:page_hinkley(x,ref,**a)
    elif method=='EWMA':
        grid=[{'alpha':alpha,'L':L} for alpha in (.1,.2,.3) for L in (1.5,2,2.5,3,3.5,4,5,6)]
        runner=lambda a:ewma(x,ref,**a)
    elif method=='ADWIN':
        if not HAS_ADWIN: return None
        grid=[{'delta':d} for d in (.1,.05,.02,.01,.005,.002,.001,.0001)]
        runner=lambda a:adwin_detector(x,ref,**a)
    else: raise ValueError(method)
    # Deterministic tie-breaking; prefer fewer reference false alarms on equal errors.
    scores=[]
    for idx,settings in enumerate(grid):
        alarms=runner(settings)
        n=int(alarms[ref].sum()); r=float(alarms[ref].mean())
        scores.append((abs(r-target), max(0.,r-target), idx,settings))
    return min(scores,key=lambda q:q[:3])[3]


def run_method(method,x,ref,params):
    if method=='Fixed P95': return fixed_threshold(x,ref,**params)
    if method=='CUSUM': return cusum(x,ref,**params)
    if method=='Page-Hinkley': return page_hinkley(x,ref,**params)
    if method=='EWMA': return ewma(x,ref,**params)
    if method=='ADWIN': return adwin_detector(x,ref,**params)
    raise ValueError(method)


def label_scenarios(signal_name):
    if signal_name in ('coverage_signed',):
        return ['shift_up','shift_down','temporary_up','none']
    if signal_name in ('coverage_absolute_deviation',):
        return ['shift_up','gradual_up','temporary_up','none']
    return ['shift_up','gradual_up','temporary_up','none']


def simulate(base,reference,scenario,rng):
    """Signal-level stress test of detector mechanics.

    Events are planted in a moving block bootstrap continuation of reference.
    The bootstrap preserves some serial dependence, but not full calendar,
    market-regime or cross-indicator consistency.
    """
    ref_vals=np.asarray(base[reference],float)
    n=200
    blocks=[]
    while len(blocks)<n:
        start=int(rng.integers(0,max(1,len(ref_vals)-7)))
        blocks.extend(ref_vals[start:start+7].tolist())
    y=np.asarray(blocks[:n],float)
    event_onset=65
    scale=robust_scale(ref_vals)
    if scenario=='shift_up': y[event_onset:]+=1.8*scale
    elif scenario=='shift_down': y[event_onset:]-=1.8*scale
    elif scenario=='gradual_up': y[event_onset:]+=1.8*scale*np.minimum(1,np.arange(n-event_onset)/35)
    elif scenario=='temporary_up': y[event_onset:event_onset+3]+=3.0*scale
    elif scenario!='none': raise ValueError(scenario)
    # Provide clean historical reference, then evaluation starting after reference.
    combined=np.r_[ref_vals,y]
    mask=np.arange(len(combined))<len(ref_vals)
    return combined,mask,len(ref_vals)+event_onset,len(ref_vals)+n


def evaluate_event(alarms, ref_length, onset, stop,scenario):
    a=np.flatnonzero(alarms[ref_length:stop])+ref_length
    # Pre-event warnings count as false alarms for controlled *detection* tests,
    # because no elevated-risk labels exist before known onset.
    false_count=int(np.sum(a<onset))
    pre_days=onset-ref_length
    if scenario=='none':
        return {'detected':np.nan,'delay':np.nan,'pre_false_alarms':int(len(a)),
                'pre_days':stop-ref_length}
    # For transient spike, success window is only 3 days; for persistent 30 days.
    horizon=3 if scenario=='temporary_up' else 30
    hits=a[(a>=onset)&(a<min(onset+horizon,stop))]
    detected=int(len(hits)>0)
    return {'detected':detected,'delay':int(hits[0]-onset) if detected else np.nan,
            'pre_false_alarms':false_count,'pre_days':pre_days}


def plot_signals(df,signals,ref,figure_out,tag):
    keys=[k for k in ('absolute_crps','coverage_signed','coverage_undercoverage','coverage_overcoverage','coverage_absolute_deviation','rolling_coverage_deviation_14d',
                      'rolling_coverage_instability_14d','interval_width_change','relative_crps_difference','relative_crps_skill_loss','relative_rmae','market_crps_residual',
                      'volatility_adjusted_crps') if k in signals]
    fig,axes=plt.subplots(len(keys),1,figsize=(12,2.6*len(keys)),sharex=True)
    if len(keys)==1: axes=[axes]
    for ax,k in zip(axes,keys):
        ax.plot(df.date,signals[k],lw=.85)
        ax.axvline(df.date.iloc[np.where(~ref)[0][0]],color='gray',linestyle='--',alpha=.7)
        if k=='coverage_signed': ax.axhline(0,color='black',lw=.8)
        ax.set_ylabel(k.replace('_','\n'),fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle(f'Experiment 15 daily signals — {tag}')
    fig.tight_layout();fig.savefig(figure_out/'experiment_15_monitoring_signals.png',dpi=155);plt.close(fig)


def plot_summary(summary,figure_out,tag):
    s=summary[summary['scenario'].eq('shift_up') & summary['signal'].eq('absolute_crps')].dropna(subset=['detection_rate'])
    if s.empty: return
    fig,ax=plt.subplots(figsize=(9,4.8))
    ax.bar(s.detector,s.detection_rate)
    ax.set_ylim(0,1);ax.set_ylabel('Detection rate (within 30 days)')
    ax.set_title('Illustrative synthetic CRPS shift detection — '+tag)
    ax.tick_params(axis='x',rotation=25)
    fig.tight_layout();fig.savefig(figure_out/'experiment_15_synthetic_detection_rates.png',dpi=160);plt.close(fig)


def locate_project_root(a):
    if a.project_root is not None:
        return a.project_root.expanduser().resolve()
    if a.output is not None:
        warnings.warn('--output is deprecated; outputs now go into figures/reports/tables under that root')
        return a.output.expanduser().resolve()
    here = Path(__file__).resolve()
    if here.parent.name == 'scripts' and here.parent.parent.name == 'monitoring_research':
        return here.parent.parent
    if (Path.cwd() / 'monitoring_research').is_dir():
        return (Path.cwd() / 'monitoring_research').resolve()
    raise ValueError('Place script in monitoring_research/scripts/ or pass --project-root PATH')


def write_report(path, *, tag, df, ref, signals, methods, settings_df, alarms_df, summary, repetitions):
    from datetime import datetime
    lines = [
        '# Experiment 15 — Multi-Dimensional Degradation Detection and Detector Comparison',
        '',
        f'**Data:** {tag}  ',
        f'**Period:** {df.date.min().date()} to {df.date.max().date()}  ',
        f'**Reference:** {df.date[ref].min().date()} to {df.date[ref].max().date()} ({int(ref.sum())} days)  ',
        f'**Evaluation:** {df.date[~ref].min().date()} to {df.date[~ref].max().date()} ({int((~ref).sum())} days)',
        '',
        '## Objective',
        'Compare fixed P95, CUSUM, Page-Hinkley, EWMA and (if installed) ADWIN across complementary probabilistic monitoring signals. Assess both below- and above-target PI80 coverage, its temporal variation, CRPS, interval width, benchmark-relative performance and realized-volatility-adjusted performance.',
        '',
        '## Methods',
        f'- Signals evaluated: {", ".join(signals)}.',
        f'- Detectors evaluated: {", ".join(methods)}.',
        '- Thresholds are chosen from the reference period only using a heuristic raw alarm-day rate target, not a validated event-level false-alarm constraint.',
        f'- Synthetic signal-level stress tests: {repetitions} trials per scenario, method and signal.',
        '- Coverage deviation is two-sided: sustained undercoverage **and** sustained overcoverage may be problematic.',
        '- The input file might omit PI80 width; this metric is then **skipped** rather than fabricated.',
        '',
        '## Observed historical alerts (unvalidated)',
        'Counts below are *not* false-positive rates: there are no independently verified real-world deterioration labels.',
        '',
        '| Detector | Evaluation alarm rows (across all signals) |',
        '|---|---:|',
    ]
    for method in methods:
        lines.append(f'| {method} | {int((alarms_df.detector == method).sum())} |')
    lines += ['', 'Counts sum across separate signal alarms and must not be interpreted as distinct events or as a ranking.', '',
              '## Synthetic controlled-change comparison (illustrative)',
              'These trials perturb daily monitoring signals, **not the underlying 101 predictive quantiles**. Results only characterize detector mechanics.', '',
              '| Signal | Detector | Scenario | Detection rate | Median detection delay (days) | Pre-event false alarms / 100 days |',
              '|---|---|---|---:|---:|---:|']
    candidate=summary[summary.scenario.eq('shift_up')]
    for row in candidate.itertuples(index=False):
        rate='n/a' if pd.isna(row.detection_rate) else f'{row.detection_rate:.2f}'
        delay='n/a' if pd.isna(row.median_detection_delay) else f'{row.median_detection_delay:.1f}'
        fa=f'{row.pre_event_false_alarms_per_100_days:.1f}'
        lines.append(f'| {row.signal} | {row.detector} | shift_up | {rate} | {delay} | {fa} |')
    lines += ['', '## Interpretation and limitations',
              '- A high detection rate with many pre-event false alarms is not necessarily preferable to a slower detector.',
              '- Upward and downward deviations from 80% PI80 coverage both matter; daily coverage from 24 hours is noisy, so use rolling windows and dependence-aware assessment.',
              '- Increasing interval width can reflect greater anticipated market uncertainty; interpret width and coverage jointly.',
              '- PI80 interval width is omitted if not present in the Experiment 14 daily CSV. Add a verified matched series to enable it.',
              '- ADWIN is bidirectional; CUSUM, Page-Hinkley and EWMA are upward-only. Dedicated undercoverage and overcoverage signals allow explicit directional testing, but raw rankings are not fair without matched detection targets.',
              '- A natural historical alarm is not proof of model failure. A single retrospective March 2026 screening episode is insufficient for reliable method ranking.',
              '- Realized volatility, CRPS and coverage require observed prices and are not valid as earlier forecast-time warning inputs.',
              '- Future research should introduce coherent perturbations to hourly forecast quantiles and recompute all metrics jointly, and compare event-level false alarms under matched budgets.',
              '', '## Output files',
              '- `figures/experiment_experiment_15_monitoring_signals.png`',
              '- `figures/experiment_experiment_15_synthetic_detection_rates.png`',
              '- `tables/experiment_15_input_daily_checked.csv`',
              '- `tables/experiment_15_reference_settings.csv`',
              '- `tables/experiment_15_historical_alerts_unvalidated.csv`',
              '- `tables/experiment_15_synthetic_trials.csv`',
              '- `tables/experiment_15_synthetic_comparison.csv`',
              '- `reports/experiment_15_run_metadata.json`',
              '- `reports/experiment_15_report.md`', '',
              '**Status:** Research prototype, not independently validated degradation-detection performance.']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    a=arguments()
    if a.repetitions<1: raise ValueError('Repetitions must be >=1')
    root=locate_project_root(a)
    folders={name:root/name for name in ('figures','reports','scripts','tables')}
    for path in folders.values(): path.mkdir(parents=True,exist_ok=True)
    table_out=folders['tables']; figure_out=folders['figures']; report_out=folders['reports']
    df,tag=load_data(a)
    df=clean_data(df,a)
    ref=reference_mask(df,a)
    signals=make_signals(df,ref)
    if 'pi80_width' not in df:
        warnings.warn('No verified PI80 interval-width column in Experiment 14 table: width diagnostics omitted; all other available signals will run.')
    if not HAS_ADWIN:
        warnings.warn('Optional river package not installed: ADWIN will be SKIPPED. Run pip install river.')
    df.to_csv(table_out/'experiment_15_input_daily_checked.csv',index=False)
    plot_signals(df,signals,ref,figure_out,tag)
    methods=['Fixed P95','CUSUM','Page-Hinkley','EWMA']+(['ADWIN'] if HAS_ADWIN else [])
    detection_rows=[]; settings_rows=[]; simulation_rows=[]
    rng=np.random.default_rng(a.seed)
    for signal_name,x in signals.items():
        for method in methods:
            settings=reference_calibration(x,ref,method)
            alarms=run_method(method,x,ref,settings)
            settings_rows.append({'signal':signal_name,'detector':method,
                                  'settings':json.dumps(settings,sort_keys=True),
                                  'reference_alarm_days':int(alarms[ref].sum()),
                                  'reference_days':int(ref.sum()),
                                  'reference_alarm_day_rate':float(alarms[ref].mean())})
            for ix in np.flatnonzero(alarms & ~ref):
                detection_rows.append({'date':df.date.iloc[ix].date().isoformat(),
                                       'signal':signal_name,'detector':method,
                                       'value':float(x[ix])})
            for scenario in label_scenarios(signal_name):
                for trial in range(a.repetitions):
                    simulated,sim_ref,onset,stop=simulate(x,ref,scenario,rng)
                    # IMPORTANT: hyperparameters selected ONLY on original reference.
                    hit=run_method(method,simulated,sim_ref,settings)
                    result=evaluate_event(hit,int(sim_ref.sum()),onset,stop,scenario)
                    simulation_rows.append({'signal':signal_name,'detector':method,
                                            'scenario':scenario,'trial':trial,**result})
    settings_df=pd.DataFrame(settings_rows)
    settings_df.to_csv(table_out/'experiment_15_reference_settings.csv',index=False)
    alarms_df=pd.DataFrame(detection_rows,columns=['date','signal','detector','value'])
    alarms_df.to_csv(table_out/'experiment_15_historical_alerts_unvalidated.csv',index=False)
    sim_df=pd.DataFrame(simulation_rows)
    sim_df.to_csv(table_out/'experiment_15_synthetic_trials.csv',index=False)
    summary=(sim_df.groupby(['signal','detector','scenario'],as_index=False)
        .agg(detection_rate=('detected','mean'),median_detection_delay=('delay','median'),
             mean_pre_event_false_alarms=('pre_false_alarms','mean'),
             mean_pre_event_days=('pre_days','mean'),trials=('trial','count')))
    summary['pre_event_false_alarms_per_100_days']=100*summary.mean_pre_event_false_alarms/summary.mean_pre_event_days
    summary.to_csv(table_out/'experiment_15_synthetic_comparison.csv',index=False)
    plot_summary(summary,figure_out,tag)
    report={
      'data_origin':tag, 'input_rows':len(df), 'reference_days':int(ref.sum()),
      'evaluation_days':int((~ref).sum()),'reference_end':df.date[ref].max().date().isoformat(),
      'evaluation_start':df.date[~ref].min().date().isoformat(),
      'signals_tested':list(signals),'detectors_tested':methods,
      'adwin_available':HAS_ADWIN,'simulation_repetitions_per_scenario':a.repetitions,
      'important_limits':[
        'Historical alerts have NO independent ground-truth failure labels.',
        'Controlled perturbations were made to monitoring SIGNALS, not coherent forecast distributions.',
        'Synthetic detection rates are NOT estimates of real operational failure-detection performance.',
        'Observed volatility and realized CRPS are ex-post, NOT forecast-time information.',
        'Calibration deviation is bidirectional (undercoverage AND overcoverage), with dedicated directional signals.',
        'Daily coverage has finite 24-hour sample resolution and serial dependence.',
        'Detector thresholds were approximately matched on REFERENCE raw alarm-day rates, not independent event-level significance.',
        'ADWIN is a two-sided mean-change detector while other sequential detectors here are one-sided upward; do not rank without direction-matched tests.',
        'Coverage-width changes require joint interpretation; high width or high coverage alone does not prove model deterioration.',
        'Default reference split may span less than one annual cycle; prefer explicit --reference-end and check seasonal sensitivity.'
      ]}
    (report_out/'experiment_15_run_metadata.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    write_report(report_out/'experiment_15_report.md', tag=tag, df=df, ref=ref, signals=signals, methods=methods, settings_df=settings_df, alarms_df=alarms_df, summary=summary, repetitions=a.repetitions)
    print('\nExperiment 15 completed:',tag)
    print('Days:',len(df),'reference:',ref.sum(),'evaluation:',(~ref).sum())
    print('Signals:',', '.join(signals))
    print('Detectors:',', '.join(methods))
    print('Results root:',root)
    print('Tables:',table_out)
    print('Figures:',figure_out)
    print('Report:',report_out/'experiment_15_report.md')
    print('IMPORTANT: Synthetic stress tests are NOT independent model-failure validation.')



# -----------------------------------------------------------------------------
# Expanded visual analytics. All figures use the actual CSV results produced
# above; synthetic validation is explicitly marked as synthetic everywhere.
# -----------------------------------------------------------------------------

def _bar_numbers(ax, bars, fmt='{:.0f}', fontsize=8):
    for bar in bars:
        v = bar.get_height()
        if np.isfinite(v):
            ax.annotate(fmt.format(v), (bar.get_x()+bar.get_width()/2, v),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=fontsize)


def _finish(fig, folder, filename):
    fig.tight_layout()
    fig.savefig(folder / filename, dpi=210, bbox_inches='tight')
    plt.close(fig)
    return filename


def save_coverage_30day_deviation(raw, evaluation_start, folder):
    """Export the lower two-sided coverage panel as a standalone PNG and PDF."""
    # Calculate the rolling mean before selecting evaluation dates so the first
    # evaluation points retain the same reference history as the original panel.
    coverage = raw['pi80_coverage'].to_numpy(float)
    deviation = pd.Series(coverage).rolling(30, min_periods=30).mean()*100 - 80
    evaluation = raw['date'] >= pd.Timestamp(evaluation_start)
    dates, values = raw.loc[evaluation, 'date'], deviation[evaluation]
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.plot(dates, values, color='#58529e', lw=1.8)
    ax.axhline(0, color='black', ls='--')
    ax.fill_between(dates, values, 0, where=values < 0,
                    alpha=.2, color='red', label='Undercoverage')
    ax.fill_between(dates, values, 0, where=values > 0,
                    alpha=.2, color='blue', label='Overcoverage')
    ax.set(title='30-day coverage deviation from the 80% target',
           ylabel='30-day deviation (percentage points)', xlabel='Date')
    ax.legend(loc='best')
    ax.grid(alpha=.2)
    fig.tight_layout()
    name = 'experiment_15_coverage_30day_deviation'
    for extension in ('png', 'pdf'):
        fig.savefig(folder / f'{name}.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return f'{name}.png'


def expanded_figures(root):
    t, f = root/'tables', root/'figures'
    raw = pd.read_csv(t/'experiment_15_input_daily_checked.csv', parse_dates=['date'])
    alerts = pd.read_csv(t/'experiment_15_historical_alerts_unvalidated.csv')
    settings = pd.read_csv(t/'experiment_15_reference_settings.csv')
    summ = pd.read_csv(t/'experiment_15_synthetic_comparison.csv')
    trials = pd.read_csv(t/'experiment_15_synthetic_trials.csv')
    evaluation_start = pd.Timestamp(raw['date'].min())
    meta = json.loads((root/'reports'/'experiment_15_run_metadata.json').read_text())
    evaluation_start = pd.Timestamp(meta['evaluation_start'])
    evaluation_days = int(meta['evaluation_days'])
    figures = []

    # 1. A genuinely useful two-sided calibration figure, preserving direction.
    cov = raw['pi80_coverage'].to_numpy(float)
    r7 = pd.Series(cov).rolling(7, min_periods=7).mean()*100
    r30 = pd.Series(cov).rolling(30, min_periods=30).mean()*100
    ev = raw['date'] >= evaluation_start
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].scatter(raw.date[ev], 100*cov[ev], s=9, alpha=.25, label='Daily observed coverage')
    axes[0].plot(raw.date[ev], r7[ev], lw=1.25, label='Rolling 7 days')
    axes[0].plot(raw.date[ev], r30[ev], lw=2, label='Rolling 30 days')
    axes[0].axhline(80, color='black', ls='--', label='Nominal 80%')
    axes[0].set(ylabel='Coverage (%)', title='PI80 coverage — both under- and overcoverage matter', ylim=(0, 102))
    axes[0].legend(ncol=4, fontsize=9)
    dev = r30 - 80
    axes[1].plot(raw.date[ev], dev[ev], color='#58529e', lw=1.8)
    axes[1].axhline(0, color='black', ls='--')
    axes[1].fill_between(raw.date[ev], dev[ev], 0, where=dev[ev]<0, alpha=.2, color='red', label='Undercoverage')
    axes[1].fill_between(raw.date[ev], dev[ev], 0, where=dev[ev]>0, alpha=.2, color='blue', label='Overcoverage')
    axes[1].set(ylabel='30-day deviation (percentage points)', xlabel='Date')
    axes[1].legend(loc='best')
    for ax in axes: ax.grid(alpha=.2)
    figures.append(_finish(fig, f, 'experiment_15_coverage_two_sided.png'))
    figures.append(save_coverage_30day_deviation(raw, evaluation_start, f))

    # 2. Normalization to a 0..1 visual not appropriate for raw signals; use 2-panel
    #    calibration deviation and daily coverage standard deviation instead.
    sd30 = pd.Series(100*cov).rolling(30,min_periods=30).std()
    reference = raw.date < evaluation_start
    ref_abs_dev = dev[reference].dropna().abs()
    ref_sd = sd30[reference].dropna()
    fig, ax = plt.subplots(2,1,figsize=(13,7),sharex=True)
    ax[0].plot(raw.date[ev],abs(dev[ev]),lw=1.8,label='Absolute distance from 80%')
    if len(ref_abs_dev):
        lim=float(ref_abs_dev.quantile(.95))
        ax[0].axhline(lim,color='red',ls=':',label=f'Reference P95 = {lim:.2f} pp')
    ax[0].set_ylabel('Absolute calibration deviation (pp)'); ax[0].legend(fontsize=9)
    ax[1].plot(raw.date[ev],sd30[ev],lw=1.8,label='30-day SD of daily coverage')
    if len(ref_sd):
        lim=float(ref_sd.quantile(.95))
        ax[1].axhline(lim,color='red',ls=':',label=f'Reference P95 = {lim:.2f} pp')
    ax[1].set(xlabel='Date',ylabel='SD (percentage points)'); ax[1].legend(fontsize=9)
    for aa in ax: aa.grid(alpha=.2)
    fig.suptitle('Calibration instability versus distance from nominal coverage')
    figures.append(_finish(fig,f,'experiment_15_coverage_stability_comparison.png'))

    # 3. All historical detector / signal alarm counts (unvalidated, descriptive).
    pivot = alerts.groupby(['signal','detector']).size().unstack(fill_value=0)
    pivot = pivot.reindex(index=settings.signal.drop_duplicates(),columns=settings.detector.drop_duplicates(),fill_value=0)
    fig, ax=plt.subplots(figsize=(max(10, len(pivot.columns)*2.2),max(5,len(pivot)*.62+2)))
    image=ax.imshow(pivot.to_numpy(),aspect='auto',cmap='Blues')
    ax.set_xticks(np.arange(len(pivot.columns)),pivot.columns,rotation=25,ha='right')
    ax.set_yticks(np.arange(len(pivot.index)),pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value=int(pivot.iloc[i,j]); ax.text(j,i,str(value),ha='center',va='center',
                color='white' if value>max(1,pivot.to_numpy().max())*.58 else 'black',fontsize=9)
    ax.set_title(f'Observed evaluation alerts by signal and detector (N={evaluation_days} days)\nCounts are not verified true/false positives')
    fig.colorbar(image,ax=ax,shrink=.8,label='Alarm days')
    figures.append(_finish(fig,f,'experiment_15_historical_alert_count_heatmap.png'))

    # 4. The actual burden of notifications: distinct alarm days per detector
    #    across all metrics; a day with 5 signal alerts counts only once.
    count_unique=alerts.groupby('detector')['date'].nunique().reindex(settings.detector.drop_duplicates(),fill_value=0)
    fig,ax=plt.subplots(figsize=(10,5))
    bars=ax.bar(np.arange(len(count_unique)), count_unique.values, color='#457b9d')
    _bar_numbers(ax,bars);ax.set_xticks(np.arange(len(count_unique)),count_unique.index,rotation=15)
    ax.set_ylim(0,max(1,count_unique.max())*1.18)
    ax.set(ylabel='Distinct evaluation days with ≥1 alert',title='Operational alert burden (combined across monitored signals)')
    ax.grid(axis='y',alpha=.2)
    figures.append(_finish(fig,f,'experiment_15_distinct_alert_days.png'))

    # 5. Pairwise Jaccard index for alerts from different algorithms, using
    #    the same core absolute CRPS signal. (0=no overlap,1=identical days.)
    focus='absolute_crps'
    sub=alerts.loc[alerts.signal.eq(focus),['date','detector']]
    names=list(settings.detector.drop_duplicates())
    sets={d:set(sub.loc[sub.detector.eq(d),'date']) for d in names}
    jac=np.full((len(names),len(names)),np.nan)
    for i,a in enumerate(names):
        for j,b in enumerate(names):
            uni=sets[a]|sets[b]
            jac[i,j]=len(sets[a]&sets[b])/len(uni) if uni else (1 if i==j else np.nan)
    fig,ax=plt.subplots(figsize=(8,6))
    im=ax.imshow(jac,vmin=0,vmax=1,cmap='YlGnBu')
    ax.set_xticks(range(len(names)),names,rotation=25,ha='right');ax.set_yticks(range(len(names)),names)
    for i in range(len(names)):
        for j in range(len(names)):
            value=jac[i,j]
            ax.text(j,i,'—' if not np.isfinite(value) else f'{value:.2f}',ha='center',va='center',
                    color='white' if value>.66 else 'black')
    fig.colorbar(im,ax=ax,label='Jaccard overlap')
    ax.set_title('Historical CRPS alert agreement — Jaccard index')
    figures.append(_finish(fig,f,'experiment_15_crps_detector_agreement.png'))

    # 6. Synthetic detection rate for selected complementary indicator groups,
    #    with numeric values next to cells for direct comparability.
    priority=['absolute_crps','coverage_undercoverage','coverage_overcoverage',
              'rolling_coverage_deviation_14d','rolling_coverage_instability_14d',
              'interval_width_change','relative_crps_skill_loss','relative_rmae',
              'market_crps_residual','volatility_adjusted_crps']
    focus_summ=summ.loc[summ.scenario.eq('shift_up')].copy()
    keep=[x for x in priority if x in set(focus_summ.signal)]
    view=focus_summ.pivot(index='signal',columns='detector',values='detection_rate').reindex(index=keep,columns=names)
    if len(view):
        fig,ax=plt.subplots(figsize=(max(10,len(names)*1.7),max(5,len(view)*.57+2)))
        im=ax.imshow(view.to_numpy(dtype=float),vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        ax.set_xticks(range(len(names)),names,rotation=20,ha='right');ax.set_yticks(range(len(view)),view.index)
        for i in range(len(view)):
            for j in range(len(names)):
                x=view.iloc[i,j]
                ax.text(j,i,'—' if pd.isna(x) else f'{100*x:.0f}%',ha='center',va='center',fontsize=9,
                        color='white' if pd.notna(x) and x>.65 else 'black')
        fig.colorbar(im,ax=ax,label='Detected within 30 days (fraction)')
        ax.set_title('SYNTHETIC sustained upward shifts — detection rate by signal and detector')
        figures.append(_finish(fig,f,'experiment_15_simulated_detection_heatmap.png'))

    # 7. Three-panel actual tradeoff: detection, delay, and no-event alarms
    #    on the main CRPS signal; numerical labels on every bar.
    core=summ.loc[summ.signal.eq('absolute_crps')]
    ups=core.loc[core.scenario.eq('shift_up')].set_index('detector').reindex(names)
    no=core.loc[core.scenario.eq('none')].set_index('detector').reindex(names)
    fig,axes=plt.subplots(1,3,figsize=(16,4.9))
    specs=[(100*ups.detection_rate,'Detection rate (%)','Sustained shift detection','{:.0f}%'),
           (ups.median_detection_delay,'Delay (days)','Median delay when detected','{:.1f}'),
           (no.pre_event_false_alarms_per_100_days,'Alerts / 100 days','No-change false alarm burden','{:.1f}')]
    for ax,(vals,ylabel,title,fmt) in zip(axes,specs):
        vals=vals.to_numpy(float)
        bars=ax.bar(np.arange(len(names)),np.nan_to_num(vals,nan=0),color='#497c9b')
        _bar_numbers(ax,bars,fmt); ax.set_xticks(np.arange(len(names)),names,rotation=30,ha='right',fontsize=8)
        ax.set_ylim(0,max(1,np.nanmax(vals))*1.3 if np.isfinite(vals).any() else 1)
        ax.set(ylabel=ylabel,title=title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('SYNTHETIC CRPS stress test — detection/latency/false-alarm tradeoff')
    figures.append(_finish(fig,f,'experiment_15_crps_detection_delay_falsealarms.png'))

    # 8. Bidirectional coverage detection, comparing under/over as separate
    #    positive-shift monitoring signals.
    compare=summ.loc[summ.signal.isin(['coverage_undercoverage','coverage_overcoverage']) & summ.scenario.eq('shift_up')]
    if len(compare):
        fig, ax=plt.subplots(figsize=(10,5))
        positions=np.arange(len(names)); width=.36
        for i,(signal,title) in enumerate([('coverage_undercoverage','Undercoverage'),('coverage_overcoverage','Overcoverage')]):
            vals=compare[compare.signal.eq(signal)].set_index('detector').reindex(names).detection_rate.to_numpy()*100
            bars=ax.bar(positions+(i-.5)*width,np.nan_to_num(vals),width,label=title)
            _bar_numbers(ax,bars,'{:.0f}%',7)
        ax.set_xticks(positions,names,rotation=20,ha='right');ax.set_ylim(0,115)
        ax.set(ylabel='Detection rate (%)',title='SYNTHETIC calibration deterioration — both directions tested')
        ax.legend();ax.grid(axis='y',alpha=.2)
        figures.append(_finish(fig,f,'experiment_15_under_vs_overcoverage_detection.png'))

    # 9. False alarm rate versus detection delay scatter plot to present the
    #    central detector tradeoff, no invented performance ranking.
    sub=summ.loc[summ.scenario.eq('shift_up') & summ.signal.isin(keep)]
    fig,ax=plt.subplots(figsize=(10,6))
    for detector,grp in sub.groupby('detector'):
        good=grp.dropna(subset=['median_detection_delay','pre_event_false_alarms_per_100_days'])
        if not good.empty:
            ax.scatter(good.pre_event_false_alarms_per_100_days,good.median_detection_delay,
                       s=55,alpha=.7,label=detector)
    ax.set(xlabel='Pre-event alarm days / 100 days (synthetic)',ylabel='Detection delay (days; detected trials)',
           title='SYNTHETIC tradeoff across monitored signals')
    ax.legend(fontsize=9);ax.grid(alpha=.2)
    figures.append(_finish(fig,f,'experiment_15_detection_tradeoff_scatter.png'))

    # 10. Numeric table from synthetic shifts, sorted by SIGNAL for examination
    #     rather than an unqualified global ranking.
    summary=focus_summ[['signal','detector','detection_rate','median_detection_delay','pre_event_false_alarms_per_100_days','trials']].copy()
    summary.to_csv(t/'experiment_15_numeric_detector_comparison.csv',index=False)
    # 11. Historical descriptive, coverage signedness by observed period.
    def _summary_cover(x):
        x=np.asarray(x,float)
        return {'days':len(x),'mean_coverage_pct':100*np.mean(x),
                'mean_absolute_deviation_pp':100*np.mean(np.abs(x-.8)),
                'under_80_days':int((x<.8).sum()),'over_80_days':int((x>.8).sum()),
                'exactly_80_days':int((x==.8).sum())}
    covrows=[]
    for name,part in [('reference',cov[~ev]),('evaluation',cov[ev])]:
        covrows.append({'period':name,**_summary_cover(part)})
    pd.DataFrame(covrows).to_csv(t/'experiment_15_coverage_numeric_summary.csv',index=False)

    # Append to automatically generated report, without confusing simulation
    # outcomes with actual reliability or model-specific failure ground truth.
    report=root/'reports'/'experiment_15_report.md'
    with report.open('a',encoding='utf-8') as out:
        out.write('\n## Expanded quantitative visual comparisons\n\n')
        out.write('The following plots were generated automatically from experiment outputs. '\
                  'Historical plots use observed monitoring data but lack independent failure labels. '\
                  'Simulation plots use controlled **signal-level**, not forecast-level, perturbations.\n\n')
        out.write('| Figure | Meaning |\n|---|---|\n')
        for figure in figures: out.write(f'| `figures/{figure}` | Numerical visualization |\n')
        out.write('\n### Observed PI80 coverage summary\n\n')
        out.write(pd.DataFrame(covrows).round(3).to_markdown(index=False)+'\n\n') if hasattr(pd.DataFrame,'to_markdown') else out.write(pd.DataFrame(covrows).to_string(index=False)+'\n\n')
        out.write('### Selected synthetic controlled upward-shift comparison\n\n')
        out.write('See `tables/experiment_15_numeric_detector_comparison.csv` for the full numerical comparison. '\
                  'High simulated detection alone does not establish a superior real-world model-health detector.\n')
    return figures


_original_main = main

def main():
    _original_main()
    a=arguments()
    root=locate_project_root(a)
    generated=expanded_figures(root)
    print('\nADDITIONAL EXPERIMENT 15 FIGURES')
    for x in generated: print(' - figures/'+x)
    print('Extra numeric tables: tables/experiment_15_numeric_detector_comparison.csv, '
          'tables/experiment_15_coverage_numeric_summary.csv')


if __name__ == '__main__':
    main()
