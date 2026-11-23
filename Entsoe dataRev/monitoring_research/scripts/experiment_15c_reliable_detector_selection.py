#!/usr/bin/env python3
"""Experiment 15c — reliable detector selection under a common notification budget.

Inputs (one of):
  --forecasts ENTSOE_Probabilistic_LEAR_v3_predictions.csv
  --daily monitoring_research/tables/experiment_14_daily_monitoring.csv
Defaults search the existing project layout. Outputs go to monitoring_research/{tables,figures,reports}.

Scientific scope:
  - Forecast-derived daily metrics are REAL; historic notifications are UNLABELED.
  - Controlled signal-level stress tests are SYNTHETIC and do not constitute
    forecast-level validation or real-world false alarm estimates.
  - Null sequences are 7-day block-bootstrap samples from the FIT baseline.
  - Detector thresholds are chosen ONLY using calibration null simulations.
  - A common overall notification budget is imposed on the UNION across signals.
  - Results are descriptive for this configuration, not a conclusive best method.
  - No future outcomes are used to produce retrospective health alarms; this is
    NOT genuine early warning, and no retraining action is automated.

Requirements: numpy, pandas, matplotlib. Optional: river (ADWIN).
Run: python3 monitoring_research/scripts/experiment_15c_reliable_detector_selection.py
     python3 monitoring_research/scripts/experiment_15c_reliable_detector_selection.py --forecasts /path/to/predictions.csv
     python3 monitoring_research/scripts/experiment_15c_reliable_detector_selection.py --demo
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from river.drift import ADWIN
    RIVER=True
except ImportError:
    ADWIN=None
    RIVER=False

SIGNALS=['absolute_crps','coverage_under','coverage_over','width_high']
METHODS=['Fixed P95','CUSUM','Page-Hinkley','EWMA']+(['ADWIN'] if RIVER else [])
SEED=15032026


def args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    g=p.add_mutually_exclusive_group()
    g.add_argument('--forecasts',type=Path)
    g.add_argument('--daily',type=Path)
    g.add_argument('--demo',action='store_true')
    p.add_argument('--root',type=Path,help='monitoring_research directory; default inferred from script path')
    p.add_argument('--reference-end',default='2026-02-28')
    p.add_argument('--fit-days',type=int,default=160)
    p.add_argument('--budget-days',type=float,default=30,help='Target <=1 combined notification per N null days')
    p.add_argument('--cooldown',type=int,default=7)
    p.add_argument('--null-cal-trials',type=int,default=12)
    p.add_argument('--null-test-trials',type=int,default=25)
    p.add_argument('--change-trials',type=int,default=25)
    p.add_argument('--event-day',type=int,default=65)
    p.add_argument('--shift-multipliers',type=float,nargs='+',default=[.75,1.5,2.5],help='Change size in baseline robust scales')
    p.add_argument('--window',type=int,default=30)
    p.add_argument('--block-days',type=int,default=7)
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--quick',action='store_true',help='Faster smoke test with 4 trials each')
    return p.parse_args()


def root_path(a):
    if a.root:return a.root.expanduser().resolve()
    parent=Path(__file__).resolve().parent
    if parent.name=='scripts' and parent.parent.name=='monitoring_research': return parent.parent
    return Path.cwd()/'monitoring_research'


def forecast_daily(path):
    f=pd.read_csv(path)
    if 'Split' in f:f=f[f['Split'].astype(str).str.casefold()=='test'].copy()
    needed={'Target_UTC','Target_Price_EUR_MWh'}
    if not needed.issubset(f):raise ValueError('Missing forecast columns: '+str(needed-set(f)))
    qs=sorted([(float(c[2:]),c) for c in f if c.startswith('Q_')],key=lambda x:x[0])
    levels=np.asarray([q for q,_ in qs])
    if len(qs)<20 or not np.all(np.diff(levels)>0):raise ValueError('Quantile grid is insufficient or unsorted')
    y=f['Target_Price_EUR_MWh'].to_numpy(float)
    q=f[[c for _,c in qs]].to_numpy(float)
    if not np.isfinite(y).all() or not np.isfinite(q).all():raise ValueError('Non-finite forecast entries')
    if 0.10 not in levels or 0.90 not in levels:raise ValueError('Q_0.10 and Q_0.90 are required')
    lo=q[:,np.where(levels==.10)[0][0]];hi=q[:,np.where(levels==.90)[0][0]]
    if 'Target_Date_Italy' in f:dates=pd.to_datetime(f['Target_Date_Italy']).dt.normalize()
    else:dates=pd.to_datetime(f['Target_UTC'],utc=True).dt.tz_convert('Europe/Rome').dt.tz_localize(None).dt.normalize()
    # Two times integral of quantile loss, finite approximation to CRPS.
    u=y[:,None]-q
    loss=np.where(u>=0,levels[None,:]*u,(levels[None,:]-1)*u)
    crps=2*np.trapezoid(loss,levels,axis=1) if hasattr(np,'trapezoid') else 2*np.trapz(loss,levels,axis=1)
    df=pd.DataFrame({'date':dates,'crps':crps,'coverage':((y>=lo)&(y<=hi)).astype(float),'width':hi-lo,'actual':y})
    day=df.groupby('date',sort=True).agg(crps=('crps','mean'),coverage=('coverage','mean'),width=('width','mean'),observed_volatility=('actual','std'),hours=('actual','size')).reset_index()
    return day


def normalize_daily(path):
    d=pd.read_csv(path)
    rename={}
    names={'date':['date','delivery_date','Target_Date_Italy'],'crps':['crps','absolute_crps','daily_crps','mean_crps'],
           'coverage':['pi80_coverage','coverage80','coverage_80','coverage','coverage_80_pct'],
           'width':['pi80_width','interval_width','width','mean_pi80_width']}
    for target,aliases in names.items():
        col=next((k for k in aliases if k in d),None)
        if col is None: raise ValueError(f'Cannot find {target}; columns = {list(d.columns)}')
        rename[col]=target
    d=d.rename(columns=rename)
    d['date']=pd.to_datetime(d['date'],utc=True).dt.tz_localize(None).dt.normalize()
    for k in ['crps','coverage','width']:d[k]=pd.to_numeric(d[k],errors='raise')
    if d.coverage.max()>1 and d.coverage.max()<=100:d['coverage']/=100
    return d.sort_values('date').reset_index(drop=True)


def demo_daily(seed):
    rng=np.random.default_rng(seed)
    n=470;dates=pd.date_range('2025-07-01',periods=n,freq='D');v=np.zeros(n)
    for i in range(1,n):v[i]=.75*v[i-1]+rng.normal()
    vol=np.maximum(6,17+3*v+np.where(np.arange(n)>=243,12,0))
    crps=np.maximum(.3,5+0.2*vol+rng.normal(0,.7,n))
    coverage=np.clip(.82-0.003*(vol-17)+rng.normal(0,.04,n),.01,.99)
    width=np.maximum(1,25+.45*vol+rng.normal(0,3,n))
    return pd.DataFrame({'date':dates,'crps':crps,'coverage':coverage,'width':width,'observed_volatility':vol,'hours':24})


def validate(d,a):
    d=d.sort_values('date').reset_index(drop=True)
    if d.date.duplicated().any():raise ValueError('Duplicate days in input')
    if d[['crps','coverage','width']].isna().any().any():raise ValueError('Missing core daily indicator(s)')
    if not np.isfinite(d[['crps','coverage','width']].to_numpy(float)).all():raise ValueError('Non-finite monitoring values')
    if not d.coverage.between(0,1).all():raise ValueError('Coverage outside [0,1]')
    if (d.crps<0).any() or (d.width<0).any():raise ValueError('Negative CRPS or width')
    ref=d.date<=pd.Timestamp(a.reference_end)
    if ref.sum()<=a.fit_days+20 or (~ref).sum()<35:raise ValueError('Insufficient chronological reference/evaluation data')
    if a.cooldown<0 or a.event_day<15 or a.window<1 or a.block_days<1:raise ValueError('Invalid event/cooldown settings')
    return d,ref.to_numpy()


def signals(d):
    # Directional calibration signals use 14-day rolling empirical coverage.
    # This smooths daily binomial noise and ensures BOTH under/over deviations are checked.
    cov=d.coverage.rolling(14,min_periods=7).mean().bfill().to_numpy(float)
    return {'absolute_crps':d.crps.to_numpy(float),
            'coverage_under':np.maximum(0,.8-cov),
            'coverage_over':np.maximum(0,cov-.8),
            'width_high':d.width.to_numpy(float)}


def scales(X,fit_n):
    out={}
    for k,x in X.items():
        r=np.asarray(x[:fit_n],float);m=np.median(r);s=1.4826*np.median(np.abs(r-m))
        if s<1e-5:s=float(np.std(r))
        out[k]=(float(m),max(float(s),.005 if k.startswith('coverage') else .1))
    return out


def norm(x,ms):return (np.asarray(x,float)-ms[0])/ms[1]


def configurations(method):
    if method=='Fixed P95':return [{'q':v} for v in [.9,.95,.975,.99,.995,1.0]]
    if method=='CUSUM':return [{'h':h,'k':.5} for h in [3,5,8,12,18,25,35,50,75]]
    if method=='Page-Hinkley':return [{'h':h,'delta':.15} for h in [3,5,8,12,18,25,35,50,75]]
    if method=='EWMA':return [{'L':v,'alpha':.2} for v in [1.5,2,2.5,3,3.5,4,5,6,8,11]]
    if method=='ADWIN':return [{'delta':v} for v in [.1,.05,.02,.01,.005,.002,.001,.0001,.00001]]
    raise ValueError(method)


def detection_one(z,method,cfg,reference_z):
    # z starts with warm-up reference followed by evaluation; alert outputs include warm-up.
    n=len(z);alarms=np.zeros(n,dtype=bool)
    if method=='Fixed P95':
        threshold=float(np.quantile(reference_z,cfg['q']))
        return z>threshold
    if method=='CUSUM':
        acc=0.
        for i,v in enumerate(z):
            acc=max(0,acc+v-cfg['k'])
            if acc>cfg['h']:alarms[i]=True;acc=0.
    elif method=='Page-Hinkley':
        total=0.;minimum=0.
        for i,v in enumerate(z):
            total+=v-cfg['delta'];minimum=min(minimum,total)
            if total-minimum>cfg['h']:alarms[i]=True;total=0.;minimum=0.
    elif method=='EWMA':
        st=0.;threshold=cfg['L']*np.sqrt(cfg['alpha']/(2-cfg['alpha']))
        for i,v in enumerate(z):
            st=cfg['alpha']*v+(1-cfg['alpha'])*st
            if st>threshold:alarms[i]=True;st=0.
    elif method=='ADWIN':
        model=ADWIN(delta=cfg['delta'])
        for i,v in enumerate(z):
            model.update(float(v))
            if model.drift_detected:alarms[i]=True
    return alarms


def notifications(signals_data,ref_z,method,cfg,stats,cooldown):
    """Union all component alarms; suppress recurring alerts within shared cooldown."""
    n=len(next(iter(signals_data.values())))
    raw=np.zeros((n,len(SIGNALS)),bool)
    for j,key in enumerate(SIGNALS):
        z=norm(signals_data[key],stats[key])
        raw[:,j]=detection_one(z,method,cfg,ref_z[key])
    union=raw.any(axis=1)
    notified=np.zeros(n,bool);last=-10**9
    for i,active in enumerate(union):
        if active and i-last>cooldown: notified[i]=True;last=i
    return raw,notified


def resample_blocks(X,n,block,rng):
    L=len(next(iter(X.values())));indices=[]
    while len(indices)<n:
        start=int(rng.integers(0,L))
        indices.extend([(start+j)%L for j in range(block)])
    indices=np.array(indices[:n]);return {k:v[indices].copy() for k,v in X.items()}


def build_case(base_fit,eval_n,block,rng):
    return resample_blocks(base_fit,eval_n,block,rng)


def make_case(fit,ev):return {k:np.r_[fit[k],ev[k]] for k in SIGNALS}


def sample_rate(fit,eval_signals,stats,ref_z,method,cfg,cooldown):
    combined=make_case(fit,eval_signals)
    _,note=notifications(combined,ref_z,method,cfg,stats,cooldown)
    return int(note[len(next(iter(fit.values()))):].sum())


def inject(ev,key,onset,kind,scale,magnitude):
    e={k:v.copy() for k,v in ev.items()};n=len(e[key])
    if kind=='sustained':e[key][onset:]+=magnitude*scale
    elif kind=='gradual':e[key][onset:]+=magnitude*scale*np.minimum(1,np.arange(n-onset)/30)
    elif kind=='temporary':e[key][onset:onset+5]+=magnitude*scale
    else:raise ValueError(kind)
    return e


def evaluate(a):
    root=root_path(a);(root/'tables').mkdir(parents=True,exist_ok=True)
    (root/'figures').mkdir(parents=True,exist_ok=True);(root/'reports').mkdir(parents=True,exist_ok=True)
    source=''
    if a.demo:
        d=demo_daily(a.seed);source='SYNTHETIC DEMO DATA: no empirical LEAR conclusions'
    elif a.daily:
        d=normalize_daily(a.daily);source=f'Daily input: {a.daily.name}'
    else:
        forecast=a.forecasts
        if not forecast:
            locations=[root.parent/'ENTSOE_Probabilistic_LEAR_v3_predictions.csv',Path.cwd()/'ENTSOE_Probabilistic_LEAR_v3_predictions.csv']
            forecast=next((f for f in locations if f.exists()),None)
        if forecast and Path(forecast).exists():
            d=forecast_daily(forecast);source=f'Actual saved probabilistic forecast outputs: {Path(forecast).name}'
        else:
            default=root/'tables'/'experiment_14_daily_monitoring.csv'
            if not default.exists():raise FileNotFoundError('Cannot find forecast/daily input; use --forecasts PATH or --daily PATH')
            d=normalize_daily(default);source=f'Daily input: {default.name}'
    d,ref=validate(d,a)
    end=int(ref.sum());fit_n=a.fit_days;eval_n=int((~ref).sum())
    fit_dates=(str(d.date.iloc[0].date()),str(d.date.iloc[fit_n-1].date()))
    cal_dates=(str(d.date.iloc[fit_n].date()),str(d.date.iloc[end-1].date()))
    eva_dates=(str(d.date.iloc[end].date()),str(d.date.iloc[-1].date()))
    X=signals(d);fit={k:v[:fit_n] for k,v in X.items()};stats=scales(X,fit_n)
    rz={k:norm(fit[k],stats[k]) for k in SIGNALS}
    rng=np.random.default_rng(a.seed)
    ncal=4 if a.quick else a.null_cal_trials
    ntest=4 if a.quick else a.null_test_trials
    nshift=4 if a.quick else a.change_trials
    # Synthetic nulls for threshold CALIBRATION, based ONLY on fit data.
    # Calibration calendar interval is reported but unused for tuning; bootstraps
    # guard against falsely treating potential deviations there as verified null.
    null_cal=[build_case(fit,eval_n,a.block_days,rng) for _ in range(ncal)]
    target=eval_n/a.budget_days
    selections={};fitrows=[]
    for method in METHODS:
        candidates=[]
        for j,cfg in enumerate(configurations(method)):
            counts=[sample_rate(fit,ev,stats,rz,method,cfg,a.cooldown) for ev in null_cal]
            mean=float(np.mean(counts))
            candidates.append((cfg,mean,float(np.std(counts)),j))
            fitrows.append({'method':method,'candidate':j,'settings':json.dumps(cfg),'cal_null_mean_notifications':mean,
                            'cal_null_notifications_per_100_days':mean/eval_n*100,'budget_per_100_days':100/a.budget_days})
        eligible=[t for t in candidates if t[1]<=target]
        if eligible:
            chosen=max(eligible,key=lambda t:(t[1],-t[3]));meets=True
        else:
            chosen=min(candidates,key=lambda t:(t[1],t[3]));meets=False
        selections[method]={'settings':chosen[0],'cal_null_mean_notifications':chosen[1],
                            'cal_null_rate_per_100_days':chosen[1]/eval_n*100,'meets_cal_budget':meets}
    # Holdout synthetic null tests from *fresh* independent draws.
    null_test=[build_case(fit,eval_n,a.block_days,rng) for _ in range(ntest)]
    nullrows=[]
    for method in METHODS:
        for r,ev in enumerate(null_test):
            count=sample_rate(fit,ev,stats,rz,method,selections[method]['settings'],a.cooldown)
            nullrows.append({'method':method,'trial':r,'notifications':count,'notifications_per_100_days':count/eval_n*100})
    # Controlled changes on paired independent null trajectories (one indicator at a time).
    # The artificial change is known; it does NOT prove genuine model degradation.
    if a.event_day+a.window>=eval_n:raise ValueError('Evaluation period too short for event and detection horizon')
    scenarios=['sustained','gradual','temporary']
    trials=[]
    for trial in range(nshift):
        ev=build_case(fit,eval_n,a.block_days,rng)
        for key in SIGNALS:
            for kind in scenarios:
                for magnitude in a.shift_multipliers:
                    changed=inject(ev,key,a.event_day,kind,stats[key][1],magnitude)
                    for method in METHODS:
                        cfg=selections[method]['settings']
                        basecase=make_case(fit,ev);newcase=make_case(fit,changed)
                        _,base_note=notifications(basecase,rz,method,cfg,stats,a.cooldown)
                        _,new_note=notifications(newcase,rz,method,cfg,stats,a.cooldown)
                        st=fit_n+a.event_day;ed=st+a.window+1
                        detections=np.flatnonzero(new_note[st:ed]);null_detections=np.flatnonzero(base_note[st:ed])
                        trials.append({'method':method,'trial':trial,'signal':key,'scenario':kind,'shift_magnitude':magnitude,
                                       'detected_30d':int(len(detections)>0),
                                       'detection_delay_days':int(detections[0]) if len(detections) else np.nan,
                                       'paired_null_notification_30d':int(len(null_detections)>0),
                                       'incremental_detection_30d':int(bool(len(detections)>0) and not bool(len(null_detections)>0)),
                                       'before_onset_notifications':int(new_note[fit_n:st].sum())})
    null=pd.DataFrame(nullrows);sim=pd.DataFrame(trials)
    historical=[]
    for method in METHODS:
        _,notified=notifications(X,rz,method,selections[method]['settings'],stats,a.cooldown)
        # critical: do not count reference days as historical evaluation alerts.
        for i in range(end,len(d)):
            if notified[i]: historical.append({'method':method,'date':d.date.iloc[i].strftime('%Y-%m-%d'),
                                               'status':'UNLABELED: not known true or false'})
    hist=pd.DataFrame(historical,columns=['method','date','status'])
    summary=[]
    for method in METHODS:
        nr=null[null.method==method];sr=sim[sim.method==method]
        sens=sr[sr.scenario=='sustained'];grad=sr[sr.scenario=='gradual'];temp=sr[sr.scenario=='temporary']
        weak=sens[sens.shift_magnitude==min(a.shift_multipliers)]; moderate=sens[sens.shift_magnitude==sorted(a.shift_multipliers)[len(a.shift_multipliers)//2]]
        histcount=int((hist.method==method).sum())
        summary.append({'method':method,'settings':json.dumps(selections[method]['settings']),
                        'cal_budget_met':selections[method]['meets_cal_budget'],
                        'cal_null_notifications_per_100_days':round(selections[method]['cal_null_rate_per_100_days'],2),
                        'heldout_null_notifications_per_100_days':round(float(nr.notifications_per_100_days.mean()),2),
                        'sustained_detected_30d_pct':round(float(sens.detected_30d.mean()*100),1),
                        'weak_sustained_detected_30d_pct':round(float(weak.detected_30d.mean()*100),1),
                        'moderate_sustained_detected_30d_pct':round(float(moderate.detected_30d.mean()*100),1),
                        'gradual_detected_30d_pct':round(float(grad.detected_30d.mean()*100),1),
                        'temporary_detected_30d_pct':round(float(temp.detected_30d.mean()*100),1),
                        'sustained_median_delay_detected_days':round(float(sens.detection_delay_days.median()),1) if sens.detected_30d.any() else np.nan,
                        'incremental_sustained_detection_pct':round(float(sens.incremental_detection_30d.mean()*100),1),
                        'paired_null_notifications_in_event_window_pct':round(float(sens.paired_null_notification_30d.mean()*100),1),
                        'historical_evaluation_notifications_unlabeled':histcount,
                        'historical_notif_per_100_days':round(100*histcount/eval_n,2)})
    summary=pd.DataFrame(summary).sort_values(['weak_sustained_detected_30d_pct','heldout_null_notifications_per_100_days'],ascending=[False,True])
    table=root/'tables';fig=root/'figures';report=root/'reports'
    d.to_csv(table/'experiment_15c_daily_metrics.csv',index=False)
    pd.DataFrame(fitrows).to_csv(table/'experiment_15c_calibration_grid.csv',index=False)
    null.to_csv(table/'experiment_15c_heldout_null_trials.csv',index=False)
    sim.to_csv(table/'experiment_15c_controlled_change_trials.csv',index=False)
    hist.to_csv(table/'experiment_15c_historical_notifications_UNLABELED.csv',index=False)
    summary.to_csv(table/'experiment_15c_method_comparison.csv',index=False)
    figobj,ax=plt.subplots(figsize=(11,5))
    sr=summary.sort_values('heldout_null_notifications_per_100_days')
    xpos=np.arange(len(sr));ax.bar(xpos,sr.heldout_null_notifications_per_100_days)
    ax.axhline(100/a.budget_days,linestyle='--',color='black',label='Target null notification budget')
    ax.set_xticks(xpos,sr.method,rotation=15,ha='right');ax.set_ylabel('Notifications / 100 synthetic no-change days')
    ax.set_title('Experiment 15c — held-out synthetic null notification rate');ax.legend();figobj.tight_layout()
    figobj.savefig(fig/'experiment_15c_heldout_null_notifications.png',dpi=180);plt.close(figobj)
    figobj,ax=plt.subplots(figsize=(11,5))
    sr=summary.sort_values('weak_sustained_detected_30d_pct',ascending=False)
    xpos=np.arange(len(sr));ax.bar(xpos-.19,sr.weak_sustained_detected_30d_pct,width=.38,label='Weak sustained shifts')
    ax.bar(xpos+.19,sr.moderate_sustained_detected_30d_pct,width=.38,label='Moderate sustained shifts')
    ax.set_xticks(xpos,sr.method,rotation=15,ha='right');ax.set_ylabel('Synthetic detection rate within 30 days (%)');ax.set_ylim(0,105)
    ax.set_title('Experiment 15c — weak and moderate signal-shift detection');ax.legend();figobj.tight_layout()
    figobj.savefig(fig/'experiment_15c_detection_rates.png',dpi=180);plt.close(figobj)
    figobj,ax=plt.subplots(figsize=(10,6))
    for _,r in summary.iterrows():
        ax.scatter(r.heldout_null_notifications_per_100_days,r.weak_sustained_detected_30d_pct,s=100)
        ax.annotate(r.method,(r.heldout_null_notifications_per_100_days,r.weak_sustained_detected_30d_pct),xytext=(4,5),textcoords='offset points')
    ax.axvline(100/a.budget_days,linestyle='--',color='black',label='Budget')
    ax.set(xlabel='Held-out null notifications / 100 days (lower is better)',ylabel='Weak sustained change detection within 30 days (%)',
           title='Experiment 15c — detection vs notification trade-off');ax.set_ylim(-5,105);ax.legend();figobj.tight_layout()
    figobj.savefig(fig/'experiment_15c_tradeoff.png',dpi=180);plt.close(figobj)
    figobj,ax=plt.subplots(figsize=(11,5))
    histcounts=[int((hist.method==method).sum()) for method in METHODS]
    ax.bar(METHODS,histcounts);ax.set(ylabel='Historical evaluation days with notification',title='Experiment 15c — historical alerts are UNLABELED');ax.set_ylim(0,max(histcounts+[1])*1.16)
    for i,n in enumerate(histcounts):ax.text(i,n+1,str(n),ha='center')
    figobj.tight_layout();figobj.savefig(fig/'experiment_15c_historical_alert_burden.png',dpi=180);plt.close(figobj)
    metadata={'experiment':'15c','source':source,'reference_end':a.reference_end,'fit_period':fit_dates,
              'calibration_days_present_but_not_used_for_null_training':cal_dates,'evaluation_period':eva_dates,
              'fit_days':fit_n,'reference_total_days':end,'evaluation_days':eval_n,'calibration_strategy':'Block-bootstrap on FIT baseline only; no evaluation leakage',
              'threshold_selection':'Closest calibrated mean notification count at or under target on synthetic null; otherwise most conservative and flag',
              'budget_notifications_per_100_null_days':100/a.budget_days,'cooldown_days':a.cooldown,
              'random_seed':a.seed,'calibration_null_trials':ncal,'heldout_null_trials':ntest,'controlled_change_trials_per_signal_scenario':nshift,
              'block_days':a.block_days,'shift_multipliers':a.shift_multipliers,'event_day':a.event_day,'detection_window_days':a.window,
              'signals':SIGNALS,'methods':METHODS,'river_ADWIN_available':RIVER,
              'warning':'Historical notifications are unlabeled. Bootstrap no-change days may not reproduce true no-change conditions. Synthetic signal shifts are not coherent forecast perturbations. Do not interpret as real false-alarm rate or auto-retrain recommendation.'}
    (table/'experiment_15c_metadata.json').write_text(json.dumps(metadata,indent=2))
    with (report/'experiment_15c_report.md').open('w') as f:
        f.write('# Experiment 15c — Reliable detector selection under a common notification budget\n\n')
        f.write(f'**Input:** {source}  \n**Fit period:** {fit_dates[0]} to {fit_dates[1]} ({fit_n} days)  \n')
        f.write(f'**Reference remainder (not assumed verified null):** {cal_dates[0]} to {cal_dates[1]}  \n')
        f.write(f'**Historical evaluation:** {eva_dates[0]} to {eva_dates[1]} ({eval_n} days)  \n')
        f.write(f'**Target:** <=1 grouped notification per {a.budget_days:g} synthetic null days, with {a.cooldown} day cooldown.\n\n')
        f.write('## Controlled test method\n\n')
        f.write('1. Fit only on the first chronological reference days. Bootstrap contiguous 7-day baseline blocks to construct no-change series.\n')
        f.write('2. For each detector, select parameters on calibration synthetic null trials, not test trials. Match the **combined** notification budget across four signals.\n')
        f.write('3. Evaluate held-out synthetic no-change sequences and inject signal-level sustained, gradual, and temporary changes with known onset.\n')
        f.write('4. Save unverified historical notifications separately; no historical alert is called a false alarm.\n\n')
        f.write('## Method comparison\n\n```\n'+summary.to_string(index=False)+'\n```\n\n')
        f.write('## Interpretation and limitations\n\n')
        f.write('- `heldout_null_notifications_per_100_days` describes **synthetic no-change** notifications, not confirmed real-world false alarms.\n')
        f.write('- `sustained_detected_30d_pct` is the fraction of **injected signal shifts** detected within 30 days, irrespective of whether a paired null series would also have alerted.\n')
        f.write('- `incremental_sustained_detection_pct` is stricter: records an injected-series notification only when the same paired no-change series had none in that window.\n')
        f.write('- Historical notifications may be justified or unjustified; no ground-truth failure labels were supplied.\n')
        f.write('- Simulation changes only ONE monitoring signal at a time; it does not modify raw forecasts consistently. For coherent forecast-level validation, use Experiment 15b.\n')
        f.write('- The original reference period was selected after volatility inspection. The fit window may include seasonal or behavioral patterns not representative of a stationary null.\n')
        f.write('- The selected detector settings are research candidates, not deployment-ready settings; evaluate in several chronological folds and on external models.\n')
        f.write('- Upward risks of coverage below/above nominal are separate signals; high interval width can also reflect appropriate uncertainty.\n\n')
        f.write('## Next steps\n\n1. Audit signal definitions against Exp 14/15; investigate intervals and regime-aware thresholds.\n')
        f.write('2. Validate and integrate forecast-level perturbations from Exp 15b, including unchanged forecast nulls.\n')
        f.write('3. Add documented real model deterioration labels or external expert adjudication.\n')
        f.write('4. Choose a detector based on held-out detection, delayed/missed events, and allowed notifications, not synthetic detection alone.\n')
        f.write('5. Add a separate action-policy study before recommending recalibration/retraining.\n')
    print('Experiment 15c complete')
    print('Input:',source)
    print('Fit/reference/evaluation:',fit_n,end,eval_n)
    print(summary[['method','cal_budget_met','heldout_null_notifications_per_100_days','weak_sustained_detected_30d_pct','historical_evaluation_notifications_unlabeled']].to_string(index=False))
    print('Outputs:',root)
    return summary

if __name__=='__main__':
    a=args();evaluate(a)
