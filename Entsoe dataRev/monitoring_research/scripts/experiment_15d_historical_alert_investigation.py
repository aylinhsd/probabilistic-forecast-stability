#!/usr/bin/env python3
"""Experiment 15d — Historical alert investigation and chronological controlled tests.

Run from the LEAR project root:
 python3 monitoring_research/scripts/experiment_15d_historical_alert_investigation.py
Or explicitly:
 python3 monitoring_research/scripts/experiment_15d_historical_alert_investigation.py \
   --forecasts /path/to/ENTSOE_Probabilistic_LEAR_v3_predictions.csv \
   --point-forecasts /path/to/ENTSOE_LEAR_v3_predictions.csv

Dependencies: numpy, pandas, matplotlib; optional river for ADWIN.
Put the companion experiment_15c_reliable_detector_selection.py in the same scripts folder.
This experiment never modifies source data or calls observed historical alerts true/false.
"""
from __future__ import annotations
import argparse, importlib.util, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
CANDIDATES=[HERE/'experiment_15c_reliable_detector_selection.py',HERE.parent/'experiment_15c_reliable_detector_selection.py',Path('/mnt/data/experiment_15c_reliable_detector_selection.py')]
CORE=next((p for p in CANDIDATES if p.exists()),None)
if CORE is None: raise FileNotFoundError('Place experiment_15c_reliable_detector_selection.py beside this script')
spec=importlib.util.spec_from_file_location('exp15c_core', CORE);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)

SELECTED={'Fixed P95':{'q':.99},'CUSUM':{'h':5,'k':.5},'Page-Hinkley':{'h':8,'delta':.15},'EWMA':{'L':2.5,'alpha':.2}}
if core.RIVER: SELECTED['ADWIN']={'delta':.002} # exploratory, not 15c-calibrated
SIGNALS=core.SIGNALS

def cli():
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--forecasts',type=Path,default=None)
 p.add_argument('--point-forecasts',type=Path,default=None)
 p.add_argument('--root',type=Path,default=None,help='Output monitoring_research directory')
 p.add_argument('--fit-days',type=int,default=160)
 p.add_argument('--reference-end',default='2026-02-28')
 p.add_argument('--cooldown',type=int,default=7)
 p.add_argument('--episode-gap',type=int,default=3,help='Number of clear days allowed inside an alert episode')
 p.add_argument('--prepost',type=int,default=14)
 p.add_argument('--allow-synthetic-fixture',action='store_true',help='ALLOW clearly artificial forecast file for smoke tests only')
 return p.parse_args()

def paths(a):
 root=a.root or (HERE.parent if HERE.name=='scripts' else Path.cwd()/'monitoring_research')
 root=root.resolve();[ (root/q).mkdir(parents=True,exist_ok=True) for q in ['figures','tables','reports'] ]
 return root

def locate(root,name,explicit):
 if explicit:return explicit
 for p in [root.parent/name,root/name,Path.cwd()/name,Path('/mnt/data/exp15b_audit')/name]:
  if p.exists():return p
 return None

def daily_with_context(fpath,point):
 f=pd.read_csv(fpath)
 if 'Split' in f:f=f.loc[f.Split.astype(str).str.lower()=='test'].copy()
 f['time_utc']=pd.to_datetime(f['Target_UTC'],utc=True)
 f['date']=f.time_utc.dt.tz_convert('Europe/Rome').dt.tz_localize(None).dt.normalize()
 qs=sorted([(float(c[2:]),c) for c in f if c.startswith('Q_')],key=lambda x:x[0]);lv=np.array([x[0] for x in qs]);q=f[[x[1] for x in qs]].to_numpy(float)
 y=f['Target_Price_EUR_MWh'].to_numpy(float)
 crps=2*np.trapezoid(np.where((y[:,None]-q)>=0,lv[None,:]*(y[:,None]-q),(lv[None,:]-1)*(y[:,None]-q)),lv,axis=1) if hasattr(np,'trapezoid') else 2*np.trapz(np.where((y[:,None]-q)>=0,lv[None,:]*(y[:,None]-q),(lv[None,:]-1)*(y[:,None]-q)),lv,axis=1)
 def lookup(lev):return q[:,np.where(np.isclose(lv,lev))[0][0]]
 low=lookup(.1);high=lookup(.9);mid=lookup(.5)
 f=f[['time_utc','date']].copy();f['actual']=y;f['crps']=crps;f['coverage']=((y>=low)&(y<=high)).astype(float);f['width']=high-low;f['median_ae']=np.abs(y-mid)
 f['point_ae']=np.nan
 if point and point.exists():
  pp=pd.read_csv(point,usecols=['Target_UTC','LEAR_v3_Prediction_EUR_MWh']);pp['time_utc']=pd.to_datetime(pp.Target_UTC,utc=True)
  pp=pp.drop_duplicates('time_utc');f=f.merge(pp[['time_utc','LEAR_v3_Prediction_EUR_MWh']],on='time_utc',how='left')
  f['point_ae']=np.abs(f.actual-f['LEAR_v3_Prediction_EUR_MWh']);f=f.drop(columns=['LEAR_v3_Prediction_EUR_MWh'])
 out=f.groupby('date',sort=True).agg(crps=('crps','mean'),coverage=('coverage','mean'),width=('width','mean'),observed_volatility=('actual','std'),median_mae=('median_ae','mean'),point_mae=('point_ae','mean'),hours=('actual','size')).reset_index()
 out['relative_point_mae']=out['median_mae']/out['point_mae'].replace(0,np.nan)
 return out

def episode_segments(raw_all,start,end,gap):
 """Episode clustering based on all raw component alert days and <= gap unalerted days.
 Episodes are for narrative/diagnostics, NOT independent ground-truth events.
 """
 ii=np.flatnonzero(raw_all[start:end])+start
 if len(ii)==0:return []
 groups=[]; cur=[int(ii[0])]
 for i in ii[1:]:
  if i-cur[-1] <= gap+1:cur.append(int(i))
  else:groups.append(cur);cur=[int(i)]
 groups.append(cur)
 return groups

def assessment(row):
 """Transparent descriptive contextual flags; never make truth claims."""
 flags=[]
 if row['post_CRPS_ref_p95_fraction']>=.5:flags.append('persistent_high_CRPS')
 if row['post_PI80_abs_dev']>=.08:flags.append('substantial_PI80_deviation')
 if pd.notna(row['post_relative_point_mae']) and row['post_relative_point_mae']>1.1:flags.append('median_MAE_worse_than_point_baseline')
 if pd.notna(row['post_volatility']) and row['post_volatility']>row['ref_volatility_p75']:flags.append('high_market_volatility')
 if len(flags)==0: cls='no_strong_contextual_evidence'
 elif 'high_market_volatility' in flags and not any(x in flags for x in ['substantial_PI80_deviation','median_MAE_worse_than_point_baseline']): cls='high_volatility_context_only'
 else:cls='performance_deviation_present_unverified_cause'
 return cls,';'.join(flags)

def run(a):
 root=paths(a);forecast=locate(root,'ENTSOE_Probabilistic_LEAR_v3_predictions.csv',a.forecasts);point=locate(root,'ENTSOE_LEAR_v3_predictions.csv',a.point_forecasts)
 if forecast is None:raise FileNotFoundError('Real probabilistic forecasts missing; provide --forecasts')
 d=daily_with_context(forecast,point)
 # An earlier model-audit fixture imitates the real file name but contains fixed-width quantiles
 # and a point prediction exactly equal to the quantile median. It is not empirical LEAR output.
 fixture=(d.width.std()<1e-6 and d.point_mae.notna().any() and np.allclose(d.median_mae,d.point_mae,rtol=0,atol=1e-10))
 if fixture and not a.allow_synthetic_fixture:
  raise ValueError('Input looks like the SYNTHETIC audit fixture: constant PI80 width and point-model MAE exactly equals quantile-median MAE. Supply the real LEAR v3 forecast CSV, or run --allow-synthetic-fixture for a labeled smoke test only.')
 d=d.sort_values('date').reset_index(drop=True);R=(d.date<=pd.Timestamp(a.reference_end)).sum();N=len(d);K=a.fit_days
 if R< K+20 or N-R<50: raise ValueError('Insufficient reference/evaluation data')
 X=core.signals(d);S=core.scales(X,K);RZ={k:core.norm(X[k][:K],S[k]) for k in SIGNALS}
 methods=list(SELECTED)
 alerts=[];episodes=[];raw_summ=[];allraw={};allnotes={}; q95=float(d.crps.iloc[:K].quantile(.95));vol75=float(d.observed_volatility.iloc[:K].quantile(.75))
 for meth in methods:
  raw,note=core.notifications(X,RZ,meth,SELECTED[meth],S,a.cooldown)
  allraw[meth]=raw;allnotes[meth]=note;union=raw.any(axis=1)
  for i in range(R,N):
   if union[i] or note[i]: alerts.append({'date':d.date.iloc[i].date().isoformat(),'method':meth,'raw_alert':int(union[i]),'notification':int(note[i]),'signals':';'.join(k for j,k in enumerate(SIGNALS) if raw[i,j]),'crps':d.crps.iloc[i],'pi80_coverage':d.coverage.iloc[i],'pi80_width':d.width.iloc[i],'observed_volatility':d.observed_volatility.iloc[i]})
  segs=episode_segments(union,R,N,a.episode_gap)
  for idx,g in enumerate(segs,1):
   st,ed=g[0],g[-1];post=d.iloc[st:min(N,ed+1+a.prepost)];pre=d.iloc[max(0,st-a.prepost):st]
   rr={'method':meth,'event_id':f'{meth.replace(" ","_")}_{idx:03d}','first_alert_date':str(d.date.iloc[st].date()),'last_alert_date':str(d.date.iloc[ed].date()),'span_calendar_days':int(ed-st+1),'raw_alert_days':len(g),'notifications_in_episode':int(note[st:ed+1].sum()),
      'trigger_signals':';'.join(k for j,k in enumerate(SIGNALS) if raw[g,j].any()),'pre_CRPS':float(pre.crps.mean()) if len(pre) else np.nan,'post_CRPS':float(post.crps.mean()),'post_CRPS_ref_p95_fraction':float((post.crps>q95).mean()),
      'pre_PI80':float(pre.coverage.mean()) if len(pre) else np.nan,'post_PI80':float(post.coverage.mean()),'post_PI80_abs_dev':float(abs(post.coverage.mean()-.8)),
      'pre_volatility':float(pre.observed_volatility.mean()) if len(pre) else np.nan,'post_volatility':float(post.observed_volatility.mean()),'ref_volatility_p75':vol75,
      'pre_relative_point_mae':float(pre.relative_point_mae.mean()) if len(pre) else np.nan,'post_relative_point_mae':float(post.relative_point_mae.mean()),'ground_truth':'UNAVAILABLE'}
   rr['context_category'],rr['context_flags']=assessment(rr);episodes.append(rr)
  raw_summ.append({'method':meth,'evaluation_raw_alert_days':int(union[R:].sum()),'evaluation_notifications':int(note[R:].sum()),'evaluation_episodes':len(segs),'evaluation_days':N-R,'configured_cooldown_days':a.cooldown,'historical_false_alarm_rate':'NOT IDENTIFIABLE WITHOUT GROUND TRUTH'})
 alerts=pd.DataFrame(alerts);eps=pd.DataFrame(episodes);summary=pd.DataFrame(raw_summ)
 # Cross-detector grouping: merge all raw alert days into one overall incident timeline.
 global_union=np.zeros(N,bool)
 for raw in allraw.values():global_union |= raw.any(axis=1)
 combined=[]
 for i,g in enumerate(episode_segments(global_union,R,N,a.episode_gap),1):
  st,ed=g[0],g[-1];active_methods=[m for m in methods if allraw[m][g,:].any()]
  active_signals=[k for j,k in enumerate(SIGNALS) if any(allraw[m][g,j].any() for m in methods)]
  sub=eps[(eps.first_alert_date<=str(d.date.iloc[ed].date())) & (eps.last_alert_date>=str(d.date.iloc[st].date()))]
  combined.append({'incident_id':f'INC_{i:03d}','start_date':str(d.date.iloc[st].date()),'end_date':str(d.date.iloc[ed].date()),'span_calendar_days':ed-st+1,'active_raw_alert_days':len(g),'detectors':';'.join(active_methods),'number_detectors':len(active_methods),'trigger_signals':';'.join(active_signals),'related_per_method_episodes':len(sub),'confirmed_deterioration':'UNKNOWN'})
 incidents=pd.DataFrame(combined,columns=['incident_id','start_date','end_date','span_calendar_days','active_raw_alert_days','detectors','number_detectors','trigger_signals','related_per_method_episodes','confirmed_deterioration'])
 # Dedicated chronological real-series controlled change tests, using untouched calendar order.
 # First 160 baseline days fit detector, chronological reference tail for testing; no bootstrapping.
 # NOTE: reference tail is not certified healthy, so unmodified alert rate != true false alarm.
 chrono=[]
 kinds=('sustained','gradual','temporary');magnitudes=(.75,1.5,2.5)
 fit={k:v[:K] for k,v in X.items()};ref_tail={k:v[K:R] for k,v in X.items()};L=R-K
 onsets=[o for o in [14,28,42] if o+30 < L]
 for onset in onsets:
  for sig in SIGNALS:
   for kind in kinds:
    for magnitude in magnitudes:
     changed=core.inject(ref_tail,sig,onset,kind,S[sig][1],magnitude)
     for method in methods:
      braw,bnote=core.notifications(core.make_case(fit,ref_tail),RZ,method,SELECTED[method],S,a.cooldown)
      craw,cnote=core.notifications(core.make_case(fit,changed),RZ,method,SELECTED[method],S,a.cooldown)
      start=K+onset;end=min(K+L,start+31)
      bc=np.flatnonzero(bnote[start:end]);cc=np.flatnonzero(cnote[start:end])
      chrono.append({'method':method,'signal':sig,'scenario':kind,'shift_multiple_robust_scale':magnitude,'onset_date':str(d.date.iloc[start].date()),'baseline_alert_in_window':int(len(bc)>0),'injected_alert_in_window':int(len(cc)>0),'incremental_detected_without_paired_baseline_alert':int(bool(len(cc)) and not bool(len(bc))),'delay_days_if_detected':int(cc[0]) if len(cc) else np.nan,'note':'Chronological reference-tail perturbation, not verified real degradation'})
 chrono=pd.DataFrame(chrono)
 chrono_summary=chrono.groupby('method',as_index=False).agg(tests=('injected_alert_in_window','size'),detection_pct=('injected_alert_in_window',lambda x:round(100*x.mean(),1)),paired_baseline_alert_pct=('baseline_alert_in_window',lambda x:round(100*x.mean(),1)),incremental_detection_pct=('incremental_detected_without_paired_baseline_alert',lambda x:round(100*x.mean(),1)),median_detected_delay_days=('delay_days_if_detected','median'))
 # Summaries by week and monthly diagnostics.
 monthly=d.iloc[R:].copy();monthly['month']=monthly.date.dt.to_period('M').astype(str)
 monthly=monthly.groupby('month',as_index=False).agg(mean_crps=('crps','mean'),mean_pi80_coverage=('coverage','mean'),mean_width=('width','mean'),mean_observed_volatility=('observed_volatility','mean'),mean_median_mae=('median_mae','mean'),mean_point_mae=('point_mae','mean'))
 for method in methods:
  masks=np.array([d.date.iloc[i].strftime('%Y-%m') for i in range(R,N)])
  counts={mon:int(allnotes[method][R:][masks==mon].sum()) for mon in monthly.month}
  monthly[f'notifications_{method.replace("-","_").replace(" ","_")}']=monthly.month.map(counts)
 # Save
 t=root/'tables';p=root/'figures';rep=root/'reports'
 artifacts={'daily':d,'alerts':alerts,'episodes':eps,'method_summary':summary,'monthly':monthly,'chronological_tests':chrono,'chronological_summary':chrono_summary,'cross_detector_incidents':incidents}
 for k,v in artifacts.items():v.to_csv(t/f'experiment_15d_{k}.csv',index=False)
 # plot 1: actual time series + notified days for each method, no metric causality assumed
 fig,axes=plt.subplots(3,1,figsize=(14,10),sharex=True)
 ev=d.iloc[R:];axes[0].plot(ev.date,ev.crps,lw=1.1,label='Observed daily CRPS');axes[0].axhline(q95,ls='--',color='0.4',label='Fit CRPS P95');axes[0].set(ylabel='CRPS (EUR/MWh)',title='Experiment 15d — Real historical monitoring outcomes and notifications');axes[0].legend(loc='upper left',fontsize=8)
 axes[1].plot(ev.date,ev.coverage,color='#287b7e',lw=1,label='Daily PI80 coverage');axes[1].plot(ev.date,ev.coverage.rolling(14,min_periods=7).mean(),color='#83558c',lw=2,label='14-day rolling coverage');axes[1].axhline(.8,ls='--',color='0.3');axes[1].set(ylabel='Coverage',ylim=(0,1));axes[1].legend(loc='lower left',fontsize=8)
 axes[2].plot(ev.date,ev.observed_volatility,color='#bb7f40',lw=1,label='Observed within-day volatility');axes[2].set(ylabel='Volatility (EUR/MWh)',xlabel='Delivery date');axes[2].legend(loc='upper left',fontsize=8)
 colors=plt.cm.tab10(np.linspace(0,1,len(methods)))
 crps_bottom=axes[0].get_ylim()[0]
 for m,c in zip(methods,colors):
  ids=np.flatnonzero(allnotes[m][R:]);dates=ev.date.iloc[ids]
  axes[0].vlines(dates,crps_bottom,ev.crps.iloc[ids],color=c,linewidth=1.2,label=m,zorder=3)
 axes[0].set_ylim(bottom=crps_bottom)
 # legend method markers on subplot 0
 h,l=axes[0].get_legend_handles_labels();axes[0].legend(h,l,ncol=3,fontsize=7,loc='upper right')
 fig.tight_layout();fig.savefig(p/'experiment_15d_real_timeline.png',dpi=170);plt.close(fig)
 # method v date raster
 fig,ax=plt.subplots(figsize=(14,4));
 for i,m in enumerate(methods):
  for idx in np.flatnonzero(allnotes[m][R:]):ax.scatter(ev.date.iloc[idx],i,marker='|',s=95)
 ax.set_yticks(range(len(methods)),methods);ax.set(xlabel='Delivery date',title='Unlabeled historical notifications (real chronological evaluation)');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(p/'experiment_15d_notification_calendar.png',dpi=170);plt.close(fig)
 # Episode counts versus raw/notification burden
 fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(summary));w=.27
 for j,col in enumerate(['evaluation_raw_alert_days','evaluation_notifications','evaluation_episodes']):ax.bar(x+(j-1)*w,summary[col],width=w,label=col.replace('evaluation_','').replace('_',' '))
 ax.set_xticks(x,summary.method,rotation=15,ha='right');ax.set(ylabel='Counts (different units; compare within series)',title='From raw alert days to grouped incidents');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(p/'experiment_15d_alert_grouping.png',dpi=170);plt.close(fig)
 # Context by episode categories
 if not eps.empty:
  pivot=eps.groupby(['method','context_category']).size().unstack(fill_value=0).reindex(methods,fill_value=0)
  fig,ax=plt.subplots(figsize=(11,5));pivot.plot.bar(stacked=True,ax=ax);ax.set(xlabel='',ylabel='Episodes',title='Historical episode evidence categories — NOT true/false labels');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(p/'experiment_15d_context_assessment.png',dpi=170);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(chrono_summary));ax.bar(x-.2,chrono_summary.incremental_detection_pct,width=.4,label='Incremental injected change detections');ax.bar(x+.2,chrono_summary.paired_baseline_alert_pct,width=.4,label='Paired unmodified window alerts');ax.set_xticks(x,chrono_summary.method,rotation=15,ha='right');ax.set(ylabel='Percentage of paired tests',title='Real chronological reference-tail tests: synthetic indicator changes');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(p/'experiment_15d_chronological_controlled_tests.png',dpi=170);plt.close(fig)
 # explanatory report
 context=eps.context_category.value_counts().to_dict() if not eps.empty else {}
 data_warning=('**SMOKE-TEST ONLY: INPUT IS A SYNTHETIC AUDIT FIXTURE. ALL NUMERICAL RESULTS ARE DEMONSTRATIONS, NOT REAL LEAR RESULTS.**\n\n' if fixture else '')
 msg=f'''# Experiment 15d — Historical Alert Investigation and Validation\n\n{data_warning}## Goal and source\n\nAnalyse original chronological LEAR v3 monitoring results **before** interpreting synthetic detector rankings. Inputs: `{forecast.name}`; separate point-forecast MAE reference: `{point.name if point else 'not provided'}`.\n\n**Time split:** Fit = {d.date.iloc[0].date()} to {d.date.iloc[K-1].date()} ({K} days); remaining reference = {d.date.iloc[K].date()} to {d.date.iloc[R-1].date()} ({R-K} days); historical evaluation = {d.date.iloc[R].date()} to {d.date.iloc[-1].date()} ({N-R} days).\n\nHistorical timestamps are NEVER shuffled, synthetic perturbations occur only in separate copied arrays.\n\n## 1. Real alerts and grouping\n\n```text\n{summary.to_string(index=False)}\n```\n\n'Raw alert days' are days where at least one monitored signal fired. Notifications use a seven-day cooldown; descriptive episodes combine raw alert days separated by up to {a.episode_gap} clear days. Episodes are NOT confirmed deterioration events.\n\n## Combined incidents across all detectors\n\nTotal combined descriptive incidents: {len(incidents)}. `experiment_15d_cross_detector_incidents.csv` clusters alerts across different detector methods into one overall timeline. These remain UNLABELED.\n\n## 2. Context around episodes\n\n{json.dumps(context,indent=2)}\n\nFor each episode, `experiment_15d_episodes.csv` reports preceding {a.prepost}-day means and the episode plus following {a.prepost} days: CRPS, PI80 coverage, realized volatility, a point-forecast MAE comparison and the originating monitoring signal(s). Evidence categories are heuristic, descriptive, and **not ground truth**. The separate point model is not a probabilistic CRPS benchmark.\n\n## 3. Real-world true and false alarms\n\nCannot be established with current records alone. We have no independently adjudicated model-failure labels. High volatility is contextual evidence, not proof alerts are false; high CRPS is a monitoring signal, not independent verification. Assess against documented model failures, expert review, and prospective tests of model updates.\n\n## 4. Controlled tests on original chronological data (not block-mixed)\n\nFor transparency, artificial signal-level sustained, gradual, and five-day temporary changes were added to COPIES of the unshuffled final {R-K} reference days. Tested injected and paired unmodified sequences with the same detector configurations; incremental detection is counted only if the paired unmodified series did not alert in the same window.\n\n```text\n{chrono_summary.to_string(index=False)}\n```\n\nThis is not a genuine no-change reference: chronological reference-tail days may already include changes. Thus paired unmodified alerts are NOT established false alarms. These tests have only {len(onsets)} onset positions and are a sensitivity check, not an independent detector ranking.\n\n## 5. Conclusions and follow-up\n\n- Examine which real episodes coincide with larger CRPS, poor PI80 calibration, or benchmark skill loss; do not label them causal or confirmed model deterioration.\n- Distinguish raw alert days, post-cooldown notifications, and grouped episodes.\n- Analyse all held-out 15c synthetic-null and controlled tests to compare sensitivity under common budget.\n- Add forecast-level perturbations (Experiment 15b) for coherent CRPS/coverage changes.\n- Obtain independently documented degradation periods or expert labels before computing real false alarm rates.\n- Do not automate retraining or claim a best detector from this analysis alone.\n- The detector named `Fixed P95` reuses the Experiment 15c family label, but its selected `q=0.99` setting corresponds to a **99th-percentile** threshold. Do not describe its current 15d cutoff as P95.\n- ADWIN is omitted unless independently calibrated; the original 15c run had no `river` ADWIN results.\n\n## Output files\n\n- `tables/experiment_15d_daily.csv`: actual chronological monitoring indicators.\n- `tables/experiment_15d_alerts.csv`: individual alert days and signals.\n- `tables/experiment_15d_episodes.csv`: grouped events and pre/post evidence.\n- `tables/experiment_15d_method_summary.csv`: operational burden.\n- `tables/experiment_15d_monthly.csv`: monthly conditions and notifications.\n- `tables/experiment_15d_chronological_tests.csv`: controlled perturbations on original order.\n- `tables/experiment_15d_chronological_summary.csv`: paired test outcomes.\n- `figures/experiment_15d_*.png`: five diagnostic figures.\n'''
 (rep/'experiment_15d_report.md').write_text(msg)
 (t/'experiment_15d_metadata.json').write_text(json.dumps({'source':str(forecast),'point_source':str(point) if point else None,'fit_days':int(K),'ref_days':int(R),'eval_days':int(N-R),'reference_end':a.reference_end,'core_script':str(CORE),'synthetic_fixture':bool(fixture),'methods_and_settings':SELECTED,'note':'Real alerts unlabeled; chronological controlled tests change only copied indicator arrays.'},indent=2))
 print('Experiment 15d completed. Original data untouched.');print('SYNTHETIC FIXTURE (NOT REAL RESULTS):',fixture);print(summary.to_string(index=False));print('Chronological synthetic:',chrono_summary.to_string(index=False));print('Output:',root)
 return summary
if __name__=='__main__':run(cli())
