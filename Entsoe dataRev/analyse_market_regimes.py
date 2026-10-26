#!/usr/bin/env python3
"""Exploratory Italian North electricity-price forecast regime/calendar analysis.

Run in the same folder as the four prediction CSVs:
    python analyze_market_regimes.py

No retraining. Reconstructs probabilistic baselines from Jan-Jun 2025 residuals.
Evaluation: July 2025 onward. All models restricted to common UTC hours.
"""
import calendar
import json
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "market_regime_results"
SOURCES = {
    "LEAR_v3": ("ENTSOE_LEAR_v3_predictions.csv", ["LEAR_v3_Prediction_EUR_MWh"]),
    "Previous_LEAR": ("ENTSOE_improved_LEAR_predictions.csv", ["Improved_LEAR_Prediction_EUR_MWh"]),
    "LEAR_style": ("ENTSOE_LEAR_style_predictions.csv", ["LEAR_Style_Prediction_EUR_MWh"]),
    "Seasonal_naive": ("ENTSOE_seasonal_naive_predictions.csv", ["Predicted_Price_EUR_MWh"]),
}
QS = np.array(sorted(set(np.round(np.arange(.01, 1, .01), 3)) | {.075, .925}))
CAL_END = date(2025, 6, 30)
TEST_START = date(2025, 7, 1)
PERIODS = [
    ("Baseline_Jul25_Feb26", date(2025,7,1), date(2026,2,28)),
    ("Transition_MarApr26", date(2026,3,1), date(2026,4,30)),
    ("Elevated_MayJun26", date(2026,5,1), date(2026,6,30)),
    ("July26", date(2026,7,1), date(2026,7,31)),
    ("Later_AugOct26", date(2026,8,1), date(2026,10,31)),
]

def easter(year):
    # Gregorian Easter Sunday, Meeus/Jones/Butcher algorithm
    a=year%19; b=year//100; c=year%100; d=b//4; e=b%4
    f=(b+8)//25; g=(b-f+1)//3
    h=(19*a+b-d-g+15)%30; i=c//4; k=c%4
    l=(32+2*e+2*i-h-k)%7; m=(a+11*h+22*l)//451
    month=(h+l-7*m+114)//31
    day=((h+l-7*m+114)%31)+1
    return date(year,month,day)

def holidays(year):
    # National Italian holidays. Local patron-saint holidays excluded.
    fixed = {
        (1,1): "New Year's Day", (1,6): "Epiphany",
        (4,25): "Liberation Day", (5,1): "Labour Day",
        (6,2): "Republic Day", (8,15): "Assumption",
        (11,1): "All Saints", (12,8): "Immaculate Conception",
        (12,25): "Christmas", (12,26): "St Stephen's Day",
    }
    h={date(year,mo,da):name for (mo,da),name in fixed.items()}
    es=easter(year)
    h[es]="Easter Sunday"
    h[es+timedelta(days=1)]="Easter Monday"
    return h

def load(name, filename, preferred):
    df=pd.read_csv(ROOT/filename,low_memory=False)
    pred=next((c for c in preferred if c in df.columns),None)
    if pred is None:
        options=[c for c in df.columns if "pred" in c.lower() and
                 "error" not in c.lower() and "quantile" not in c.lower()]
        if len(options)!=1:
            raise ValueError(f"{name}: ambiguous prediction columns {options}")
        pred=options[0]
    ts=pd.to_datetime(df["Target_UTC"],utc=True,errors="raise")
    local=ts.dt.tz_convert("Europe/Rome")
    dates=local.dt.date
    y=pd.to_numeric(df["Target_Price_EUR_MWh"],errors="raise").to_numpy(float)
    p=pd.to_numeric(df[pred],errors="raise").to_numpy(float)
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError(f"{name}: non-finite prices/predictions")
    if ts.duplicated().any():
        raise ValueError(f"{name}: duplicate UTC timestamps")
    val=((dates>=date(2025,1,1))&(dates<=CAL_END)).to_numpy()
    test=(dates>=TEST_START).to_numpy()
    if not val.any() or not test.any():
        raise ValueError(f"{name}: missing validation or test observations")
    residual_quantiles=np.quantile(y[val]-p[val],QS)
    qhat=p[test,None]+residual_quantiles[None,:]
    actual=y[test]
    diff=actual[:,None]-qhat
    pinball=np.maximum(QS[None,:]*diff,(QS[None,:]-1)*diff)
    crps=2*np.trapezoid(pinball,QS,axis=1)
    out=pd.DataFrame({
        "Model":name, "UTC":ts[test].astype(str).to_numpy(),
        "Date":dates[test].astype(str).to_numpy(),
        "Hour_Italy":local[test].dt.hour.to_numpy(),
        "Actual":actual,"Point":p[test],
        "Signed_Error":actual-p[test], "Absolute_Error":np.abs(actual-p[test]),
        "Squared_Error":(actual-p[test])**2,
        "Approx_CRPS":crps, "Mean_Pinball":pinball.mean(axis=1),
    })
    for nominal,lo,hi in [(80,.10,.90),(85,.075,.925),(90,.05,.95)]:
        li=np.where(np.isclose(QS,lo))[0][0]
        ui=np.where(np.isclose(QS,hi))[0][0]
        out[f"PI{nominal}_Hit"]=((actual>=qhat[:,li])&(actual<=qhat[:,ui])).astype(int)
        out[f"PI{nominal}_Width"]=qhat[:,ui]-qhat[:,li]
    print(f"{name}: validation {val.sum()}, test {test.sum()}, prediction={pred}",flush=True)
    return out

def summarize(frame, columns):
    records=[]
    for keys,part in frame.groupby(columns,dropna=False,sort=True):
        if not isinstance(keys,tuple): keys=(keys,)
        rec=dict(zip(columns,keys))
        rec.update({
            "N_Hours":len(part),
            "N_Days":part.Date.nunique(),
            "Actual_Mean":part.Actual.mean(),
            "Actual_SD":part.Actual.std(),
            "Actual_Min":part.Actual.min(),
            "Actual_Max":part.Actual.max(),
            "MAE":part.Absolute_Error.mean(),
            "RMSE":np.sqrt(part.Squared_Error.mean()),
            "Bias_ActualMinusPred":part.Signed_Error.mean(),
            "Approx_CRPS":part.Approx_CRPS.mean(),
            "Mean_Pinball":part.Mean_Pinball.mean(),
        })
        for level in (80,85,90):
            rec[f"PI{level}_Coverage_Pct"]=100*part[f"PI{level}_Hit"].mean()
            rec[f"PI{level}_Coverage_Error_PP"]=rec[f"PI{level}_Coverage_Pct"]-level
            rec[f"PI{level}_Width"]=part[f"PI{level}_Width"].mean()
        records.append(rec)
    return pd.DataFrame(records)

def draw_lines(data, x, y, title, filename, reference=None):
    fig,ax=plt.subplots(figsize=(12,4.5))
    for name,g in data.groupby("Model"):
        ax.plot(pd.to_datetime(g[x]),g[y],label=name,linewidth=1.35)
    if reference is not None:
        ax.axhline(reference,color="black",linestyle="--",linewidth=1)
    ax.set(title=title,xlabel="Italy-local date",ylabel=y)
    ax.legend(ncol=2,fontsize=8)
    fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(OUT/filename,dpi=160)
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    frames=[]
    for name,(filename,preds) in SOURCES.items():
        if not (ROOT/filename).exists():
            raise FileNotFoundError(f"Missing {filename} in {ROOT}")
        frames.append(load(name,filename,preds))
    df=pd.concat(frames,ignore_index=True)
    sets=[set(g.UTC) for _,g in df.groupby("Model")]
    common=set.intersection(*sets)
    if not common: raise ValueError("No shared UTC timestamps")
    df=df[df.UTC.isin(common)].copy()
    consistency=df.groupby("UTC").Actual.agg(["min","max"])
    if not np.allclose(consistency["min"],consistency["max"],atol=1e-8):
        raise ValueError("Actual prices differ between models")
    df["Date_dt"]=pd.to_datetime(df.Date)
    df["Weekday"]=df.Date_dt.dt.day_name()
    df["Weekend"]=df.Date_dt.dt.weekday>=5
    df["Month"]=df.Date_dt.dt.month
    df["Season"]=df.Month.map({
        12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
        6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"})
    all_holidays={}
    for year in range(df.Date_dt.dt.year.min(),df.Date_dt.dt.year.max()+1):
        all_holidays.update(holidays(year))
    df["Holiday_Name"]=df.Date_dt.dt.date.map(all_holidays).fillna("")
    df["Holiday"]=df.Holiday_Name.ne("")
    df["Day_Type"]=np.select(
        [df.Holiday & ~df.Weekend,df.Holiday & df.Weekend,df.Weekend],
        ["Weekday_holiday","Weekend_holiday","Ordinary_weekend"],
        default="Ordinary_weekday")
    df["Adjacent_Holiday"]=df.Date_dt.dt.date.map(
        lambda d: (d-timedelta(days=1) in all_holidays or
                   d+timedelta(days=1) in all_holidays) and d not in all_holidays)
    df["Period"]="Outside_defined_period"
    for label,start,end in PERIODS:
        df.loc[df.Date_dt.between(pd.Timestamp(start),pd.Timestamp(end)),"Period"]=label

    # Realized volatility: within-day standard deviation and mean absolute
    # successive hourly change. Calculated once per day using actual prices.
    unique=df[df.Model.eq("LEAR_v3")].sort_values("UTC").copy()
    daily_price=unique.groupby("Date").agg(
        Daily_Price_Mean=("Actual","mean"),
        Daily_Price_SD=("Actual","std"),
        Daily_Price_Min=("Actual","min"),
        Daily_Price_Max=("Actual","max"),
        N_Hours=("Actual","size")).reset_index()
    daily_price["Daily_Price_Range"]=daily_price.Daily_Price_Max-daily_price.Daily_Price_Min
    abs_changes=unique.groupby("Date").Actual.apply(
        lambda s: float(np.abs(np.diff(s.to_numpy())).mean()) if len(s)>1 else np.nan)
    daily_price["Daily_Mean_Abs_Hourly_Change"]=daily_price.Date.map(abs_changes)
    baseline=daily_price[(daily_price.Date>="2025-07-01")&
                         (daily_price.Date<="2026-02-28")]
    # Baseline thresholds held fixed for all later periods.
    volatility_cuts=np.quantile(baseline.Daily_Price_SD.dropna(),[1/3,2/3])
    price_cuts=np.quantile(baseline.Daily_Price_Mean.dropna(),[1/3,2/3])
    if volatility_cuts[0]>=volatility_cuts[1] or price_cuts[0]>=price_cuts[1]:
        raise ValueError("Baseline tertile thresholds are not distinct")
    daily_price["Volatility_Regime"]=pd.cut(daily_price.Daily_Price_SD,
        [-np.inf,*volatility_cuts,np.inf],labels=["Low","Medium","High"])
    daily_price["Price_Regime"]=pd.cut(daily_price.Daily_Price_Mean,
        [-np.inf,*price_cuts,np.inf],labels=["Low","Medium","High"])
    df=df.merge(daily_price,on="Date",validate="many_to_one")
    df["Negative_Price_Hour"]=df.Actual<0
    df["Extreme_Price_Hour"]=(
        (df.Actual < np.quantile(unique[unique.Date.between("2025-07-01","2026-02-28")].Actual,.05)) |
        (df.Actual > np.quantile(unique[unique.Date.between("2025-07-01","2026-02-28")].Actual,.95))
    )

    # All summaries use hourly-level losses and hits, not daily averages.
    groupings={
        "period_comparison":["Model","Period"],
        "weekday_weekend":["Model","Day_Type"],
        "weekday_detail":["Model","Weekday"],
        "holidays":["Model","Holiday_Name"],
        "holiday_adjacent":["Model","Adjacent_Holiday"],
        "hour_of_day":["Model","Hour_Italy"],
        "month":["Model","Month"],
        "season":["Model","Season"],
        "volatility_regime":["Model","Volatility_Regime"],
        "price_regime":["Model","Price_Regime"],
        "negative_price":["Model","Negative_Price_Hour"],
        "extreme_price":["Model","Extreme_Price_Hour"],
        "period_by_day_type":["Model","Period","Day_Type"],
        "period_by_volatility":["Model","Period","Volatility_Regime"],
    }
    for label,cols in groupings.items():
        summarize(df,cols).to_csv(OUT/f"{label}.csv",index=False)
    daily=summarize(df,["Model","Date"])
    daily.to_csv(OUT/"daily_model_metrics.csv",index=False)
    daily_price.to_csv(OUT/"daily_market_conditions.csv",index=False)
    df.to_csv(OUT/"hourly_evaluation.csv",index=False)

    for period,start,end in [
        ("march_april_2026",date(2026,3,1),date(2026,4,30)),
        ("july_2026",date(2026,7,1),date(2026,7,31)),
    ]:
        sel=daily[daily.Date.between(str(start),str(end))]
        for metric in ("MAE","Approx_CRPS","PI80_Coverage_Pct"):
            draw_lines(sel,"Date",metric,f"{period}: daily {metric}",
                       f"{period}_{metric}.png",
                       80 if metric=="PI80_Coverage_Pct" else None)
        market=daily_price[daily_price.Date.between(str(start),str(end))]
        for col in ("Daily_Price_Mean","Daily_Price_SD"):
            fig,ax=plt.subplots(figsize=(12,4))
            ax.plot(pd.to_datetime(market.Date),market[col],linewidth=1.5)
            ax.set(title=f"{period}: {col}",xlabel="Italy-local date",ylabel="EUR/MWh")
            fig.autofmt_xdate();fig.tight_layout()
            fig.savefig(OUT/f"{period}_{col}.png",dpi=160)
            plt.close(fig)

    meta={
        "scope":"Exploratory subgroup analysis; not a causal attribution or formal change-point test",
        "models":list(SOURCES),"common_test_hours":len(common),
        "validation_period":"2025-01-01 to 2025-06-30, Italy local",
        "test_period_start":"2025-07-01, Italy local",
        "probabilistic_method":"pooled validation residual quantile shifts",
        "crps":"2*trapezoidal integration of pinball losses over Q01..Q99; tails omitted",
        "regime_threshold_source":"Jul 2025-Feb 2026 daily realized prices (test-period exploratory baseline)",
        "volatility_tertile_cutoffs":volatility_cuts.tolist(),
        "price_tertile_cutoffs":price_cuts.tolist(),
        "volatility_definition":"standard deviation of realized hourly prices within each local day",
        "price_definition":"mean realized hourly price within each local day",
        "holiday_scope":"Italian national public holidays; Easter Sunday and Monday; no regional holidays",
        "cautions":[
            "Regime labels use realized prices: descriptive only, not information available at forecast issue time",
            "Period boundaries selected after viewing test outcomes; exploratory, not confirmatory",
            "Daily PI coverage is noisy with roughly 24 hourly observations",
            "Holiday and extreme-price subsets may be small",
            "All probabilistic baseline intervals have constant widths",
            "Month/season comparisons confounded with changing year/regime",
            "No statistical significance or causal claim is implied",
        ],
    }
    (OUT/"methodology.json").write_text(json.dumps(meta,indent=2)+"\n")
    overview=summarize(df,["Model","Period"])
    print("\nPERIOD COMPARISON")
    print(overview[["Model","Period","N_Hours","MAE","Approx_CRPS",
                    "PI80_Coverage_Pct","PI80_Width"]].to_string(index=False))
    print(f"\nSaved outputs: {OUT}",flush=True)

if __name__=="__main__":
    main()
