from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"market_regime_results"/"lear_v3_calendar_plots"
PERIODS=[
("baseline","Jul 2025 – Feb 2026","2025-07-01","2026-02-28"),
("march_april","Mar – Apr 2026","2026-03-01","2026-04-30"),
("may_june","May – Jun 2026","2026-05-01","2026-06-30"),
("july","July 2026","2026-07-01","2026-07-31"),
("aug_oct","Aug – Oct 2026","2026-08-01","2026-10-13")]

def easter(y):
    a=y%19;b=y//100;c=y%100;d=b//4;e=b%4;f=(b+8)//25;g=(b-f+1)//3
    h=(19*a+b-d-g+15)%30;i=c//4;k=c%4;l=(32+2*e+2*i-h-k)%7
    m=(a+11*h+22*l)//451;n=h+l-7*m+114
    return date(y,n//31,n%31+1)

def holidays(y):
    fixed={(1,1):"New Year",(1,6):"Epiphany",(4,25):"Liberation Day",
           (5,1):"Labour Day",(6,2):"Republic Day",(8,15):"Assumption",
           (11,1):"All Saints",(12,8):"Immaculate Conception",
           (12,25):"Christmas",(12,26):"St Stephen"}
    h={date(y,m,d):name for (m,d),name in fixed.items()}
    e=easter(y);h[e]="Easter";h[e+timedelta(days=1)]="Easter Monday"
    return h

def make_plot(df,key,title,start,end,daily):
    part=df[df["Date"].between(start.date(),end.date())].copy()
    if part.empty:
        print("No data:",title);return
    if daily:
        part=part.groupby("Date",as_index=False)[["Actual","Predicted"]].mean()
        x=pd.to_datetime(part["Date"])+pd.Timedelta(hours=12)
    else:
        x=part["Local"]
    fig,ax=plt.subplots(figsize=(17,6))
    first=True
    for day in pd.date_range(start,end,freq="D"):
        if day.weekday()>=5:
            ax.axvspan(day,day+pd.Timedelta(days=1),color="gray",alpha=.13,
                       label="Weekend" if first else None)
            first=False
    h={}
    for year in range(start.year,end.year+1):h.update(holidays(year))
    relevant=[(d,n) for d,n in sorted(h.items()) if start.date()<=d<=end.date()]
    for i,(day,name) in enumerate(relevant):
        xh=pd.Timestamp(day)+pd.Timedelta(hours=12)
        ax.axvline(xh,color="firebrick",alpha=.8,lw=1,
                   label="Italian national holiday" if i==0 else None)
        ax.annotate(name,xy=(xh,1),xycoords=("data","axes fraction"),
                    xytext=(2,-5-(i%3)*14),textcoords="offset points",
                    rotation=90,va="top",ha="left",fontsize=7,color="firebrick",
                    bbox=dict(facecolor="white",edgecolor="none",alpha=.65,pad=1))
    ax.plot(x,part["Actual"],color="tab:blue",lw=1.4 if daily else .8,
            label="Actual price")
    ax.plot(x,part["Predicted"],color="tab:orange",ls="--",
            lw=1.4 if daily else .8,label="LEAR v3 prediction")
    ax.set_xlim(start,end+pd.Timedelta(days=1))
    ax.set(title=f"{title} — {'daily mean' if daily else 'hourly'} prices",
           ylabel="Price (EUR/MWh)",xlabel="Italian local date")
    ax.grid(axis="y",alpha=.2)
    if (end-start).days>120:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    elif (end-start).days>45:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper left",ncol=4,fontsize=9)
    fig.autofmt_xdate()
    fig.text(.01,.01,"Gray = weekend; red = national holiday; local patron-saint holidays excluded.",
             fontsize=8,color="gray")
    fig.tight_layout(rect=(0,.025,1,1))
    path=OUT/f"{key}_{'daily' if daily else 'hourly'}.png"
    fig.savefig(path,dpi=160);plt.close(fig)
    print("Saved",path.name)

def main():
    src=ROOT/"ENTSOE_LEAR_v3_predictions.csv"
    if not src.exists():raise FileNotFoundError(f"Missing {src}; put script in Entsoe dataRev")
    df=pd.read_csv(src,usecols=["Target_UTC","Target_Price_EUR_MWh",
                                "LEAR_v3_Prediction_EUR_MWh"])
    utc=pd.to_datetime(df["Target_UTC"],utc=True)
    if utc.duplicated().any():raise ValueError("Duplicate timestamps")
    df["Local"]=utc.dt.tz_convert("Europe/Rome").dt.tz_localize(None)
    df["Date"]=df["Local"].dt.date
    df["Actual"]=pd.to_numeric(df["Target_Price_EUR_MWh"])
    df["Predicted"]=pd.to_numeric(df["LEAR_v3_Prediction_EUR_MWh"])
    df=df.sort_values("Local")
    OUT.mkdir(parents=True,exist_ok=True)
    for key,title,lo,hi in PERIODS:
        for daily in (False,True):
            make_plot(df,key,title,pd.Timestamp(lo),pd.Timestamp(hi),daily)
    print("Output folder:",OUT)

if __name__=="__main__":main()
