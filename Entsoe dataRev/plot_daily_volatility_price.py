"""Plot observed within-day electricity price volatility, with periods and holidays."""
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "ENTSOE_LEAR_v3_predictions.csv"
OUT = ROOT / "market_regime_results"
PERIODS = [
    ("Baseline", "2025-07-01", "2026-02-28"),
    ("March–April", "2026-03-01", "2026-04-30"),
    ("May–June", "2026-05-01", "2026-06-30"),
    ("July", "2026-07-01", "2026-07-31"),
    ("August–October", "2026-08-01", "2026-10-13"),
]

def easter(year):
    a=year%19; b=year//100; c=year%100; d=b//4; e=b%4
    f=(b+8)//25; g=(b-f+1)//3; h=(19*a+b-d-g+15)%30
    i=c//4; k=c%4; l=(32+2*e+2*i-h-k)%7
    m=(a+11*h+22*l)//451; n=h+l-7*m+114
    return date(year,n//31,n%31+1)

def italian_holidays(year):
    fixed={(1,1):"New Year",(1,6):"Epiphany",(4,25):"Liberation Day",
           (5,1):"Labour Day",(6,2):"Republic Day",(8,15):"Assumption",
           (11,1):"All Saints",(12,8):"Immaculate Conception",
           (12,25):"Christmas",(12,26):"St Stephen"}
    result={date(year,m,d):name for (m,d),name in fixed.items()}
    e=easter(year)
    result[e]="Easter"; result[e+timedelta(days=1)]="Easter Monday"
    return result

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Expected {INPUT}. Save this script in Entsoe dataRev.")
    df=pd.read_csv(INPUT,usecols=["Target_UTC","Target_Price_EUR_MWh"])
    df["utc"]=pd.to_datetime(df["Target_UTC"],utc=True,errors="raise")
    if df["utc"].duplicated().any():
        raise ValueError("Duplicate UTC timestamps found.")
    df["local_day"]=df["utc"].dt.tz_convert("Europe/Rome").dt.date
    df["price"]=pd.to_numeric(df["Target_Price_EUR_MWh"],errors="coerce")
    if df["price"].isna().any():
        raise ValueError("Missing or invalid actual price values.")
    daily=df.groupby("local_day").agg(
        daily_sd_eur_mwh=("price","std"),
        mean_price_eur_mwh=("price","mean"),
        hourly_observations=("price","size")).reset_index()
    daily["date"]=pd.to_datetime(daily["local_day"])
    start=pd.Timestamp("2025-07-01")
    end=pd.Timestamp("2026-10-13")
    daily=daily[daily["date"].between(start,end)].copy()
    if daily.empty:
        raise ValueError("No price observations in the selected periods.")
    daily["rolling_7d_sd"]=daily["daily_sd_eur_mwh"].rolling(7,min_periods=1).mean()
    OUT.mkdir(exist_ok=True,parents=True)
    daily.to_csv(OUT/"daily_price_volatility.csv",index=False)

    fig,ax=plt.subplots(figsize=(18,7))
    ax.plot(daily["date"],daily["daily_sd_eur_mwh"],color="#9aa4b2",
            lw=1,alpha=.85,label="Daily within-day price SD")
    ax.plot(daily["date"],daily["rolling_7d_sd"],color="#1466a3",
            lw=2,label="7-day rolling mean of daily SD")
    period_colors=["#e9f3fa","#fff4e5","#fcebec","#e8f5e9","#f1edfa"]
    for idx,(name,lo,hi) in enumerate(PERIODS):
        a=pd.Timestamp(lo); b=pd.Timestamp(hi)+pd.Timedelta(days=1)
        ax.axvspan(a,b,facecolor=period_colors[idx],alpha=.55,zorder=-2)
        subset=daily[daily["date"].between(a,b-pd.Timedelta(days=1))]
        if not subset.empty:
            mean=subset["daily_sd_eur_mwh"].mean()
            ax.hlines(mean,a,b,color="#b45f06",lw=2.1,linestyles="dashed",
                      label="Period average daily SD" if idx==0 else None)
            ax.text(a+(b-a)/2,1.035,name,transform=ax.get_xaxis_transform(),
                    ha="center",va="bottom",fontsize=9,fontweight="bold")
        if idx:
            ax.axvline(a,color="#4b5563",lw=1,alpha=.7)

    holidays={}
    for y in (2025,2026):holidays.update(italian_holidays(y))
    visible=[(pd.Timestamp(day),name) for day,name in holidays.items()
             if start<=pd.Timestamp(day)<=end]
    for idx,(day,name) in enumerate(sorted(visible)):
        ax.axvline(day,color="#c22b36",lw=.9,alpha=.65,
                   label="Italian national holiday" if idx==0 else None)
        ax.annotate(name,(day,0.99),xycoords=("data","axes fraction"),
                    xytext=(2,-3-(idx%2)*11),textcoords="offset points",
                    rotation=90,ha="left",va="top",fontsize=7,color="#a2222a")
    ax.set_xlim(start,end+pd.Timedelta(days=1))
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Within-day standard deviation of actual hourly price (EUR/MWh)")
    ax.set_xlabel("Italian local date")
    ax.set_title("Observed electricity price volatility — July 2025 to October 2026",pad=38)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y",alpha=.2)
    ax.legend(loc="upper left",ncol=2,fontsize=9)
    fig.autofmt_xdate()
    fig.text(.02,.01,
             "SD is calculated across observed hourly prices each local day; not forecast error or financial return volatility. "
             "Local holidays excluded.",fontsize=8,color="#555")
    fig.tight_layout(rect=(0,.035,1,.94))
    dest=OUT/"daily_price_volatility_five_periods.png"
    fig.savefig(dest,dpi=180,bbox_inches="tight")
    plt.close(fig)
    print(f"Saved graph: {dest}")
    print(f"Saved daily data: {OUT / 'daily_price_volatility.csv'}")
    print("\nPeriod mean daily volatility (EUR/MWh):")
    for name,lo,hi in PERIODS:
        subset=daily[daily["date"].between(pd.Timestamp(lo),pd.Timestamp(hi))]
        print(f"{name}: {subset['daily_sd_eur_mwh'].mean():.2f} ({len(subset)} days)")

if __name__=="__main__":
    main()
