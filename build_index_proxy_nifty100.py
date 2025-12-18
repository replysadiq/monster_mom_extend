# build_index_proxy_nifty100.py
import pandas as pd
import yfinance as yf

TICKERS = [
 "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","LT.NS","ITC.NS",
 "HINDUNILVR.NS","BHARTIARTL.NS","ASIANPAINT.NS","MARUTI.NS","TITAN.NS","SUNPHARMA.NS","NTPC.NS","POWERGRID.NS","ONGC.NS","TATASTEEL.NS",
 "JSWSTEEL.NS","HINDALCO.NS","ADANIENT.NS","ADANIPORTS.NS","COALINDIA.NS","BAJFINANCE.NS","BAJAJFINSV.NS","HDFCLIFE.NS","SBILIFE.NS","ICICIPRULI.NS",
 "ULTRACEMCO.NS","GRASIM.NS","WIPRO.NS","TECHM.NS","HCLTECH.NS","DIVISLAB.NS","DRREDDY.NS","CIPLA.NS","APOLLOHOSP.NS","BRITANNIA.NS",
 "NESTLEIND.NS","TATACONSUM.NS","EICHERMOT.NS","HEROMOTOCO.NS","BAJAJ-AUTO.NS","M&M.NS","INDUSINDBK.NS","BANKBARODA.NS","PNB.NS","CANBK.NS",
 "IOC.NS","BPCL.NS","GAIL.NS","HDFCAMC.NS","DLF.NS","GODREJCP.NS","DABUR.NS","PIDILITIND.NS","SHREECEM.NS","AMBUJACEM.NS",
 "ACC.NS","LTIM.NS","TATAMOTORS.NS","TVSMOTOR.NS","BEL.NS","HAL.NS","IRCTC.NS","ADANIGREEN.NS","ADANITRANS.NS","INDIGO.NS",
 "BAJAJHLDNG.NS","CHOLAFIN.NS","SBICARD.NS","HAVELLS.NS","SIEMENS.NS","DMART.NS","MARICO.NS","BERGEPAINT.NS","COLPAL.NS","MUTHOOTFIN.NS",
 "SRF.NS","ABB.NS","LUPIN.NS","BIOCON.NS","AUROPHARMA.NS","TORNTPHARM.NS","ICICIGI.NS","PERSISTENT.NS","MPHASIS.NS","COFORGE.NS",
 "TATAPOWER.NS","JINDALSTEL.NS","VEDL.NS","SAIL.NS","ASHOKLEY.NS","UPL.NS","IGL.NS","NAUKRI.NS","SBIN.NS","ZOMATO.NS"
]
TICKERS = sorted(set(TICKERS))  # de-dup

if __name__ == "__main__":
    df = yf.download(TICKERS, period="5y", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        adj = df.xs("Adj Close", level=1, axis=1)
    else:
        adj = df.filter(like="Adj Close")
    proxy = adj.median(axis=1, skipna=True).dropna()
    out = proxy.rename("close").reset_index().rename(columns={"Date": "date", "index": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    out[["date", "close"]].to_csv("data/index_proxy_nifty100_yahoo.csv", index=False)
    print("Wrote data/index_proxy_nifty100_yahoo.csv rows=", len(out))
