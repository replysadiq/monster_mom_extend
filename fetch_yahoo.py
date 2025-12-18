from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf


def load_tickers(raw_dir: Path) -> List[str]:
    return sorted([p.stem for p in raw_dir.glob("*.parquet")])


def download_batch(tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    records = []
    failed = []
    for t in tickers:
        ysym = f"{t}.NS"
        try:
            df = yf.Ticker(ysym).history(start=start, end=end, auto_adjust=False)
        except Exception as e:
            failed.append((t, str(e)))
            continue
        if df.empty:
            failed.append((t, "empty"))
            continue
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        df["symbol"] = t
        df = df.reset_index().rename(columns={"Date": "date"})
        records.append(df)
    if failed:
        print(f"Skipped {len(failed)} tickers: {failed[:10]}")
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("data/ohlcv_raw"))
    ap.add_argument("--out", type=Path, default=Path("data/ohlcv_yahoo.parquet"))
    ap.add_argument("--years", type=int, default=3)
    args = ap.parse_args()

    tickers = load_tickers(args.raw_dir)
    start = datetime.today() - timedelta(days=365 * args.years)
    end = datetime.today()

    df = download_batch(tickers, start, end)
    if df.empty:
        raise SystemExit("No data downloaded.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"])
    df = df.sort_values(["symbol", "date"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Saved {len(df)} rows for {df['symbol'].nunique()} symbols to {args.out}")


if __name__ == "__main__":
    main()
