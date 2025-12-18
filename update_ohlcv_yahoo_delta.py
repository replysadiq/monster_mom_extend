"""Incremental Yahoo OHLCV updater with overlap and deduplication.

This script appends only NEW daily data to an existing parquet while
re-fetching an overlap window to keep long-horizon indicators intact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf


DEFAULT_OHLCV_PATH = Path("data/ohlcv_yahoo.parquet")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Delta-update Yahoo OHLCV parquet with overlap and dedupe.")
    ap.add_argument("--symbols-csv", type=Path, required=True, help="CSV with a 'symbol' column (Yahoo tickers).")
    ap.add_argument("--ohlcv-out", type=Path, default=DEFAULT_OHLCV_PATH, help="Parquet to update.")
    ap.add_argument("--overlap-days", type=int, default=400, help="Refetch this many days before last_date for indicator stability.")
    return ap.parse_args()


def _normalize_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
            "Dividends": "Dividends",
            "Stock Splits": "Stock Splits",
        }
    )
    expected = ["open", "high", "low", "close", "adj_close", "volume", "Dividends", "Stock Splits"]
    for col in expected:
        if col not in df.columns:
            df[col] = 0.0 if col != "volume" else 0
    out = df[expected].copy()
    out["symbol"] = symbol
    out.reset_index(inplace=True)
    out.rename(columns={"Date": "date"}, inplace=True)
    # Ensure timezone aware to avoid accidental date collisions across DST/timezones.
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    # Cast types to align with existing parquet
    out["volume"] = out["volume"].astype("int64", errors="ignore")
    return out


def fetch_symbols(symbols: List[str], start: datetime, end: Optional[datetime]) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        data = yf.download(sym, start=start, end=end, auto_adjust=False, progress=False, group_by="column")
        if data.empty:
            continue
        frames.append(_normalize_columns(data, sym))
    if not frames:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume", "Dividends", "Stock Splits", "symbol"])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    symbols_df = pd.read_csv(args.symbols_csv)
    if "symbol" not in symbols_df.columns:
        raise RuntimeError(f"symbols CSV must have a 'symbol' column. Found: {symbols_df.columns.tolist()}")
    symbols = symbols_df["symbol"].dropna().astype(str).unique().tolist()

    existing = None
    last_date: Optional[pd.Timestamp] = None
    if args.ohlcv_out.exists():
        existing = pd.read_parquet(args.ohlcv_out)
        last_date = pd.to_datetime(existing["date"]).max()

    if last_date is None or pd.isna(last_date):
        start = datetime.now(timezone.utc) - timedelta(days=args.overlap_days)
    else:
        start = (last_date - timedelta(days=args.overlap_days)).to_pydatetime()
    end = None  # fetch until today

    fresh = fetch_symbols(symbols, start=start, end=end)
    if fresh.empty and existing is not None:
        print("No new data fetched; existing parquet left unchanged.")
        return

    if existing is not None:
        combined = pd.concat([existing, fresh], ignore_index=True)
    else:
        combined = fresh

    # Deduplicate and sort
    combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
    combined.sort_values(["symbol", "date"], inplace=True)
    combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")

    args.ohlcv_out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.ohlcv_out, index=False)
    print(f"Wrote {args.ohlcv_out} rows={len(combined)} (fetched {len(fresh)} new rows)")


if __name__ == "__main__":
    main()
