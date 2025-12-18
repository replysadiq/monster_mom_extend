import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

def _fix_scale_breaks(close: pd.Series, max_jump: float = 20.0) -> pd.Series:
    """
    Fix multiplicative unit/scale regime breaks in a level series.
    If close[t]/close[t-1] is absurd (>max_jump or <1/max_jump),
    assume units changed and apply a cumulative divisor from that point onward.
    """
    s = close.astype(float).copy()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 3:
        return close.astype(float)

    r = s / s.shift(1)
    breaks = (r > max_jump) | (r < (1.0 / max_jump))
    if not breaks.any():
        return close.astype(float)

    div = pd.Series(1.0, index=s.index)
    cum_div = 1.0
    for t in s.index[1:]:
        if bool(breaks.loc[t]):
            cum_div *= float(r.loc[t])
        div.loc[t] = cum_div

    fixed = s / div
    out = close.astype(float).copy()
    out.loc[fixed.index] = fixed
    return out


def _rescale_to_band(close: pd.Series, lo: float = 50.0, hi: float = 500000.0) -> Tuple[pd.Series, float]:
    """
    Apply a single constant divisor to bring median into [lo, hi].
    Uses powers of 10 up to 1e12 for safety.
    """
    med = close.median()
    if not np.isfinite(med) or med <= 0:
        return close, 1.0
    factors = [10.0**k for k in range(0, 13)]  # 1,10,...,1e12
    factor = 1.0
    for f in factors:
        m = med / f
        if lo <= m <= hi:
            factor = f
            break
    return close / factor, factor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("data/index_nifty500.csv"))
    ap.add_argument("--out", dest="out", type=Path, default=Path("data/index_nifty500_clean.csv"))
    ap.add_argument("--report", dest="report", type=Path, default=Path("data/index_nifty500_clean_report.csv"))
    ap.add_argument("--max-jump", type=float, default=20.0)
    ap.add_argument("--proxy-ohlcv", type=Path, default=None, help="If provided, build index from OHLCV median close per day.")
    args = ap.parse_args()

    if args.proxy_ohlcv:
        ohlcv = pd.read_parquet(args.proxy_ohlcv)
        ohlcv["date"] = pd.to_datetime(ohlcv["date"], errors="coerce")
        ohlcv = ohlcv.dropna(subset=["date", "close"])
        daily = ohlcv.groupby("date")["close"].median().rename("close").reset_index()
        daily["date"] = daily["date"].dt.date.astype(str)
        daily.to_csv(args.out, index=False)
        qc = {
            "rows_in": int(ohlcv.shape[0]),
            "rows_out": int(daily.shape[0]),
            "date_min": str(pd.to_datetime(daily["date"]).min().date()),
            "date_max": str(pd.to_datetime(daily["date"]).max().date()),
            "close_median": float(daily["close"].median()),
            "rescale_factor": 1.0,
            "proxy": True,
        }
        pd.DataFrame([qc]).to_csv(args.report, index=False)
        print(f"Wrote proxy index to {args.out}")
        print(qc)
        return

    df = pd.read_csv(args.inp)
    if "date" not in df.columns or "close" not in df.columns:
        raise RuntimeError(f"Expected columns date,close. Found: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # 1) drop invalid
    df = df.dropna(subset=["date", "close"])

    # 2) sort + de-dup (keep last)
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")

    # 3) drop weekends (Sat=5, Sun=6)
    df = df[df["date"].dt.dayofweek < 5]

    # 4) drop non-positive
    df = df[df["close"] > 0].copy()

    # 5) scale-break fix
    df["close_fixed"] = _fix_scale_breaks(df["close"], max_jump=args.max_jump)
    df["close_fixed"], rescale_factor = _rescale_to_band(df["close_fixed"])

    # QC metrics (post-fix)
    df["ret_1d"] = df["close_fixed"].pct_change()
    qc = {
        "rows_in": int(pd.read_csv(args.inp).shape[0]),
        "rows_out": int(df.shape[0]),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "close_median": float(df["close_fixed"].median()),
        "ret_1d_p01": float(np.nanpercentile(df["ret_1d"], 1)),
        "ret_1d_p99": float(np.nanpercentile(df["ret_1d"], 99)),
        "ret_1d_min": float(np.nanmin(df["ret_1d"])),
        "ret_1d_max": float(np.nanmax(df["ret_1d"])),
        "suspect_days_abs_ret_gt_20pct": int((df["ret_1d"].abs() > 0.20).sum()),
        "rescale_factor": float(rescale_factor),
    }
    pd.DataFrame([qc]).to_csv(args.report, index=False)

    # final output format required by your pipeline
    out = df[["date", "close_fixed"]].rename(columns={"close_fixed": "close"})
    out["date"] = out["date"].dt.date.astype(str)  # YYYY-MM-DD
    out.to_csv(args.out, index=False)

    print(f"Wrote: {args.out}")
    print(f"Wrote: {args.report}")
    print(qc)


if __name__ == "__main__":
    main()
