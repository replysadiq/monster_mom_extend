from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEFAULT_FEATURES = [
    "ret_12m",
    "ret_6m",
    "ret_3m",
    "ret_1m",
    "ret_8w",
    "ret_2w",
    "ret_1w",
    "excess_ret_3m_vs_index",
    "pct_from_52w_high",
    "pct_above_200d_sma",
    "pct_above_50d_sma",
    "pct_above_20d_sma",
    "adx_14",
    "atr_pct_14",
    "macd_hist_12_26_9",
    "rsi_14",
    "hist_vol_20d",
    "turnover_ratio_20d",
    "sma_200d",
    "sma_50d",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate feature ICs for weekly horizons.")
    ap.add_argument("--ohlcv", type=Path, default=Path("data/ohlcv_yahoo.parquet"))
    ap.add_argument("--features", type=Path, default=None, help="Weekly features parquet (optional).")
    ap.add_argument("--index", type=Path, default=Path("data/index_nifty500.csv"))
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/feature_ic_yahoo_3y"))
    ap.add_argument("--horizons", type=str, default="1w,4w")
    ap.add_argument("--feature-list", type=Path, default=None, help="Optional txt file of features to evaluate.")
    return ap.parse_args()


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    return df


def weekly_last_close(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"])
    weekly = (
        df.set_index("date")
        .groupby("symbol")
        .resample("W-FRI")
        .last()
    )
    if "symbol" in weekly.columns:
        weekly = weekly.drop(columns=["symbol"])
    weekly = weekly.reset_index().rename(columns={"date": "week_date", "close": "weekly_close"})
    weekly = weekly.set_index(["symbol", "week_date"]).sort_index()
    return weekly[["weekly_close"]]


def compute_targets(
    weekly_close: pd.Series, index_weekly: pd.Series, horizons: List[str]
) -> Dict[str, pd.Series]:
    targets: Dict[str, pd.Series] = {}
    for h in horizons:
        steps = 1 if h == "1w" else 4
        fwd = weekly_close.groupby(level=0).shift(-steps) / weekly_close - 1
        fwd_idx = index_weekly.shift(-steps) / index_weekly - 1
        # align index forward to symbol index
        fwd_idx_aligned = fwd_idx.reindex(weekly_close.index.get_level_values("week_date")).values
        fwd_idx_series = pd.Series(fwd_idx_aligned, index=weekly_close.index)
        targets[h] = fwd - fwd_idx_series
    return targets


def compute_weekly_ic(
    features: pd.DataFrame,
    target: pd.Series,
    feats: Iterable[str],
    min_symbols: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    records = []
    summaries: Dict[str, Dict[str, float]] = {}
    weeks = sorted(set(features.index.get_level_values("week_date")) & set(target.index.get_level_values("week_date")))
    for feat in feats:
        if feat not in features.columns:
            continue
        ic_list = []
        n_list = []
        for wk in weeks:
            feat_slice = features.xs(wk, level="week_date", drop_level=False)[[feat]]
            tgt_slice = target.xs(wk, level="week_date", drop_level=False).rename("target")
            merged = feat_slice.join(tgt_slice, how="inner").dropna()
            if len(merged) < min_symbols:
                continue
            ic, _ = spearmanr(merged[feat], merged["target"])
            if np.isnan(ic):
                continue
            ic_list.append(ic)
            n_list.append(len(merged))
            records.append({"week_date": wk, "feature": feat, "ic": ic, "n_symbols": len(merged)})
        if not ic_list:
            continue
        ic_arr = np.array(ic_list)
        summaries[feat] = {
            "ic_mean": ic_arr.mean(),
            "ic_std": ic_arr.std(ddof=0),
            "ic_tstat": ic_arr.mean() / (ic_arr.std(ddof=0) / np.sqrt(len(ic_arr))) if ic_arr.std(ddof=0) > 0 else np.nan,
            "ic_pos_frac": float((ic_arr > 0).mean()),
            "n_weeks": len(ic_arr),
            "avg_n_symbols": float(np.mean(n_list)) if n_list else np.nan,
        }
    ts_df = pd.DataFrame(records)
    return ts_df, summaries


def load_features(path: Path) -> pd.DataFrame:
    feats = pd.read_parquet(path)
    if isinstance(feats.index, pd.MultiIndex):
        feats = feats.sort_index()
    else:
        feats = feats.set_index(["symbol", "week_date"]).sort_index()
    bad_cols = [c for c in feats.columns if c.startswith("target_") or c.startswith("fwd_") or "forward" in c]
    if bad_cols:
        raise RuntimeError(f"Features parquet contains target/forward columns: {bad_cols}")
    return feats


def main() -> None:
    args = parse_args()
    horizons = [h.strip() for h in args.horizons.split(",") if h.strip() in ("1w", "4w")]
    if not horizons:
        raise SystemExit("No valid horizons specified.")
    feature_list = (
        [line.strip() for line in args.feature_list.read_text().splitlines() if line.strip()]
        if args.feature_list
        else DEFAULT_FEATURES
    )

    ohlcv = load_ohlcv(args.ohlcv)
    start = pd.to_datetime(args.start) if args.start else ohlcv["date"].min()
    end = pd.to_datetime(args.end) if args.end else ohlcv["date"].max()
    ohlcv = ohlcv[(ohlcv["date"] >= start) & (ohlcv["date"] <= end)]

    weekly = weekly_last_close(ohlcv)
    weekly = weekly.loc[(weekly.index.get_level_values("week_date") >= start) & (weekly.index.get_level_values("week_date") <= end)]

    if args.features:
        feats = load_features(args.features)
    else:
        raise SystemExit("Features parquet is required (use --features).")
    feats = feats.loc[(feats.index.get_level_values("week_date") >= start) & (feats.index.get_level_values("week_date") <= end)]

    # Index weekly close
    if args.index and args.index.exists():
        idx = pd.read_csv(args.index)
        idx["date"] = pd.to_datetime(idx["date"])
        idx = idx[(idx["date"] >= start) & (idx["date"] <= end)]
        idx_w = idx.set_index("date")["close"].resample("W-FRI").last()
        idx_w.index.name = "week_date"
    else:
        # proxy: median close across symbols each week
        pivot = weekly["weekly_close"].reset_index().pivot(index="week_date", columns="symbol", values="weekly_close")
        idx_w = pivot.median(axis=1).rename("index_close")
    # drop last 4 weeks to ensure forward targets exist
    max_wk = idx_w.index.max()
    cutoff = max_wk - pd.Timedelta(weeks=4)
    idx_w = idx_w[idx_w.index <= cutoff]
    weekly = weekly[weekly.index.get_level_values("week_date") <= cutoff]
    feats = feats[feats.index.get_level_values("week_date") <= cutoff]

    targets = compute_targets(weekly["weekly_close"], idx_w, horizons)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "start": str(start.date()),
        "end": str(end.date()),
        "horizons": horizons,
        "features_requested": feature_list,
        "features_present": [f for f in feature_list if f in feats.columns],
        "features_missing": [f for f in feature_list if f not in feats.columns],
    }
    manifest_path = args.out_dir / "run_manifest.json"

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)

    for h in horizons:
        ts_df, summaries = compute_weekly_ic(feats, targets[h], feature_list)
        if ts_df.empty:
            print(f"[WARN] No ICs computed for horizon {h}.")
            continue
        ts_df.sort_values(["feature", "week_date"]).to_csv(args.out_dir / f"ic_timeseries_{h}.csv", index=False)
        summ_df = (
            pd.DataFrame.from_dict(summaries, orient="index")
            .reset_index()
            .rename(columns={"index": "feature"})
            .sort_values("ic_mean", key=lambda s: s.abs(), ascending=False)
        )
        summ_df.to_csv(args.out_dir / f"ic_summary_{h}.csv", index=False)
        top = summ_df.head(10)
        print(f"=== Top features horizon {h} ===")
        print(top[["feature", "ic_mean", "ic_tstat", "ic_pos_frac", "n_weeks"]])


if __name__ == "__main__":
    main()
