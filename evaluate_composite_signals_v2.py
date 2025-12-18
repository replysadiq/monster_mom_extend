from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from price_scale import choose_scale_factor


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate composite signals on weekly data (v2, fixed index excess math).")
    ap.add_argument("--ohlcv", type=Path, default=Path("data/ohlcv_yahoo.parquet"))
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--index", type=Path, default=Path("data/index_nifty500.csv"))
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/composite_eval_yahoo_3y"))
    ap.add_argument("--rolling-window", type=int, default=26)
    ap.add_argument("--nonoverlap-step", type=int, default=4)
    ap.add_argument("--mr-selected", type=Path, default=None, help="JSON list from feature selection (mr_1w)")
    ap.add_argument("--trend-selected", type=Path, default=None, help="JSON list from feature selection (trend_4w)")
    return ap.parse_args()


def _strip_tz(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    return dt


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_index_weekly_close(idx_close: pd.Series, out_dir: Path) -> None:
    df = idx_close.rename("index_close").reset_index()
    if df.columns.tolist() != ["week_date", "index_close"]:
        df.columns = ["week_date", "index_close"]
    df.to_csv(out_dir / "index_weekly_close.csv", index=False)


def load_selected_features(path: Optional[Path]) -> Dict[str, int]:
    """
    JSON format: [{"feature": "ret_1w", "sign": -1}, ...]
    Returns dict: {feature: sign}
    """
    if path is None:
        return {}
    data = json.loads(path.read_text())
    out: Dict[str, int] = {}
    for row in data:
        f = row["feature"]
        s = int(row.get("sign", 1))
        out[f] = 1 if s >= 0 else -1
    return out


def assert_selected_present(feats: pd.DataFrame, selected: Dict[str, int], label: str) -> None:
    missing = [f for f in selected.keys() if f not in feats.columns]
    if missing:
        raise RuntimeError(f"{label} selected features missing from features parquet: {missing}")


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "date" not in df.columns or "symbol" not in df.columns or "close" not in df.columns:
        raise RuntimeError(f"OHLCV parquet missing required columns. Found: {df.columns.tolist()}")
    df["date"] = _strip_tz(df["date"])
    df = df.sort_values(["symbol", "date"])
    return df


def weekly_close_from_ohlcv(df: pd.DataFrame) -> pd.Series:
    # Weekly last close per symbol on W-FRI (select column before resample to avoid FutureWarning)
    x = df[["date", "symbol", "close"]].copy().sort_values(["symbol", "date"])
    wk = (
        x.set_index("date")
        .groupby("symbol")["close"]
        .resample("W-FRI")
        .last()
        .dropna()
    )
    wk.index = wk.index.set_names(["symbol", "week_date"])
    wk = wk.sort_index()
    wk.name = "close"
    return wk


def weekly_index_close_from_csv(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    idx = pd.read_csv(path)
    if "date" not in idx.columns:
        raise RuntimeError(f"Index CSV must have a 'date' column. Found: {idx.columns.tolist()}")
    # user confirmed index close column is 'close'
    if "close" not in idx.columns:
        raise RuntimeError(f"Index CSV must have a 'close' column. Found: {idx.columns.tolist()}")

    idx["date"] = _strip_tz(idx["date"])
    idx = idx.sort_values("date")
    idx = idx[(idx["date"] >= start) & (idx["date"] <= end)]
    # Weekly last close on W-FRI
    idx_w = idx.set_index("date")["close"].astype(float).resample("W-FRI").last()
    # scale normalization similar to stocks (constant factor cancels in returns, but helps sanity reporting)
    med = idx_w.median()
    scale = choose_scale_factor(med)
    idx_w = idx_w / scale
    idx_w.index.name = "week_date"
    idx_w.name = "index_close"
    return idx_w


def proxy_index_from_close(wk_close: pd.Series) -> pd.Series:
    pivot = wk_close.reset_index().pivot(index="week_date", columns="symbol", values="close")
    med = pivot.median(axis=1).rename("index_close")
    med.index.name = "week_date"
    return med


def compute_targets_fixed(
    close: pd.Series, idx_close: pd.Series
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute:
      fwd1, fwd4 (raw stock forwards)
      idx_fwd1, idx_fwd4 (index forwards on index series)
      target1, target4 (excess = stock - index, broadcasted safely)
    """
    if not isinstance(close.index, pd.MultiIndex) or close.index.names != ["symbol", "week_date"]:
        raise RuntimeError("close must be a MultiIndex Series with index names ['symbol','week_date'].")

    # Stock forwards (per symbol)
    fwd1 = close.groupby(level=0).shift(-1) / close - 1.0
    fwd4 = close.groupby(level=0).shift(-4) / close - 1.0
    fwd1.name = "fwd_1w_ret"
    fwd4.name = "fwd_4w_ret"

    # Index forwards (ONLY on index series)
    idx_close = idx_close.sort_index()
    idx_fwd1 = idx_close.shift(-1) / idx_close - 1.0
    idx_fwd4 = idx_close.shift(-4) / idx_close - 1.0
    idx_fwd1.name = "idx_fwd_1w_ret"
    idx_fwd4.name = "idx_fwd_4w_ret"

    # Broadcast index forwards to stock MultiIndex by week_date
    week = close.index.get_level_values("week_date")
    idx1_b = pd.Series(week.map(idx_fwd1).to_numpy(), index=close.index, name="idx_fwd1_b")
    idx4_b = pd.Series(week.map(idx_fwd4).to_numpy(), index=close.index, name="idx_fwd4_b")

    target1 = (fwd1 - idx1_b).rename("target_1w_excess")
    target4 = (fwd4 - idx4_b).rename("target_4w_excess")
    return fwd1, fwd4, target1, target4, idx_fwd1, idx_fwd4


def zscore_weekly(series: pd.Series) -> pd.Series:
    def _z(s: pd.Series) -> pd.Series:
        mu = s.mean()
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd
    return series.groupby(level="week_date", group_keys=False).apply(_z)


def build_composites(
    feats: pd.DataFrame,
    mr_feats: Dict[str, int],
    trend_feats: Dict[str, int],
    min_feat: int = 2,
) -> Tuple[pd.Series, pd.Series]:
    df = feats.copy()

    for col, sign in {**mr_feats, **trend_feats}.items():
        if col not in df.columns:
            continue
        z = zscore_weekly(df[col])
        if sign == -1:
            z = -z
        df[col] = z

    def _compose(cols: List[str]) -> pd.Series:
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.Series(dtype=float)
        zmat = df[present]
        cnt = zmat.notna().sum(axis=1)
        comp = zmat.mean(axis=1)
        comp[cnt < min_feat] = np.nan
        return comp

    return _compose(list(mr_feats.keys())), _compose(list(trend_feats.keys()))


def ic_timeseries(score: pd.Series, target: pd.Series) -> pd.DataFrame:
    records = []
    weeks = sorted(
        set(score.dropna().index.get_level_values("week_date"))
        & set(target.dropna().index.get_level_values("week_date"))
    )
    for wk in weeks:
        s = score.xs(wk, level="week_date", drop_level=False)
        t = target.xs(wk, level="week_date", drop_level=False).rename("target")
        merged = pd.concat([s.rename("score"), t], axis=1).dropna()
        if len(merged) < 20:
            continue
        ic, _ = spearmanr(merged["score"], merged["target"])
        if np.isnan(ic):
            continue
        records.append({"week_date": wk, "ic": float(ic), "n_symbols": int(len(merged))})
    return pd.DataFrame(records)


def rolling_ic(ts: pd.DataFrame, window: int) -> pd.DataFrame:
    if ts.empty:
        return ts
    out = ts.sort_values("week_date").set_index("week_date")
    out["ic_rolling_mean"] = out["ic"].rolling(window, min_periods=1).mean()
    return out.reset_index()


def decile_backtest(score: pd.Series, target: pd.Series, horizon: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records_decile = []
    ts_records = []
    weeks = sorted(
        set(score.dropna().index.get_level_values("week_date"))
        & set(target.dropna().index.get_level_values("week_date"))
    )
    for wk in weeks:
        s = score.xs(wk, level="week_date", drop_level=False)
        t = target.xs(wk, level="week_date", drop_level=False).rename("target")
        merged = pd.concat([s.rename("score"), t], axis=1).dropna()
        if len(merged) < 50:
            continue
        try:
            merged["decile"] = pd.qcut(merged["score"], 10, labels=False, duplicates="drop")
        except ValueError:
            continue

        for d in sorted(merged["decile"].unique()):
            m = merged[merged["decile"] == d]
            records_decile.append({"decile": int(d), "target": float(m["target"].mean())})

        top = float(merged[merged["decile"] == merged["decile"].max()]["target"].mean())
        bot = float(merged[merged["decile"] == merged["decile"].min()]["target"].mean())
        ts_records.append({"week_date": wk, "top_decile_ret": top, "bottom_decile_ret": bot, "spread": top - bot})

    decile_df = pd.DataFrame(records_decile)
    if not decile_df.empty:
        decile_df = (
            decile_df.groupby("decile")["target"].mean().reset_index().rename(columns={"target": "mean_target_return"})
        )

    ts_df = pd.DataFrame(ts_records).sort_values("week_date")
    # compound only for 1w (valid)
    if not ts_df.empty and horizon == "1w":
        ts_df["cum_spread"] = (1 + ts_df["spread"]).cumprod()
        ts_df["top_decile_cum"] = (1 + ts_df["top_decile_ret"]).cumprod()
    return decile_df, ts_df


def nonoverlap_backtest(score: pd.Series, fwd4_raw: pd.Series, target4_excess: pd.Series, step: int = 4) -> pd.DataFrame:
    weeks = sorted(
        set(score.dropna().index.get_level_values("week_date"))
        & set(fwd4_raw.dropna().index.get_level_values("week_date"))
        & set(target4_excess.dropna().index.get_level_values("week_date"))
    )
    records = []
    for i, wk in enumerate(weeks):
        if i % step != 0:
            continue
        s = score.xs(wk, level="week_date", drop_level=False)
        t_raw = fwd4_raw.xs(wk, level="week_date", drop_level=False).rename("t_raw")
        t_exc = target4_excess.xs(wk, level="week_date", drop_level=False).rename("t_excess")
        merged = pd.concat([s.rename("score"), t_raw, t_exc], axis=1).dropna()
        if len(merged) < 50:
            continue
        try:
            merged["decile"] = pd.qcut(merged["score"], 10, labels=False, duplicates="drop")
        except ValueError:
            continue

        top_raw = float(merged[merged["decile"] == merged["decile"].max()]["t_raw"].mean())
        bot_raw = float(merged[merged["decile"] == merged["decile"].min()]["t_raw"].mean())
        top_exc = float(merged[merged["decile"] == merged["decile"].max()]["t_excess"].mean())
        bot_exc = float(merged[merged["decile"] == merged["decile"].min()]["t_excess"].mean())

        records.append(
            {
                "week_date": wk,
                "top_ret_raw": top_raw,
                "bot_ret_raw": bot_raw,
                "spread_raw": top_raw - bot_raw,
                "top_ret_excess": top_exc,
                "bot_ret_excess": bot_exc,
                "spread_excess": top_exc - bot_exc,
            }
        )

    df = pd.DataFrame(records).sort_values("week_date")
    if not df.empty:
        df["equity_top_raw"] = (1 + df["top_ret_raw"]).cumprod()
        df["equity_spread_raw"] = (1 + df["spread_raw"]).cumprod()
        df["equity_top_excess"] = (1 + df["top_ret_excess"]).cumprod()
        df["equity_spread_excess"] = (1 + df["spread_excess"]).cumprod()
    return df


def summary_from_ic(ts: pd.DataFrame) -> Dict[str, float]:
    if ts.empty:
        return {"ic_mean": np.nan, "ic_std": np.nan, "ic_tstat": np.nan, "ic_pos_frac": np.nan, "n_weeks": 0, "avg_n_symbols": np.nan}
    ic_arr = ts["ic"].to_numpy()
    n = len(ic_arr)
    std = float(ic_arr.std(ddof=0))
    mean = float(ic_arr.mean())
    tstat = float(mean / (std / np.sqrt(n))) if std > 0 else np.nan
    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_tstat": tstat,
        "ic_pos_frac": float((ic_arr > 0).mean()),
        "n_weeks": int(n),
        "avg_n_symbols": float(ts["n_symbols"].mean()),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ohlcv = load_ohlcv(args.ohlcv)
    start = pd.to_datetime(args.start) if args.start else ohlcv["date"].min()
    end = pd.to_datetime(args.end) if args.end else ohlcv["date"].max()
    ohlcv = ohlcv[(ohlcv["date"] >= start) & (ohlcv["date"] <= end)]

    wk_close = weekly_close_from_ohlcv(ohlcv)

    feats = pd.read_parquet(args.features)
    if not isinstance(feats.index, pd.MultiIndex):
        feats = feats.set_index(["symbol", "week_date"])
    feats.index = pd.MultiIndex.from_tuples(feats.index, names=["symbol", "week_date"])
    feats = feats.sort_index()

    # forbid leakage columns in feature parquet
    bad_cols = [c for c in feats.columns if c.startswith("target_") or c.startswith("fwd_") or "forward" in c]
    if bad_cols:
        raise RuntimeError(f"Features parquet contains target/forward columns: {bad_cols}")

    idx_close = weekly_index_close_from_csv(args.index, start, end)
    use_proxy = False

    # align cutoff to avoid NaNs due to forward shifts
    max_week = min(wk_close.index.get_level_values("week_date").max(), idx_close.index.max())
    cutoff = max_week - pd.Timedelta(weeks=4)
    wk_close = wk_close[wk_close.index.get_level_values("week_date") <= cutoff]
    feats = feats[feats.index.get_level_values("week_date") <= cutoff]
    idx_close = idx_close[idx_close.index <= cutoff]

    # compute targets
    fwd1, fwd4, target1, target4, idx_fwd1, idx_fwd4 = compute_targets_fixed(wk_close, idx_close)

    # Target sanity reporting (stocks + index)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = {
        "fwd_1w_min": float(fwd1.min()),
        "fwd_1w_max": float(fwd1.max()),
        "fwd_1w_mean": float(fwd1.mean()),
        "fwd_1w_median": float(fwd1.median()),
        "fwd_4w_min": float(fwd4.min()),
        "fwd_4w_max": float(fwd4.max()),
        "fwd_4w_mean": float(fwd4.mean()),
        "fwd_4w_median": float(fwd4.median()),
        "idx_fwd_4w_min": float(idx_fwd4.min()),
        "idx_fwd_4w_max": float(idx_fwd4.max()),
        "idx_fwd_4w_mean": float(idx_fwd4.mean()),
        "idx_fwd_4w_median": float(idx_fwd4.median()),
        "idx_proxy_used": False,
    }
    pd.DataFrame([overall]).to_csv(out_dir / "target_sanity.csv", index=False)

    # hard sanity check on INDEX fwd_4w (freeze: no fallback)
    idx_p5 = float(np.percentile(idx_fwd4.dropna(), 5))
    idx_p95 = float(np.percentile(idx_fwd4.dropna(), 95))
    idx_med = float(idx_fwd4.median())
    if idx_med < -0.20 or idx_med > 0.20 or idx_p5 < -0.60 or idx_p95 > 0.60:
        raise RuntimeError(
            "Index fwd_4w sanity FAILED. Freeze mode forbids fallback.\n"
            f"idx_med={idx_med:.4f}, idx_p5={idx_p5:.4f}, idx_p95={idx_p95:.4f}\n"
            "Fix/replace the index CSV used for the run."
        )

    # persist the weekly index series actually used
    _write_index_weekly_close(idx_close, out_dir)

    # composites (use selected lists if provided, else defaults)
    mr_feats = load_selected_features(args.mr_selected) or {"ret_1w": -1, "ret_2w": -1, "pct_above_20d_sma": -1}
    trend_feats = load_selected_features(args.trend_selected) or {"sma_50d": -1, "sma_200d": -1, "atr_pct_14": 1}
    if args.mr_selected:
        assert_selected_present(feats, mr_feats, "MR(1W)")
    if args.trend_selected:
        assert_selected_present(feats, trend_feats, "Trend(4W)")
    score_mr, score_trend = build_composites(feats, mr_feats, trend_feats, min_feat=2)

    ts_mr = ic_timeseries(score_mr, target1)
    ts_trend = ic_timeseries(score_trend, target4)

    roll_mr = rolling_ic(ts_mr.assign(composite="mr_1w"), args.rolling_window)
    roll_trend = rolling_ic(ts_trend.assign(composite="trend_4w"), args.rolling_window)

    dec_mr, ts_spread_mr = decile_backtest(score_mr, target1, "1w")
    dec_trend, ts_spread_trend = decile_backtest(score_trend, target4, "4w")
    nonoverlap_trend = nonoverlap_backtest(score_trend, fwd4, target4, step=args.nonoverlap_step)

    summ = pd.DataFrame(
        [
            {"composite": "mr_1w", **summary_from_ic(ts_mr)},
            {"composite": "trend_4w", **summary_from_ic(ts_trend)},
        ]
    )

    ts_all = pd.concat([ts_mr.assign(composite="mr_1w"), ts_trend.assign(composite="trend_4w")], ignore_index=True)
    roll_all = pd.concat(
        [roll_mr.assign(composite="mr_1w"), roll_trend.assign(composite="trend_4w")],
        ignore_index=True,
    )

    summ.to_csv(out_dir / "ic_summary.csv", index=False)
    ts_all.to_csv(out_dir / "ic_timeseries.csv", index=False)
    roll_all.to_csv(out_dir / f"ic_rolling_{args.rolling_window}w.csv", index=False)
    dec_mr.to_csv(out_dir / "deciles_mr_1w.csv", index=False)
    dec_trend.to_csv(out_dir / "deciles_trend_4w.csv", index=False)
    ts_spread_mr.to_csv(out_dir / "decile_spread_timeseries_mr_1w.csv", index=False)
    ts_spread_trend.to_csv(out_dir / "decile_spread_timeseries_trend_4w.csv", index=False)
    nonoverlap_trend.to_csv(out_dir / "equity_nonoverlap_trend_4w.csv", index=False)

    manifest = {
        "ohlcv": str(args.ohlcv),
        "features": str(args.features),
        "index": str(args.index),
        "start": str(start.date()),
        "end": str(end.date()),
        "rolling_window": args.rolling_window,
        "nonoverlap_step": args.nonoverlap_step,
        "sha256": {
            "ohlcv": _sha256_file(args.ohlcv) if args.ohlcv.exists() else None,
            "features": _sha256_file(args.features) if args.features.exists() else None,
            "index": _sha256_file(args.index) if args.index.exists() else None,
        },
    }
    with (out_dir / "run_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print("Input SHA256:", manifest["sha256"])

    # Console summary
    for comp, ts in [("mr_1w", ts_mr), ("trend_4w", ts_trend)]:
        row = summ[summ["composite"] == comp].iloc[0]
        print(
            f"{comp}: ic_mean={row['ic_mean']:.4f}, tstat={row['ic_tstat']:.2f}, "
            f"pos_frac={row['ic_pos_frac']:.2f}, n_weeks={int(row['n_weeks'])}"
        )
    if not ts_spread_mr.empty:
        print(f"mr_1w spread mean={ts_spread_mr['spread'].mean():.4f}, final_cum={ts_spread_mr['cum_spread'].iloc[-1]:.4f}")
    if not ts_spread_trend.empty:
        print(f"4W diagnostic (non-compounded) spread mean={ts_spread_trend['spread'].mean():.4f}")
    if not nonoverlap_trend.empty:
        print(
            f"4W non-overlapping backtest equity_top_raw={nonoverlap_trend['equity_top_raw'].iloc[-1]:.4f}, "
            f"equity_spread_raw={nonoverlap_trend['equity_spread_raw'].iloc[-1]:.4f}, "
            f"equity_top_excess={nonoverlap_trend['equity_top_excess'].iloc[-1]:.4f}, "
            f"equity_spread_excess={nonoverlap_trend['equity_spread_excess'].iloc[-1]:.4f}"
        )


if __name__ == "__main__":
    main()
