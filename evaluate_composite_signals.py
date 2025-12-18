from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate composite momentum signals on weekly data.")
    ap.add_argument("--ohlcv", type=Path, default=Path("data/ohlcv_yahoo.parquet"))
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--index", type=Path, default=Path("data/index_nifty500.csv"))
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/composite_eval_yahoo_3y"))
    ap.add_argument("--rolling-window", type=int, default=26)
    return ap.parse_args()


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    return df


def weekly_close(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values(["symbol", "date"])
    wk = (
        df.set_index("date")
        .groupby("symbol")
        .resample("W-FRI")
        .last()
    )
    if "symbol" in wk.columns:
        wk = wk.drop(columns=["symbol"])
    wk = wk.reset_index().rename(columns={"date": "week_date"})
    wk = wk.set_index(["symbol", "week_date"]).sort_index()
    return wk["close"]


def weekly_index(path: Optional[Path], start: pd.Timestamp, end: pd.Timestamp, proxy_from: Optional[pd.Series]) -> pd.Series:
    if path and path.exists():
        idx = pd.read_csv(path)
        idx["date"] = pd.to_datetime(idx["date"])
        idx = idx[(idx["date"] >= start) & (idx["date"] <= end)]
        idx_w = idx.set_index("date")["close"].resample("W-FRI").last()
        idx_w.index.name = "week_date"
        return idx_w
    if proxy_from is None:
        raise RuntimeError("No index provided and no proxy series available.")
    pivot = proxy_from.reset_index().pivot(index="week_date", columns="symbol", values="close")
    med = pivot.median(axis=1).rename("index_close")
    med.index.name = "week_date"
    return med


def compute_targets(close: pd.Series, idx_close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute forward returns and excess returns."""
    idx_aligned = idx_close.reindex(close.index.get_level_values("week_date"))
    idx_aligned.index = close.index
    fwd1 = close.groupby(level=0).shift(-1) / close - 1
    fwd4 = close.groupby(level=0).shift(-4) / close - 1
    idx_fwd1 = idx_aligned.groupby(level=0).shift(-1) / idx_aligned - 1
    idx_fwd4 = idx_aligned.groupby(level=0).shift(-4) / idx_aligned - 1
    target1 = fwd1 - idx_fwd1
    target4 = fwd4 - idx_fwd4
    return fwd1, fwd4, target1, target4


def zscore_weekly(series: pd.Series) -> pd.Series:
    """Z-score per week_date across symbols."""
    def _z(s: pd.Series) -> pd.Series:
        if len(s) == 0:
            return s
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
    # compute z-scores per feature
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
    score_mr = _compose(list(mr_feats.keys()))
    score_trend = _compose(list(trend_feats.keys()))
    return score_mr, score_trend


def ic_timeseries(score: pd.Series, target: pd.Series) -> pd.DataFrame:
    records = []
    weeks = sorted(set(score.dropna().index.get_level_values("week_date")) & set(target.dropna().index.get_level_values("week_date")))
    for wk in weeks:
        s = score.xs(wk, level="week_date", drop_level=False)
        t = target.xs(wk, level="week_date", drop_level=False).rename("target")
        merged = pd.concat([s.rename("score"), t], axis=1).dropna()
        if len(merged) < 20:
            continue
        ic, _ = spearmanr(merged["score"], merged["target"])
        if np.isnan(ic):
            continue
        records.append({"week_date": wk, "ic": ic, "n_symbols": len(merged)})
    return pd.DataFrame(records)


def rolling_ic(ts: pd.DataFrame, window: int) -> pd.DataFrame:
    ts = ts.sort_values("week_date").set_index("week_date")
    ts["ic_rolling_mean"] = ts["ic"].rolling(window, min_periods=1).mean()
    return ts.reset_index()


def decile_backtest(score: pd.Series, target: pd.Series, horizon: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records_decile = []
    ts_records = []
    weeks = sorted(set(score.dropna().index.get_level_values("week_date")) & set(target.dropna().index.get_level_values("week_date")))
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
            records_decile.append({"decile": int(d), "target": m["target"].mean()})
        top = merged[merged["decile"] == merged["decile"].max()]["target"].mean()
        bot = merged[merged["decile"] == merged["decile"].min()]["target"].mean()
        ts_records.append(
            {
                "week_date": wk,
                "top_decile_ret": top,
                "bottom_decile_ret": bot,
                "spread": top - bot,
            }
        )
    decile_df = pd.DataFrame(records_decile)
    if not decile_df.empty:
        decile_df = decile_df.groupby("decile")["target"].mean().reset_index().rename(columns={"target": "mean_target_return"})
    ts_df = pd.DataFrame(ts_records).sort_values("week_date")
    if not ts_df.empty and horizon == "1w":
        ts_df["cum_spread"] = (1 + ts_df["spread"]).cumprod()
        ts_df["top_decile_cum"] = (1 + ts_df["top_decile_ret"]).cumprod()
    return decile_df, ts_df


def nonoverlap_backtest(score: pd.Series, target_raw: pd.Series, target_excess: pd.Series, step: int = 4) -> pd.DataFrame:
    weeks = sorted(
        set(score.dropna().index.get_level_values("week_date"))
        & set(target_raw.dropna().index.get_level_values("week_date"))
        & set(target_excess.dropna().index.get_level_values("week_date"))
    )
    records = []
    for idx, wk in enumerate(weeks):
        if idx % step != 0:
            continue
        s = score.xs(wk, level="week_date", drop_level=False)
        t_raw = target_raw.xs(wk, level="week_date", drop_level=False).rename("t_raw")
        t_exc = target_excess.xs(wk, level="week_date", drop_level=False).rename("t_excess")
        merged = pd.concat([s.rename("score"), t_raw, t_exc], axis=1).dropna()
        if len(merged) < 50:
            continue
        try:
            merged["decile"] = pd.qcut(merged["score"], 10, labels=False, duplicates="drop")
        except ValueError:
            continue
        top_raw = merged[merged["decile"] == merged["decile"].max()]["t_raw"].mean()
        bot_raw = merged[merged["decile"] == merged["decile"].min()]["t_raw"].mean()
        top_exc = merged[merged["decile"] == merged["decile"].max()]["t_excess"].mean()
        bot_exc = merged[merged["decile"] == merged["decile"].min()]["t_excess"].mean()
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
    ic_arr = ts["ic"].to_numpy()
    n = len(ic_arr)
    return {
        "ic_mean": float(ic_arr.mean()) if n else np.nan,
        "ic_std": float(ic_arr.std(ddof=0)) if n else np.nan,
        "ic_tstat": float(ic_arr.mean() / (ic_arr.std(ddof=0) / np.sqrt(n))) if n and ic_arr.std(ddof=0) > 0 else np.nan,
        "ic_pos_frac": float((ic_arr > 0).mean()) if n else np.nan,
        "n_weeks": n,
        "avg_n_symbols": float(ts["n_symbols"].mean()) if not ts.empty else np.nan,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ohlcv = load_ohlcv(args.ohlcv)
    start = pd.to_datetime(args.start) if args.start else ohlcv["date"].min()
    end = pd.to_datetime(args.end) if args.end else ohlcv["date"].max()
    ohlcv = ohlcv[(ohlcv["date"] >= start) & (ohlcv["date"] <= end)]

    wk_close = weekly_close(ohlcv)

    feats = pd.read_parquet(args.features)
    if not isinstance(feats.index, pd.MultiIndex):
        feats = feats.set_index(["symbol", "week_date"])
    bad_cols = [c for c in feats.columns if c.startswith("target_") or c.startswith("fwd_") or "forward" in c]
    if bad_cols:
        raise RuntimeError(f"Features parquet contains target/forward columns: {bad_cols}")
    feats.index = pd.MultiIndex.from_tuples(feats.index, names=["symbol", "week_date"])
    feats = feats.sort_index()

    idx_close = None
    if args.index and args.index.exists():
        idx_close = weekly_index(args.index, start, end, None)
    else:
        idx_close = weekly_index(None, start, end, wk_close)

    # align to common weeks and drop last 4 weeks for targets
    max_week = min(wk_close.index.get_level_values("week_date").max(), idx_close.index.max())
    cutoff = max_week - pd.Timedelta(weeks=4)
    wk_close = wk_close[wk_close.index.get_level_values("week_date") <= cutoff]
    feats = feats[feats.index.get_level_values("week_date") <= cutoff]
    idx_close = idx_close[idx_close.index <= cutoff]

    fwd1, fwd4, target1, target4 = compute_targets(wk_close, idx_close)
    fwd1.name = "fwd_1w_ret"
    fwd4.name = "fwd_4w_ret"

    # Target sanity checks
    overall = {
        "fwd_1w_min": float(fwd1.min()),
        "fwd_1w_max": float(fwd1.max()),
        "fwd_1w_mean": float(fwd1.mean()),
        "fwd_1w_median": float(fwd1.median()),
        "fwd_4w_min": float(fwd4.min()),
        "fwd_4w_max": float(fwd4.max()),
        "fwd_4w_mean": float(fwd4.mean()),
        "fwd_4w_median": float(fwd4.median()),
    }
    wstats = fwd4.reset_index().groupby("week_date")["fwd_4w_ret"].agg(
        median="median",
        pct05=lambda x: np.percentile(x, 5),
        pct95=lambda x: np.percentile(x, 95),
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([overall]).to_csv(out_dir / "target_sanity.csv", index=False)
    wstats.to_csv(out_dir / "target_sanity_per_week.csv")
    p5 = float(np.percentile(fwd4.dropna(), 5))
    p95 = float(np.percentile(fwd4.dropna(), 95))
    med = float(fwd4.median())
    if not (-0.20 <= med <= 0.20) or p5 <= -0.60 or p95 >= 0.60:
        close_t4 = wk_close.groupby(level=0).shift(-4).rename("close_t4")
        extremes = (
            pd.concat([fwd4.rename("fwd_4w_ret"), wk_close.rename("close_t"), close_t4], axis=1)
            .reset_index()
            .dropna()
            .sort_values("fwd_4w_ret")
        )
        print("Target sanity failed. Extremes:")
        print(extremes.head(20))
        raise RuntimeError("fwd_4w_ret sanity check failed.")

    mr_feats = {"ret_1w": -1, "ret_2w": -1, "pct_above_20d_sma": -1}
    trend_feats = {"sma_50d": -1, "sma_200d": -1, "atr_pct_14": 1}
    score_mr, score_trend = build_composites(feats, mr_feats, trend_feats, min_feat=2)

    ts_mr = ic_timeseries(score_mr, target1)
    ts_trend = ic_timeseries(score_trend, target4)

    roll_mr = rolling_ic(ts_mr.assign(composite="mr_1w"), args.rolling_window)
    roll_trend = rolling_ic(ts_trend.assign(composite="trend_4w"), args.rolling_window)

    dec_mr, ts_spread_mr = decile_backtest(score_mr, target1, "1w")
    dec_trend, ts_spread_trend = decile_backtest(score_trend, target4, "4w")
    nonoverlap_trend = nonoverlap_backtest(score_trend, fwd4, target4, step=4)

    summ = pd.DataFrame(
        [
            {"composite": "mr_1w", **summary_from_ic(ts_mr)},
            {"composite": "trend_4w", **summary_from_ic(ts_trend)},
        ]
    )

    ts_all = pd.concat(
        [
            ts_mr.assign(composite="mr_1w"),
            ts_trend.assign(composite="trend_4w"),
        ],
        ignore_index=True,
    )
    roll_all = pd.concat([roll_mr.assign(composite="mr_1w"), roll_trend.assign(composite="trend_4w")], ignore_index=True)

    summ.to_csv(args.out_dir / "ic_summary.csv", index=False)
    ts_all.to_csv(args.out_dir / "ic_timeseries.csv", index=False)
    roll_all.to_csv(args.out_dir / f"ic_rolling_{args.rolling_window}w.csv", index=False)
    dec_mr.to_csv(args.out_dir / "deciles_mr_1w.csv", index=False)
    dec_trend.to_csv(args.out_dir / "deciles_trend_4w.csv", index=False)
    ts_spread_mr.to_csv(args.out_dir / "decile_spread_timeseries_mr_1w.csv", index=False)
    ts_spread_trend.to_csv(args.out_dir / "decile_spread_timeseries_trend_4w.csv", index=False)
    nonoverlap_trend.to_csv(args.out_dir / "equity_nonoverlap_trend_4w.csv", index=False)

    manifest = {
        "ohlcv": str(args.ohlcv),
        "features": str(args.features),
        "index": str(args.index) if args.index else None,
        "start": str(start.date()),
        "end": str(end.date()),
        "rolling_window": args.rolling_window,
    }
    with (args.out_dir / "run_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    for comp, ts in [("mr_1w", ts_mr), ("trend_4w", ts_trend)]:
        summ_row = summ[summ["composite"] == comp].iloc[0]
        print(
            f"{comp}: ic_mean={summ_row['ic_mean']:.4f}, tstat={summ_row['ic_tstat']:.2f}, "
            f"pos_frac={summ_row['ic_pos_frac']:.2f}, n_weeks={int(summ_row['n_weeks'])}"
        )
    if not ts_spread_mr.empty:
        print(
            f"mr_1w spread mean={ts_spread_mr['spread'].mean():.4f}, final_cum={ts_spread_mr['cum_spread'].iloc[-1]:.4f}"
        )
    if not ts_spread_trend.empty:
        print(
            f"4W diagnostic (non-compounded) spread mean={ts_spread_trend['spread'].mean():.4f}"
        )
    if not nonoverlap_trend.empty:
        print(
            f"4W non-overlapping backtest equity_top_raw={nonoverlap_trend['equity_top_raw'].iloc[-1]:.4f}, "
            f"equity_spread_raw={nonoverlap_trend['equity_spread_raw'].iloc[-1]:.4f}, "
            f"equity_top_excess={nonoverlap_trend['equity_top_excess'].iloc[-1]:.4f}, "
            f"equity_spread_excess={nonoverlap_trend['equity_spread_excess'].iloc[-1]:.4f}"
        )


if __name__ == "__main__":
    main()
