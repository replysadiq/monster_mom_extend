from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Weekly Top-N backtest using frozen composite ranks (W-FRI close, equal weight).")
    ap.add_argument("--features", type=Path, default=Path("data/weekly_features_yahoo.parquet"))
    ap.add_argument("--composites", type=Path, default=Path("data/frozen/composites_selected_v1.parquet"))
    ap.add_argument("--signal", type=str, default="rank_trend_4w", choices=["rank_trend_4w", "rank_mr_1w", "score_trend_4w", "score_mr_1w"])
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--fee_bps_per_turnover", type=float, default=0.0, help="Total cost in bps per 1.0 turnover (round-trip). Default 0.")
    ap.add_argument("--out-dir", type=Path, default=Path("results/backtest_selected_v1"))
    return ap.parse_args()


def _as_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        if {"symbol", "week_date"}.issubset(df.columns):
            df = df.set_index(["symbol", "week_date"])
        else:
            raise RuntimeError("Expected MultiIndex or columns ['symbol','week_date'].")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["symbol", "week_date"])
    return df.sort_index()


def _weekly_forward_returns_from_close(wk_close: pd.Series) -> pd.Series:
    fwd1 = wk_close.groupby(level=0).shift(-1) / wk_close - 1.0
    fwd1.name = "fwd_1w_ret"
    return fwd1


def _pick_topn(signal: pd.Series, week: pd.Timestamp, topn: int) -> List[str]:
    s = signal.xs(week, level="week_date", drop_level=False).droplevel("week_date")
    s = s.dropna().sort_values(ascending=False)
    return s.head(topn).index.astype(str).tolist()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feats = _as_multiindex(pd.read_parquet(args.features))
    comps = _as_multiindex(pd.read_parquet(args.composites))

    if "close" not in feats.columns:
        raise RuntimeError("features parquet must contain weekly 'close'.")

    if args.signal not in comps.columns:
        raise RuntimeError(f"Signal '{args.signal}' not found in composites. Found: {comps.columns.tolist()}")

    wk_close = feats["close"].astype(float).rename("close")
    fwd1 = _weekly_forward_returns_from_close(wk_close)

    weeks = sorted(set(comps.index.get_level_values("week_date")) & set(fwd1.dropna().index.get_level_values("week_date")))
    if not weeks:
        raise RuntimeError("No overlapping weeks between composites and forward returns.")

    signal = comps[args.signal].astype(float)

    equity = 1.0
    prev_weights = {}
    equity_rows = []
    trade_rows = []

    fee_rate = (args.fee_bps_per_turnover / 10000.0) if args.fee_bps_per_turnover else 0.0

    for wk in weeks:
        picks = _pick_topn(signal, wk, args.topn)
        if len(picks) < max(3, args.topn // 2):
            equity_rows.append({"week_date": wk, "equity": equity, "port_ret": 0.0, "turnover": 0.0, "n": len(picks)})
            continue

        w = 1.0 / len(picks)
        new_weights = {sym: w for sym in picks}

        all_syms = set(prev_weights.keys()) | set(new_weights.keys())
        turnover = 0.0
        for sym in all_syms:
            turnover += abs(new_weights.get(sym, 0.0) - prev_weights.get(sym, 0.0))
        turnover *= 0.5

        idx = pd.MultiIndex.from_product([picks, [wk]], names=["symbol", "week_date"])
        rets = fwd1.reindex(idx).droplevel("week_date")
        port_ret_gross = float(np.nanmean(rets.to_numpy()))
        port_ret_net = port_ret_gross - (fee_rate * turnover)

        equity *= (1.0 + port_ret_net)

        trade_rows.append({
            "week_date": wk,
            "n": len(picks),
            "symbols": "|".join(picks),
            "turnover": turnover,
            "port_ret_gross": port_ret_gross,
            "port_ret_net": port_ret_net,
            "equity": equity,
        })
        equity_rows.append({"week_date": wk, "equity": equity, "port_ret": port_ret_net, "turnover": turnover, "n": len(picks)})
        prev_weights = new_weights

    equity_df = pd.DataFrame(equity_rows).sort_values("week_date")
    trades_df = pd.DataFrame(trade_rows).sort_values("week_date")

    if len(equity_df) >= 2:
        rets = equity_df["port_ret"].to_numpy()
        mean = float(np.nanmean(rets))
        std = float(np.nanstd(rets, ddof=0))
        sharpe = float((mean / std) * np.sqrt(52)) if std > 0 else np.nan

        eq = equity_df["equity"].to_numpy()
        peak = np.maximum.accumulate(eq)
        dd = (eq / peak) - 1.0
        max_dd = float(np.min(dd))

        summary = {
            "signal": args.signal,
            "topn": args.topn,
            "fee_bps_per_turnover": args.fee_bps_per_turnover,
            "weeks": int(len(equity_df)),
            "final_equity": float(equity_df["equity"].iloc[-1]),
            "mean_weekly_ret": mean,
            "vol_weekly": std,
            "sharpe_annualized": sharpe,
            "max_drawdown": max_dd,
        }
    else:
        summary = {"signal": args.signal, "topn": args.topn, "weeks": int(len(equity_df))}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    equity_path = args.out_dir / "equity.csv"
    trades_path = args.out_dir / "trades.csv"
    summary_path = args.out_dir / "summary.json"

    equity_df.to_csv(equity_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote: {equity_path}")
    print(f"Wrote: {trades_path}")
    print(f"Wrote: {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
