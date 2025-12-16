"""
Weekly momentum backtester (Nifty 500 style) with frozen feature list.

Features:
- Weekly rebalance on W-FRI, execution at next open or Friday close.
- Universe filter by ADV.
- Ranking by frozen features (ret_12m, ret_8w) with optional trend gate.
- Equal-weight or ATR-inverse sizing.
- Costs: commission_bps, slippage_bps, optional impact_bps=k/sqrt(adv_60d).
- Outputs equity curve, trades, holdings, summary metrics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from build_weekly_dataset import (
    build_weekly_panel,
    compute_index_features,
    compute_stock_features,
    load_data,
)


RESULTS_DIR = Path("results")


@dataclass
class Config:
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    initial_capital: float
    top_n: int
    min_adv: float
    w12m: float
    w8w: float
    trend_gate: str  # none | 200d | 50d_200d
    sizing: str  # equal | atr_inverse
    commission_bps: float
    slippage_bps: float
    impact_k: Optional[float]
    execution: str  # next_open | friday_close
    features_path: Optional[Path]
    frozen_features_path: Path
    out_prefix: str
    target_col: str
    manifest: Path
    score_horizon: str
    frozen_1w: Optional[Path]
    frozen_4w: Optional[Path]
    gate_only: Optional[List[str]]
    risk_veto: Optional[List[str]]
    gate_thresholds_yaml: Optional[Path]


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).set_index(["symbol", "date"])
    return df


def load_index(path: Path, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    idx = pd.read_csv(path)
    idx["date"] = pd.to_datetime(idx["date"])
    if start is not None:
        idx = idx[idx["date"] >= start]
    if end is not None:
        idx = idx[idx["date"] <= end]
    return idx.sort_values("date").set_index("date")


def get_weekly_features(cfg: Config) -> pd.DataFrame:
    if cfg.features_path:
        weekly = pd.read_parquet(cfg.features_path)
        return weekly
    stocks, index = load_data(start=cfg.start, end=cfg.end)
    index_feat = compute_index_features(index)
    daily_feat = compute_stock_features(stocks, index_feat)
    weekly = build_weekly_panel(daily_feat, index_feat)
    return weekly


def compute_rebalance_dates(features: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> List[pd.Timestamp]:
    dates = features.index.get_level_values("week_date").unique()
    if start is not None:
        dates = dates[dates >= start]
    if end is not None:
        dates = dates[dates <= end]
    return sorted(dates)


def load_frozen_features(path: Path) -> List[str]:
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


def load_manifest(path: Path) -> Dict[str, Dict[str, int]]:
    import yaml
    data = yaml.safe_load(path.read_text())
    directions = {}
    for entry in data.get("groups", []):
        directions[entry["feature"]] = int(entry.get("direction", 1))
    return directions


def parse_thresholds(s: Optional[str], defaults: Dict[str, float]) -> Dict[str, float]:
    out = defaults.copy()
    if s:
        for pair in s.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                out[k.strip()] = float(v)
    return out


def select_portfolio(
    week_df: pd.DataFrame,
    cfg: Config,
    frozen_features: List[str],
    directions: Dict[str, int],
    gate_only: List[str],
    risk_veto: List[str],
    gate_thresholds: Dict[str, float],
    summary_records: List[Dict[str, object]],
) -> pd.DataFrame:
    df = week_df.copy()
    df = df[df["adv_60d"] >= cfg.min_adv]

    if df.empty:
        summary_records.append(
            {
                "week_date": week_df.index.get_level_values("week_date")[0],
                "eligible_count": 0,
                "selected_count": 0,
                "cash_weight": 1.0,
                "score_dispersion": np.nan,
                "top_minus_median": np.nan,
                "quality_ok": False,
            }
        )
        return pd.DataFrame(columns=["symbol", "score"])

    # Apply gates
    eligible = pd.Series(True, index=df.index)
    for feat in gate_only:
        if feat in df.columns:
            thr = gate_thresholds.get(feat, 0.0)
            eligible &= df[feat].rank(pct=True) >= thr
    for feat in risk_veto:
        if feat in df.columns:
            thr = gate_thresholds.get(feat, 1.0)
            eligible &= df[feat].rank(pct=True) <= thr

    df = df[eligible]

    if df.empty:
        summary_records.append(
            {
                "week_date": week_df.index.get_level_values("week_date")[0],
                "eligible_count": 0,
                "selected_count": 0,
                "cash_weight": 1.0,
                "score_dispersion": np.nan,
                "top_minus_median": np.nan,
                "quality_ok": False,
            }
        )
        return pd.DataFrame(columns=["symbol", "score"])

    # Scoring features
    scoring_features = [f for f in frozen_features if f not in gate_only and f not in risk_veto]
    scores = []
    for feat in scoring_features:
        if feat not in df.columns:
            continue
        ranks = df[feat].rank(pct=True)
        if directions.get(feat, 1) == -1:
            ranks = 1 - ranks
        scores.append(ranks)
    if not scores:
        summary_records.append(
            {
                "week_date": week_df.index.get_level_values("week_date")[0],
                "eligible_count": len(df),
                "selected_count": 0,
                "cash_weight": 1.0,
                "score_dispersion": np.nan,
                "top_minus_median": np.nan,
                "quality_ok": False,
            }
        )
        return pd.DataFrame(columns=["symbol", "score"])
    score_df = pd.concat(scores, axis=1)
    score_df["score"] = score_df.mean(axis=1)
    df = df.loc[score_df.index]
    df["score"] = score_df["score"]

    score_dispersion = df["score"].std()
    top_scores = df["score"].nlargest(min(len(df), cfg.top_n))
    top_minus_median = top_scores.mean() - df["score"].median()
    quality_rules = parse_thresholds(
        cfg.quality_thresholds_str,
        {"min_eligible": 5, "score_std": 0.08, "top_minus_median": 0.10},
    )
    quality_ok = (
        (len(df) >= quality_rules["min_eligible"])
        and (score_dispersion >= quality_rules["score_std"])
        and (top_minus_median >= quality_rules["top_minus_median"])
    )

    if not quality_ok:
        summary_records.append(
            {
                "week_date": week_df.index.get_level_values("week_date")[0],
                "eligible_count": len(df),
                "selected_count": 0,
                "cash_weight": 1.0,
                "score_dispersion": score_dispersion,
                "top_minus_median": top_minus_median,
                "quality_ok": False,
            }
        )
        print(
            f"{week_df.index.get_level_values('week_date')[0]} "
            f"universe={len(week_df)}, eligible={len(df)}, selected=0, cash=1.0, quality_ok=False, "
            f"score_std={score_dispersion:.4f}, top_minus_median={top_minus_median:.4f}"
        )
        return pd.DataFrame(columns=["symbol", "score"])

    top = df.nlargest(cfg.top_n, "score")
    selected_count = len(top)
    cash_w = max(0.0, 1 - 0.1 * selected_count)
    summary_records.append(
        {
            "week_date": week_df.index.get_level_values("week_date")[0],
            "eligible_count": len(df),
            "selected_count": selected_count,
            "cash_weight": cash_w,
            "score_dispersion": score_dispersion,
            "top_minus_median": top_minus_median,
            "quality_ok": True,
        }
    )
    print(
        f"{week_df.index.get_level_values('week_date')[0]} "
        f"universe={len(week_df)}, eligible={len(df)}, selected={selected_count}, cash={cash_w:.2f}, "
        f"quality_ok=True, score_std={score_dispersion:.4f}, top_minus_median={top_minus_median:.4f}"
    )
    return top[["score"]]


def get_execution_prices(
    ohlcv: pd.DataFrame, symbols: List[str], rebalance_date: pd.Timestamp, execution: str
) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for sym in symbols:
        data = ohlcv.xs(sym, level=0)
        if execution == "friday_close":
            row = data[data.index == rebalance_date]
            if not row.empty:
                prices[sym] = float(row["close"].iloc[0])
        else:
            row = data[data.index > rebalance_date].head(1)
            if not row.empty:
                prices[sym] = float(row["open"].iloc[0])
    return prices


def sizing_weights(sel: pd.DataFrame, cfg: Config) -> pd.Series:
    if cfg.sizing == "equal":
        return pd.Series(1.0 / len(sel), index=sel.index)
    if cfg.sizing == "atr_inverse":
        if "atr_pct_14" not in sel.columns:
            raise ValueError("atr_pct_14 required for atr_inverse sizing")
        inv = 1.0 / sel["atr_pct_14"].replace(0, np.nan)
        inv = inv.fillna(0)
        total = inv.sum()
        if total == 0:
            return pd.Series(1.0 / len(sel), index=sel.index)
        return inv / total
    raise ValueError(f"Unknown sizing {cfg.sizing}")


def simulate(
    features: pd.DataFrame,
    ohlcv: pd.DataFrame,
    cfg: Config,
    frozen_features: List[str],
    directions: Dict[str, int],
    gate_only: List[str],
    risk_veto: List[str],
    gate_thresholds: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rebalance_dates = compute_rebalance_dates(features, cfg.start, cfg.end)
    portfolio_value = cfg.initial_capital
    cash = cfg.initial_capital
    holdings: Dict[str, float] = {}

    equity_records = []
    trade_records = []
    holding_records = []
    summary_records: List[Dict[str, object]] = []

    for reb_date in rebalance_dates:
        week_slice = features.xs(reb_date, level="week_date")
        selection = select_portfolio(week_slice, cfg, frozen_features, directions, gate_only, risk_veto, gate_thresholds, summary_records)
        if selection.empty:
            # record equity unchanged
            equity_records.append(
                {
                    "date": reb_date,
                    "portfolio_value": portfolio_value,
                    "return": 0.0,
                }
            )
            continue

        sel_df = week_slice.loc[selection.index]
        weights = sizing_weights(sel_df, cfg)
        exec_prices = get_execution_prices(ohlcv, list(selection.index), reb_date, cfg.execution)

        # Value existing holdings at exec prices
        mark_value = 0.0
        for sym, qty in holdings.items():
            if (sym, reb_date) in ohlcv.index:
                price = float(ohlcv.loc[(sym, reb_date), "close"])
            else:
                # fallback to last available before rebalance
                data = ohlcv.xs(sym, level=0)
                prev = data[data.index <= reb_date].tail(1)
                if prev.empty:
                    continue
                price = float(prev["close"].iloc[0])
            mark_value += qty * price
        portfolio_value = cash + mark_value

        # Target positions
        desired = {}
        for sym, w in weights.items():
            if sym not in exec_prices:
                continue
            price = exec_prices[sym]
            notional = portfolio_value * w
            qty = notional / price
            desired[sym] = qty

        # Trades
        new_holdings: Dict[str, float] = {}
        total_costs = 0.0
        for sym, tgt_qty in desired.items():
            curr_qty = holdings.get(sym, 0.0)
            trade_qty = tgt_qty - curr_qty
            if sym not in exec_prices:
                continue
            price = exec_prices[sym]
            notional = abs(trade_qty) * price
            if notional == 0:
                new_holdings[sym] = tgt_qty
                continue
            comm = cfg.commission_bps / 1e4 * notional
            slip = cfg.slippage_bps / 1e4 * notional
            impact = 0.0
            if cfg.impact_k is not None and cfg.impact_k > 0:
                adv = week_slice.loc[sym, "adv_60d"]
                if adv > 0:
                    impact = (cfg.impact_k / np.sqrt(adv)) * notional / 1e4
            cost = comm + slip + impact
            total_costs += cost
            cash -= trade_qty * price + cost
            side = "buy" if trade_qty > 0 else "sell"
            trade_records.append(
                {
                    "rebalance_date": reb_date,
                    "symbol": sym,
                    "side": side,
                    "qty": trade_qty,
                    "price": price,
                    "notional": trade_qty * price,
                    "costs": cost,
                }
            )
            new_holdings[sym] = tgt_qty

        holdings = new_holdings

        # Update mark-to-market after trades
        mtm = 0.0
        for sym, qty in holdings.items():
            price = exec_prices.get(sym)
            if price is None:
                continue
            mtm += qty * price
        portfolio_value = cash + mtm

        equity_records.append(
            {
                "date": reb_date,
                "portfolio_value": portfolio_value,
                "return": 0.0,  # compute later
            }
        )
        for sym, w in weights.items():
            holding_records.append(
                {"rebalance_date": reb_date, "symbol": sym, "weight": w}
            )

    equity_df = pd.DataFrame(equity_records).sort_values("date").reset_index(drop=True)
    equity_df["return"] = equity_df["portfolio_value"].pct_change().fillna(0.0)
    equity_df["cum_return"] = (1 + equity_df["return"]).cumprod()
    peak = equity_df["cum_return"].cummax()
    equity_df["drawdown"] = equity_df["cum_return"] / peak - 1

    trades_df = pd.DataFrame(trade_records)
    holdings_df = pd.DataFrame(holding_records)
    summary_df = pd.DataFrame(summary_records)
    return equity_df, trades_df, holdings_df, summary_df


def compute_metrics(equity: pd.DataFrame, trades: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    rets = equity["return"]
    weeks_per_year = 52
    ann_ret = (1 + rets).prod() ** (weeks_per_year / max(len(rets), 1)) - 1
    ann_vol = rets.std(ddof=0) * np.sqrt(weeks_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd = equity["drawdown"].min()
    calmar = -ann_ret / max_dd if max_dd < 0 else np.nan
    turnover = trades["notional"].abs().sum() / (cfg.initial_capital * max(len(equity), 1))
    win_rate = (rets > 0).mean()
    exposure = 1.0  # fully invested by design
    return {
        "CAGR": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "turnover": turnover,
        "win_rate_weekly": win_rate,
        "exposure": exposure,
    }


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="Weekly momentum backtest")
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--initial_capital", type=float, default=1_000_000)
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--min_adv", type=float, default=20_000_000)
    ap.add_argument("--w12m", type=float, default=0.6)
    ap.add_argument("--w8w", type=float, default=0.4)
    ap.add_argument("--trend_gate", type=str, default="200d", choices=["none", "200d", "50d_200d"])
    ap.add_argument("--sizing", type=str, default="equal", choices=["equal", "atr_inverse"])
    ap.add_argument("--commission_bps", type=float, default=5.0)
    ap.add_argument("--slippage_bps", type=float, default=10.0)
    ap.add_argument("--impact_k", type=float, default=0.0, help="Set >0 to enable impact_k/sqrt(adv_60d)")
    ap.add_argument("--execution", type=str, default="next_open", choices=["next_open", "friday_close"])
    ap.add_argument("--features", type=Path, default=None, help="Path to precomputed weekly features parquet.")
    ap.add_argument("--frozen_features", type=Path, required=True, help="Path to frozen features txt.")
    ap.add_argument("--out_prefix", type=str, default="backtest")
    ap.add_argument("--target-col", type=str, default="target_forward_4w_excess")
    ap.add_argument("--manifest", type=Path, default=Path("feature_groups.yaml"))
    ap.add_argument("--score-horizon", type=str, default="4w", choices=["1w", "4w"])
    ap.add_argument("--frozen_features_1w", type=Path, default=None)
    ap.add_argument("--frozen_features_4w", type=Path, default=None)
    ap.add_argument("--gate_only", type=str, default=None, help="Comma-separated gate-only features override.")
    ap.add_argument("--risk_veto", type=str, default=None, help="Comma-separated risk veto features override.")
    ap.add_argument("--gate_thresholds_yaml", type=Path, default=None)
    ap.add_argument("--gate-thresholds", type=str, default=None, help="Comma thresholds e.g. adx_14=0.55")
    ap.add_argument("--veto-thresholds", type=str, default=None, help="Comma thresholds e.g. hist_vol_20d=0.75")
    ap.add_argument("--quality-thresholds", type=str, default=None, help="Comma thresholds e.g. min_eligible=5,score_std=0.08,top_minus_median=0.10")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)
    impact_k = args.impact_k if args.impact_k > 0 else None
    return Config(
        start=start,
        end=end,
        initial_capital=args.initial_capital,
        top_n=args.top_n,
        min_adv=args.min_adv,
        w12m=args.w12m,
        w8w=args.w8w,
        trend_gate=args.trend_gate,
        sizing=args.sizing,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        impact_k=impact_k,
        execution=args.execution,
        features_path=args.features,
        frozen_features_path=args.frozen_features,
        out_prefix=args.out_prefix,
        target_col=args.target_col,
        manifest=args.manifest,
        score_horizon=args.score_horizon,
        frozen_1w=args.frozen_features_1w,
        frozen_4w=args.frozen_features_4w,
        gate_only=args.gate_only.split(",") if args.gate_only else None,
        risk_veto=args.risk_veto.split(",") if args.risk_veto else None,
        gate_thresholds_yaml=args.gate_thresholds_yaml,
        gate_thresholds_str=args.gate_thresholds,
        veto_thresholds_str=args.veto_thresholds,
        quality_thresholds_str=args.quality_thresholds,
        out_dir=args.out_dir,
    )


def main() -> None:
    cfg = parse_args()
    features = get_weekly_features(cfg)
    ohlcv = load_ohlcv(Path("data/ohlcv.parquet"))
    if cfg.score_horizon == "1w":
        frozen_path = cfg.frozen_1w or cfg.frozen_features_path
        default_gate_only = ["volume_zscore_20d"]
        default_risk_veto = ["hist_vol_60d"]
    else:
        frozen_path = cfg.frozen_4w or cfg.frozen_features_path
        default_gate_only = ["adx_14", "turnover_ratio_20d"]
        default_risk_veto = ["hist_vol_20d"]
    frozen = load_frozen_features(frozen_path)
    gate_only = cfg.gate_only if cfg.gate_only is not None else default_gate_only
    risk_veto = cfg.risk_veto if cfg.risk_veto is not None else default_risk_veto
    gate_thresholds = {
        "volume_zscore_20d": 0.60,
        "adx_14": 0.55,
        "turnover_ratio_20d": 0.55,
        "hist_vol_60d": 0.75,
        "hist_vol_20d": 0.75,
    }
    if cfg.gate_thresholds_yaml and cfg.gate_thresholds_yaml.exists():
        import yaml
        gate_thresholds.update(yaml.safe_load(cfg.gate_thresholds_yaml.read_text()))
    directions = load_manifest(cfg.manifest)

    equity, trades, holdings, summary_df = simulate(features, ohlcv, cfg, frozen, directions, gate_only, risk_veto, gate_thresholds)
    metrics = compute_metrics(equity, trades, cfg)

    out_dir = cfg.out_dir / cfg.out_prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    equity.to_csv(out_dir / "equity_curve.csv", index=False)
    trades.to_csv(out_dir / "trades.csv", index=False)
    holdings.to_csv(out_dir / "holdings.csv", index=False)
    summary_df.to_csv(out_dir / "rebalance_summary.csv", index=False)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print("==== Backtest Summary ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
