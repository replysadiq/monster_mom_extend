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
from price_scale import apply_price_scale
from data_quality_gate import add_close_quality_flags, quarantine_report


RESULTS_DIR = Path("results")


@dataclass
class Config:
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    ohlcv_path: Path
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
    gate_thresholds_str: Optional[str] = None
    veto_thresholds_str: Optional[str] = None
    quality_thresholds_str: Optional[str] = None
    out_dir: Optional[str] = None
    composite_mode: str = "blend"  # trend4w | mr1w | blend
    w1: float = 0.6
    w4: float = 0.4
    selection_mode: str = "trend_confirmed_mr"  # trend_confirmed_mr (default)
    trend_shortlist_size: int = 30


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    scale_path = Path("data/scale_factors_by_symbol.csv")
    scale_override = None
    if scale_path.exists():
        scale_df = pd.read_csv(scale_path)
        if {"symbol", "scale_factor"}.issubset(scale_df.columns):
            scale_override = scale_df.set_index("symbol")["scale_factor"]
    df, _ = apply_price_scale(df, symbol_col="symbol", scale_override=scale_override)
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


def resolve_out_paths(cfg: Config) -> Path:
    if cfg.out_dir:
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    out_prefix = Path(cfg.out_prefix) if cfg.out_prefix else Path("output/run")
    out_dir = out_prefix.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def select_portfolio(
    week_date: pd.Timestamp,
    week_df: pd.DataFrame,
    cfg: Config,
    frozen_features_mr: List[str],
    frozen_features_trend: List[str],
    directions: Dict[str, int],
    gate_only: List[str],
    risk_veto: List[str],
    gate_thresholds: Dict[str, float],
    summary_records: List[Dict[str, object]],
) -> pd.DataFrame:
    # Order enforced: liquidity -> gates -> trend shortlist -> MR rank -> quality kill-switch
    if cfg.trend_shortlist_size < cfg.top_n:
        raise RuntimeError("trend_shortlist_size must be >= top_n")
    df = week_df.copy()
    df = df[df["adv_60d"] >= cfg.min_adv]

    if df.empty:
        summary_records.append(
            {
                "week_date": pd.Timestamp(week_date),
                "eligible_count": 0,
                "selected_count": 0,
                "cash_weight": 1.0,
                "score_dispersion": np.nan,
                "top_minus_median": np.nan,
                "quality_ok": False,
                "score_corr": np.nan,
                "overlap_topn": np.nan,
                "score_corr": np.nan,
                "overlap_topn": np.nan,
                "trend_shortlist_size": shortlist_size if "shortlist_size" in locals() else np.nan,
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
                "week_date": pd.Timestamp(week_date),
                "eligible_count": 0,
                "selected_count": 0,
                "cash_weight": 1.0,
                "score_dispersion": np.nan,
                "top_minus_median": np.nan,
                "quality_ok": False,
                "score_corr": np.nan,
                "overlap_topn": np.nan,
                "score_corr": corr_scores,
                "overlap_topn": overlap_pct,
                "trend_shortlist_size": shortlist_size,
            }
        )
        return pd.DataFrame(columns=["symbol", "score"])

    def compute_score(features: List[str]) -> pd.Series:
        if not features:
            return pd.Series(dtype=float)
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise RuntimeError(f"Scoring features missing in weekly data for {week_date}: {missing}")
        usable = [f for f in features if f not in gate_only and f not in risk_veto]
        if not usable:
            return pd.Series(dtype=float)
        cols = []
        for feat in usable:
            ranks = df[feat].rank(pct=True)
            if directions.get(feat, 1) == -1:
                ranks = 1 - ranks
            cols.append(ranks.rename(feat))
        if not cols:
            return pd.Series(dtype=float)
        z = pd.concat(cols, axis=1)
        return z.mean(axis=1)

    score_mr = compute_score(frozen_features_mr)
    score_trend = compute_score(frozen_features_trend)

    if score_mr.empty:
        raise RuntimeError("MR(1W) score is empty; check frozen 1W features.")
    if score_trend.empty:
        raise RuntimeError("Trend(4W) score is empty; check frozen 4W features.")

    # Align indices for diagnostics
    align_idx = score_mr.index.intersection(score_trend.index)
    score_mr = score_mr.reindex(align_idx)
    score_trend = score_trend.reindex(align_idx)
    df = df.loc[align_idx]
    df["score_mr_1w"] = score_mr
    df["score_trend_4w"] = score_trend

    # Two-stage: shortlist by trend, then rank by MR within shortlist
    shortlist_size = cfg.trend_shortlist_size
    trend_ranked = df.sort_values("score_trend_4w", ascending=False)
    shortlist = trend_ranked.head(shortlist_size)
    df = shortlist.sort_values("score_mr_1w", ascending=False)
    df["score"] = df["score_mr_1w"]

    # Diagnostics
    corr_scores = np.nan
    overlap_pct = np.nan
    merged = df[["score_mr_1w", "score_trend_4w"]].dropna()
    if len(merged) >= 20:
        corr_scores = merged.corr(method="spearman").iloc[0, 1]
    top_tr_syms = trend_ranked.head(cfg.top_n).index if not trend_ranked.empty else []
    top_mr_syms = df.head(cfg.top_n).index if not df.empty else []
    if len(top_mr_syms) and len(top_tr_syms):
        overlap = len(set(top_mr_syms).intersection(set(top_tr_syms)))
        overlap_pct = overlap / cfg.top_n

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
                "week_date": pd.Timestamp(week_date),
                "eligible_count": len(df),
                "selected_count": 0,
                "cash_weight": 1.0,
            "score_dispersion": score_dispersion,
            "top_minus_median": top_minus_median,
            "quality_ok": False,
            "score_corr": corr_scores,
            "overlap_topn": overlap_pct,
            "trend_shortlist_size": shortlist_size,
        }
    )
    print(
        f"{pd.Timestamp(week_date)} "
        f"universe={len(week_df)}, eligible={len(df)}, selected=0, cash=1.0, quality_ok=False, "
        f"score_std={score_dispersion:.4f}, top_minus_median={top_minus_median:.4f}, "
        f"score_corr={corr_scores if not np.isnan(corr_scores) else 'nan'}, "
        f"overlap_topn={overlap_pct if not np.isnan(overlap_pct) else 'nan'}, "
        f"trend_shortlist_size={shortlist_size}"
    )
    return pd.DataFrame(columns=["symbol", "score"])

    top = df.nlargest(cfg.top_n, "score")
    selected_count = len(top)
    cash_w = max(0.0, 1 - 0.1 * selected_count)
    summary_records.append(
        {
            "week_date": pd.Timestamp(week_date),
            "eligible_count": len(df),
            "selected_count": selected_count,
            "cash_weight": cash_w,
            "score_dispersion": score_dispersion,
            "top_minus_median": top_minus_median,
            "quality_ok": True,
            "score_corr": corr_scores,
            "overlap_topn": overlap_pct,
            "trend_shortlist_size": shortlist_size,
        }
    )
    print(
        f"{pd.Timestamp(week_date)} "
        f"universe={len(week_df)}, eligible={len(df)}, selected={selected_count}, cash={cash_w:.2f}, "
        f"quality_ok=True, score_std={score_dispersion:.4f}, top_minus_median={top_minus_median:.4f}, "
        f"score_corr={corr_scores if not np.isnan(corr_scores) else 'nan'}, "
        f"overlap_topn={overlap_pct if not np.isnan(overlap_pct) else 'nan'}, "
        f"trend_shortlist_size={shortlist_size}"
    )
    return top[["score"]]


def get_exit_prices(
    ohlcv: pd.DataFrame, symbols: List[str], rebalance_date: pd.Timestamp, execution: str
) -> Dict[str, Tuple[pd.Timestamp, float]]:
    """Return exit price and date for symbols as of rebalance_date."""
    prices: Dict[str, Tuple[pd.Timestamp, float]] = {}
    for sym in symbols:
        data = ohlcv.xs(sym, level=0).sort_index()
        if execution == "friday_close":
            row = data[data.index <= rebalance_date].tail(1)
            if not row.empty:
                prices[sym] = (row.index[0], float(row["close"].iloc[0]))
        else:
            row = data[data.index > rebalance_date].head(1)
            if not row.empty:
                prices[sym] = (row.index[0], float(row["open"].iloc[0]))
    return prices


def get_entry_prices(
    ohlcv: pd.DataFrame, symbols: List[str], rebalance_date: pd.Timestamp, execution: str
) -> Dict[str, Tuple[pd.Timestamp, float]]:
    """Return entry price and date after rebalance_date."""
    prices: Dict[str, Tuple[pd.Timestamp, float]] = {}
    for sym in symbols:
        data = ohlcv.xs(sym, level=0).sort_index()
        row = data[data.index > rebalance_date].head(1)
        if not row.empty:
            prices[sym] = (row.index[0], float(row["open"].iloc[0]))
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
    frozen_features_mr: List[str],
    frozen_features_trend: List[str],
    directions: Dict[str, int],
    gate_only: List[str],
    risk_veto: List[str],
    gate_thresholds: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rebalance_dates = compute_rebalance_dates(features, cfg.start, cfg.end)
    portfolio_value = cfg.initial_capital
    out_base = Path(cfg.out_dir) if cfg.out_dir else RESULTS_DIR
    out_base.mkdir(parents=True, exist_ok=True)
    equity_records = []
    trade_records: List[Dict[str, object]] = []
    holding_records: List[Dict[str, object]] = []
    summary_records: List[Dict[str, object]] = []

    prev_weights: Dict[str, float] = {}
    prev_entry: Dict[str, Tuple[pd.Timestamp, float]] = {}
    prev_cash_weight = 1.0
    debug_ret: List[Dict[str, object]] = []
    debug_rebalance_rows: List[Dict[str, object]] = []

    for i, reb_date in enumerate(rebalance_dates):
        coverage = np.nan
        week_slice = features.xs(reb_date, level="week_date")
        # Defensive drop of any target/forward columns to avoid leakage
        leak_cols = [c for c in week_slice.columns if c.startswith("target_") or c.startswith("fwd_")]
        if leak_cols:
            week_slice = week_slice.drop(columns=leak_cols, errors="ignore")

        # Realize returns from previous period
        if i > 0:
            if prev_weights:
                exit_prices = get_exit_prices(ohlcv, list(prev_weights.keys()), reb_date, cfg.execution)
                coverage = sum(sym in exit_prices for sym in prev_weights) / max(len(prev_weights), 1)
                if coverage < 0.95:
                    raise RuntimeError(f"Low exit price coverage at {reb_date}: {coverage:.2%}")
                realized_ret = 0.0
                cash_w = float(prev_cash_weight)
                invested_w = float(sum(prev_weights.values()))
                r_list = []

                for sym, w in prev_weights.items():
                    if w < 0:
                        raise ValueError(f"Negative weight for {sym}")
                    entry = prev_entry.get(sym)
                    exitp = exit_prices.get(sym)
                    # if no entry, treat as cash; if no exit, hard fail (data issue)
                    if entry is None or entry[1] <= 0 or not np.isfinite(entry[1]):
                        cash_w += w
                        invested_w -= w
                        continue
                    if exitp is None or exitp[1] <= 0 or not np.isfinite(exitp[1]):
                        raise RuntimeError(f"Missing/invalid EXIT for {sym} at {reb_date} (execution={cfg.execution})")
                    r = exitp[1] / entry[1] - 1.0
                    realized_ret += w * r
                    r_list.append(r)
                    debug_rebalance_rows.append(
                        {
                            "rebalance_date": reb_date,
                            "symbol": sym,
                            "entry_date": entry[0],
                            "entry_price": entry[1],
                            "exit_date": exitp[0],
                            "exit_price": exitp[1],
                            "return": r,
                        }
                    )

                total_w = cash_w + invested_w
                if not np.isclose(total_w, 1.0, atol=1e-6):
                    raise ValueError(f"Weight sum != 1 at {reb_date}: cash={cash_w}, invested={invested_w}, total={total_w}")
                if not np.isfinite(realized_ret):
                    raise RuntimeError(f"Non-finite realized_ret at {reb_date}")
                if portfolio_value * (1.0 + realized_ret) <= 0:
                    raise RuntimeError(f"Portfolio value would collapse at {reb_date} with realized_ret {realized_ret}")
                portfolio_value *= (1.0 + realized_ret)
                equity_records.append({"date": reb_date, "portfolio_value": portfolio_value, "return": realized_ret})
                if r_list:
                    debug_ret.append(
                        {
                            "date": reb_date,
                            "min_r": float(np.min(r_list)),
                            "max_r": float(np.max(r_list)),
                            "neg_count": int(sum(np.array(r_list) < 0)),
                            "count": len(r_list),
                        }
                    )
            else:
                equity_records.append({"date": reb_date, "portfolio_value": portfolio_value, "return": 0.0})
        elif i == 0:
            equity_records.append({"date": reb_date, "portfolio_value": portfolio_value, "return": 0.0})

        # New selection for next period
        selection = select_portfolio(
            reb_date,
            week_slice,
            cfg,
            frozen_features_mr,
            frozen_features_trend,
            directions,
            gate_only,
            risk_veto,
            gate_thresholds,
            summary_records,
        )
        if selection.empty:
            prev_weights = {}
            prev_entry = {}
            prev_cash_weight = 1.0
            holding_records.append({"rebalance_date": reb_date, "symbol": "CASH", "weight": 1.0})
            # annotate coverage for summary (no exits on first period)
            if summary_records:
                summary_records[-1]["exit_coverage"] = np.nan
                summary_records[-1]["entry_coverage"] = 1.0
            continue

        old_weights = prev_weights
        sel_df = week_slice.loc[selection.index]
        selected_count = len(sel_df)
        cash_w = max(0.0, 1 - 0.1 * selected_count)
        sym_weight = (1 - cash_w) / selected_count if selected_count > 0 else 0.0
        weights = {sym: sym_weight for sym in sel_df.index}

        entry_symbols = list(set(weights.keys()) | set(old_weights.keys()))
        entry_prices = get_entry_prices(ohlcv, entry_symbols, reb_date, cfg.execution)
        missing_entry = [
            sym
            for sym in weights
            if entry_prices.get(sym) is None
            or entry_prices.get(sym, (None, np.nan))[1] <= 0
            or not np.isfinite(entry_prices.get(sym, (None, np.nan))[1])
        ]
        if missing_entry:
            print(
                f"[WARN] {reb_date} missing entry prices for {len(missing_entry)}/{len(weights)} symbols; reallocating to cash."
            )
        valid_weights: Dict[str, float] = {}
        adj_cash = cash_w
        for sym, w in weights.items():
            price = entry_prices.get(sym)
            if price is None or price[1] <= 0 or not np.isfinite(price[1]):
                adj_cash += w
                continue
            valid_weights[sym] = w
        invested = sum(valid_weights.values())
        adj_cash = max(0.0, adj_cash)
        if invested > 0:
            scale = (1.0 - adj_cash) / invested
            valid_weights = {k: v * scale for k, v in valid_weights.items()}
        # log trades as weight deltas against previous holdings
        for sym in set(old_weights.keys()).union(valid_weights.keys()):
            old_w = old_weights.get(sym, 0.0)
            new_w = valid_weights.get(sym, 0.0)
            delta_w = new_w - old_w
            if abs(delta_w) < 1e-6:
                continue
            price = entry_prices.get(sym, np.nan)
            if price is None or not np.isfinite(price[1]) or price[1] <= 0:
                continue
            notional = abs(delta_w) * portfolio_value
            side = "buy" if delta_w > 0 else "sell"
            trade_records.append(
                {
                    "rebalance_date": reb_date,
                    "symbol": sym,
                    "side": side,
                    "delta_weight": delta_w,
                    "price": price[1],
                    "notional": notional,
                }
            )
        # Store fresh entry prices for this rebalance only
        prev_weights = valid_weights
        prev_entry = {sym: entry_prices[sym] for sym in valid_weights}
        prev_cash_weight = adj_cash

        holding_records.append({"rebalance_date": reb_date, "symbol": "CASH", "weight": prev_cash_weight})
        for sym, w in prev_weights.items():
            holding_records.append({"rebalance_date": reb_date, "symbol": sym, "weight": w})

        entry_cov = 1.0 if not weights else len(valid_weights) / len(weights)
        if summary_records:
            summary_records[-1]["exit_coverage"] = coverage if i > 0 else np.nan
            summary_records[-1]["entry_coverage"] = entry_cov

    equity_df = pd.DataFrame(equity_records).sort_values("date").reset_index(drop=True)
    if equity_df["date"].duplicated().any():
        dups = equity_df[equity_df["date"].duplicated(keep=False)]["date"].unique()
        raise RuntimeError(f"Duplicate dates in equity curve: {dups[:5]}")
    equity_df["cum_return"] = (1 + equity_df["return"]).cumprod()
    peak = equity_df["cum_return"].cummax()
    equity_df["drawdown"] = equity_df["cum_return"] / peak - 1

    trades_df = pd.DataFrame(trade_records)
    holdings_df = pd.DataFrame(holding_records)
    summary_df = pd.DataFrame(summary_records)
    if debug_ret:
        pd.DataFrame(debug_ret).to_csv(out_base / "debug_symbol_returns.csv", index=False)
    if debug_rebalance_rows:
        pd.DataFrame(debug_rebalance_rows).to_csv(out_base / "debug_rebalance_returns.csv", index=False)
        dbg = pd.DataFrame(debug_rebalance_rows)
        min_by_reb = dbg.groupby("rebalance_date")["return"].min()
        if (min_by_reb >= 0).all():
            raise RuntimeError("All per-position returns are non-negative across the window; price series likely monotonic.")
    if (equity_df["portfolio_value"] <= 0).any():
        raise ValueError("portfolio_value dropped to <= 0")
    if (equity_df["return"] <= -1).any():
        raise ValueError("Return <= -1 encountered")
    if trades_df.empty:
        print("[WARN] trades empty; turnover will be zero.")
    return equity_df, trades_df, holdings_df, summary_df


def compute_metrics(equity: pd.DataFrame, trades: pd.DataFrame, holdings: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    rets = pd.Series(equity["return"]).replace([np.inf, -np.inf], np.nan).dropna()
    rets = rets.clip(lower=-0.999999)
    weeks_per_year = 52
    log_g = np.log1p(rets).sum()
    ann_ret = np.expm1(log_g * (weeks_per_year / max(len(rets), 1)))
    ann_vol = rets.std(ddof=0) * np.sqrt(weeks_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd = equity["drawdown"].min()
    calmar = -ann_ret / max_dd if max_dd < 0 else np.nan
    if trades is None or trades.empty:
        turnover = 0.0
        print("[WARN] trades empty; turnover set to 0.0")
    else:
        tr = trades.copy()
        if "notional" not in tr.columns:
            if {"qty", "price"}.issubset(tr.columns):
                tr["notional"] = (tr["qty"] * tr["price"]).abs()
            elif "trade_value" in tr.columns:
                tr["notional"] = tr["trade_value"].abs()
            else:
                tr["notional"] = 0.0
        turnover = tr["notional"].abs().sum() / (cfg.initial_capital * max(len(equity), 1))
    win_rate = (rets > 0).mean()
    exposure = np.nan
    if holdings is not None and not holdings.empty:
        cash_weights = (
            holdings[holdings["symbol"] == "CASH"]
            .groupby("rebalance_date")["weight"]
            .sum()
        )
        exposure_series = 1 - cash_weights.reindex(equity["date"]).fillna(1.0)
        exposure = exposure_series.mean()
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
    ap.add_argument("--ohlcv", type=Path, default=Path("data/ohlcv.parquet"))
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
    ap.add_argument("--composite-mode", type=str, default="blend", choices=["trend4w", "mr1w", "blend"])
    ap.add_argument("--w1", type=float, default=0.6, help="Weight for MR(1W) score in blend mode.")
    ap.add_argument("--w4", type=float, default=0.4, help="Weight for Trend(4W) score in blend mode.")
    ap.add_argument("--selection-mode", type=str, default="trend_confirmed_mr", choices=["trend_confirmed_mr"])
    ap.add_argument("--trend-shortlist-size", type=int, default=30, help="Top-M shortlist by trend before MR rank.")
    args = ap.parse_args()
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)
    impact_k = args.impact_k if args.impact_k > 0 else None
    return Config(
        start=start,
        end=end,
        ohlcv_path=args.ohlcv,
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
        out_dir=str(args.out_dir) if args.out_dir else None,
        composite_mode=args.composite_mode,
        w1=args.w1,
        w4=args.w4,
        selection_mode=args.selection_mode,
        trend_shortlist_size=args.trend_shortlist_size,
    )


def main() -> None:
    cfg = parse_args()
    features_raw = get_weekly_features(cfg)
    # Scale invariants on weekly close to catch regressions
    med_weekly = features_raw.groupby("symbol")["close"].median()
    bad_scale = med_weekly[(med_weekly > 5e5) | (med_weekly < 1.0)]
    if not bad_scale.empty:
        raise RuntimeError(
            f"Weekly close scale invalid for {len(bad_scale)} symbols. Example:\n{bad_scale.head()}"
        )
    flagged = add_close_quality_flags(features_raw.reset_index())
    features = flagged[~flagged["is_bad_close"]].set_index(["symbol", "week_date"]).sort_index()
    out_base = Path(cfg.out_dir) if cfg.out_dir else RESULTS_DIR
    out_base.mkdir(parents=True, exist_ok=True)
    qr = quarantine_report(flagged)
    qr.to_csv(out_base / "quarantine_bad_close_weeks.csv", index=False)
    if cfg.start and cfg.end:
        qr_window = qr[(pd.to_datetime(qr["week_date"]) >= cfg.start) & (pd.to_datetime(qr["week_date"]) <= cfg.end)]
        qr_window.to_csv(out_base / "quarantine_bad_close_weeks_window.csv", index=False)
    ohlcv = load_ohlcv(cfg.ohlcv_path)
    frozen_path_1w = cfg.frozen_1w or cfg.frozen_features_path
    frozen_path_4w = cfg.frozen_4w or cfg.frozen_features_path
    frozen_mr = load_frozen_features(frozen_path_1w)
    frozen_trend = load_frozen_features(frozen_path_4w)

    # Ensure required lists exist based on composite mode
    if cfg.composite_mode == "mr1w" and not frozen_mr:
        raise RuntimeError("Composite mode mr1w selected but frozen 1W feature list is empty.")
    if cfg.composite_mode == "trend4w" and not frozen_trend:
        raise RuntimeError("Composite mode trend4w selected but frozen 4W feature list is empty.")
    if cfg.composite_mode == "blend" and (not frozen_mr or not frozen_trend):
        raise RuntimeError("Blend mode requires both 1W and 4W frozen feature lists.")

    default_gate_only = ["volume_zscore_20d"] if cfg.composite_mode == "mr1w" else ["adx_14", "turnover_ratio_20d"]
    default_risk_veto = ["hist_vol_60d"] if cfg.composite_mode == "mr1w" else ["hist_vol_20d"]
    gate_only = cfg.gate_only if cfg.gate_only is not None else default_gate_only
    risk_veto = cfg.risk_veto if cfg.risk_veto is not None else default_risk_veto
    gate_defaults = {
        "volume_zscore_20d": 0.60,
        "adx_14": 0.55,
        "turnover_ratio_20d": 0.55,
    }
    veto_defaults = {
        "hist_vol_60d": 0.75,
        "hist_vol_20d": 0.75,
    }
    quality_defaults = {"min_eligible": 5, "score_std": 0.08, "top_minus_median": 0.10}
    gate_thresholds = gate_defaults.copy()
    if cfg.gate_thresholds_yaml and cfg.gate_thresholds_yaml.exists():
        import yaml
        gate_thresholds.update(yaml.safe_load(cfg.gate_thresholds_yaml.read_text()))
    gate_thresholds = parse_thresholds(cfg.gate_thresholds_str, gate_thresholds)
    veto_thresholds = parse_thresholds(cfg.veto_thresholds_str, veto_defaults)
    quality_thresholds = parse_thresholds(cfg.quality_thresholds_str, quality_defaults)
    directions = load_manifest(cfg.manifest)

    equity, trades, holdings, summary_df = simulate(
        features,
        ohlcv,
        cfg,
        frozen_mr,
        frozen_trend,
        directions,
        gate_only,
        risk_veto,
        gate_thresholds,
    )
    metrics = compute_metrics(equity, trades, holdings, cfg)

    out_base = Path(cfg.out_dir) if cfg.out_dir else RESULTS_DIR
    out_base.mkdir(parents=True, exist_ok=True)
    equity.to_csv(out_base / "equity_curve.csv", index=False)
    trades.to_csv(out_base / "trades.csv", index=False)
    holdings.to_csv(out_base / "holdings.csv", index=False)
    summary_df.to_csv(out_base / "rebalance_summary.csv", index=False)
    with (out_base / "summary.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print("==== Backtest Summary ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
