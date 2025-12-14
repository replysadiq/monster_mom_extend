"""
Incremental information content diagnostics on weekly panel data.

For each feature and group, this script computes:
    - Marginal IC: mean weekly Spearman correlation(feature, target)
    - Incremental IC: mean weekly Spearman correlation(residual Y|X, target)
      where residualization is Y_rank minus beta * X_rank (per-week cross-section).

Decision heuristics (reported, not enforced):
    Drop Y if |corr(X, Y)| > 0.8, IC(Y) <= IC(X), and incremental IC(Y | X) ~ 0.
Only compare within economic groups.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


TARGET_COL = "target_forward_4w_excess"
MIN_CROSS_SECTION = 20  # minimum symbols per week to include the week


@dataclass
class FeatureResult:
    feature: str
    group: str
    marginal_ic: float
    marginal_std: float
    marginal_t: float
    marginal_hit_rate: float
    marginal_n_weeks: int
    base_feature: str
    base_ic: float
    base_n_weeks: int
    corr_with_base: float
    incremental_ic_vs_base: float
    incremental_std: float
    incremental_t: float
    incremental_hit_rate: float
    incremental_n_weeks: int


def _tstat(mean: float, std: float, n: int) -> float:
    if std == 0 or n == 0:
        return float("nan")
    return mean / std * np.sqrt(n)


def marginal_ic(df: pd.DataFrame, feature: str, target: str) -> Tuple[float, float, float, int]:
    """Mean weekly Spearman IC for feature vs target, with stability stats."""
    ics: List[float] = []
    for _, g in df[["week_date", feature, target]].dropna().groupby("week_date"):
        if len(g) < MIN_CROSS_SECTION:
            continue
        ic, _ = spearmanr(g[feature], g[target])
        if np.isnan(ic):
            continue
        ics.append(ic)
    if not ics:
        return float("nan"), float("nan"), float("nan"), 0
    arr = np.array(ics)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    tstat = _tstat(mean, std, len(arr))
    return mean, std, tstat, len(arr)


def incremental_ic(
    df: pd.DataFrame, x_col: str, y_col: str, target: str
) -> Tuple[float, float, float, int]:
    """Mean weekly incremental IC of y given x, with stability stats."""
    ics: List[float] = []
    for _, g in df[["week_date", x_col, y_col, target]].dropna().groupby("week_date"):
        if len(g) < MIN_CROSS_SECTION:
            continue
        x = g[x_col].rank().to_numpy()
        y = g[y_col].rank().to_numpy()
        t = g[target].rank().to_numpy()

        x = x - x.mean()
        y = y - y.mean()
        varx = x.var()
        if varx == 0:
            continue
        beta = (x * y).mean() / varx
        y_resid = y - beta * x

        ic, _ = spearmanr(y_resid, t)
        if np.isnan(ic):
            continue
        ics.append(ic)
    if not ics:
        return float("nan"), float("nan"), float("nan"), 0
    arr = np.array(ics)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    tstat = _tstat(mean, std, len(arr))
    return mean, std, tstat, len(arr)


def mean_weekly_corr(df: pd.DataFrame, a: str, b: str) -> float:
    """Mean weekly cross-sectional Spearman correlation between two features."""
    cors: List[float] = []
    for _, g in df[["week_date", a, b]].dropna().groupby("week_date"):
        if len(g) < MIN_CROSS_SECTION:
            continue
        c, _ = spearmanr(g[a], g[b])
        if np.isnan(c):
            continue
        cors.append(c)
    return float(np.nanmean(cors)) if cors else float("nan")


def run_analysis(
    df: pd.DataFrame, groups: Dict[str, Iterable[str]], target: str = TARGET_COL
) -> List[FeatureResult]:
    results: List[FeatureResult] = []

    # Precompute marginal ICs
    marginals = {
        feat: marginal_ic(df, feat, target)
        for feats in groups.values()
        for feat in feats
    }

    for group, feats in groups.items():
        feats = list(feats)
        if not feats:
            continue
        valid = [f for f in feats if np.isfinite(marginals.get(f, (np.nan,))[0])]
        if not valid:
            continue
        # Choose base as highest marginal IC within group (simple heuristic)
        base_feature = max(valid, key=lambda f: marginals[f][0])
        base_ic, base_std, base_t, base_n = marginals[base_feature]

        for feat in feats:
            corr = mean_weekly_corr(df, base_feature, feat) if feat != base_feature else 1.0
            if feat != base_feature:
                inc_mean, inc_std, inc_t, inc_n = incremental_ic(df, base_feature, feat, target)
            else:
                inc_mean = inc_std = inc_t = float("nan")
                inc_n = 0
            marg_mean, marg_std, marg_t, marg_n = marginals[feat]
            # Compute hit rates (fraction of positive weekly ICs) for marginal and incremental
            marg_hit = float("nan")
            if marg_n > 0:
                weekly_marg_ics = []
                for _, g in df[["week_date", feat, target]].dropna().groupby("week_date"):
                    if len(g) < MIN_CROSS_SECTION:
                        continue
                    ic, _ = spearmanr(g[feat], g[target])
                    if np.isnan(ic):
                        continue
                    weekly_marg_ics.append(ic)
                if weekly_marg_ics:
                    marg_hit = float(np.mean(np.array(weekly_marg_ics) > 0))

            inc_hit = float("nan")
            if inc_n > 0 and feat != base_feature:
                weekly_inc_ics = []
                for _, g in df[["week_date", base_feature, feat, target]].dropna().groupby("week_date"):
                    if len(g) < MIN_CROSS_SECTION:
                        continue
                    x = g[base_feature].rank().to_numpy()
                    y = g[feat].rank().to_numpy()
                    t = g[target].rank().to_numpy()
                    x = x - x.mean()
                    y = y - y.mean()
                    varx = x.var()
                    if varx == 0:
                        continue
                    beta = (x * y).mean() / varx
                    y_resid = y - beta * x
                    ic, _ = spearmanr(y_resid, t)
                    if np.isnan(ic):
                        continue
                    weekly_inc_ics.append(ic)
                if weekly_inc_ics:
                    inc_hit = float(np.mean(np.array(weekly_inc_ics) > 0))
            results.append(
                FeatureResult(
                    feature=feat,
                    group=group,
                    marginal_ic=marg_mean,
                    marginal_std=marg_std,
                    marginal_t=marg_t,
                    marginal_hit_rate=marg_hit,
                    marginal_n_weeks=marg_n,
                    base_feature=base_feature,
                    base_ic=base_ic,
                    base_n_weeks=base_n,
                    corr_with_base=corr,
                    incremental_ic_vs_base=inc_mean,
                    incremental_std=inc_std,
                    incremental_t=inc_t,
                    incremental_hit_rate=inc_hit,
                    incremental_n_weeks=inc_n,
                )
            )
    return results


def save_summary(results: List[FeatureResult], out_path: Path) -> None:
    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(out_path, index=False)
    print(f"Saved summary to {out_path} (rows={len(df)})")
    print(df.sort_values(["group", "marginal_ic"], ascending=[True, False]).to_string(index=False))


def default_groups() -> Dict[str, Tuple[str, ...]]:
    return {
        "momentum": (
            "ret_1w",
            "ret_2w",
            "ret_1m",
            "ret_3m",
            "ret_6m",
            "ret_8w",
            "ret_12m",
            "excess_ret_3m_vs_index",
        ),
        "trend": (
            "pct_above_20d_sma",
            "pct_above_50d_sma",
            "pct_above_200d_sma",
            "pct_from_52w_high",
            "adx_14",
            "macd_hist_12_26_9",
            "bollinger_pctb_20_2",
            "bollinger_bw_20_2",
            "rsi_14",
            "donchian_pos_20",
            "supertrend_state_10_3",
            "ema_slope_20d",
        ),
        "volatility": (
            "hist_vol_20d",
            "hist_vol_60d",
            "gk_vol_20d",
            "hv_percentile_1y",
            "atr_pct_14",
            "beta_60d",
        ),
        "liquidity": (
            "adv_20d",
            "adv_60d",
            "adv_median_60d",
            "turnover_ratio_20d",
            "volume_zscore_20d",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental IC diagnostics on weekly panel.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/weekly_features.parquet"),
        help="Path to weekly parquet file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/incremental_ic_summary.csv"),
        help="Where to write the summary CSV.",
    )
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD), optional.")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD), optional.")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    if "week_date" not in df.index.names:
        raise ValueError("Expected week_date in index; ensure data is indexed by (symbol, week_date).")
    df = df.reset_index()  # easier grouping on week_date
    if args.start:
        start = pd.to_datetime(args.start)
        df = df[df["week_date"] >= start]
    if args.end:
        end = pd.to_datetime(args.end)
        df = df[df["week_date"] <= end]

    groups = default_groups()
    results = run_analysis(df, groups, target=TARGET_COL)
    save_summary(results, args.out)


if __name__ == "__main__":
    main()
