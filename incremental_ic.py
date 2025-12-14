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
    base_feature: str
    base_ic: float
    corr_with_base: float
    incremental_ic_vs_base: float


def marginal_ic(df: pd.DataFrame, feature: str, target: str) -> float:
    """Mean weekly Spearman IC for feature vs target."""
    ics: List[float] = []
    for _, g in df[["week_date", feature, target]].dropna().groupby("week_date"):
        if len(g) < MIN_CROSS_SECTION:
            continue
        ic, _ = spearmanr(g[feature], g[target])
        if np.isnan(ic):
            continue
        ics.append(ic)
    return float(np.nanmean(ics)) if ics else float("nan")


def incremental_ic(df: pd.DataFrame, x_col: str, y_col: str, target: str) -> float:
    """Mean weekly incremental IC of y given x."""
    ics: List[float] = []
    for _, g in df[["week_date", x_col, y_col, target]].dropna().groupby("week_date"):
        if len(g) < MIN_CROSS_SECTION:
            continue
        x = g[x_col].rank()
        y = g[y_col].rank()
        t = g[target].rank()

        varx = np.var(x, ddof=0)
        if varx == 0:
            continue
        cov_xy = np.cov(x, y, ddof=0)[0, 1]
        beta = cov_xy / varx
        y_resid = y - beta * x

        ic, _ = spearmanr(y_resid, t)
        if np.isnan(ic):
            continue
        ics.append(ic)
    return float(np.nanmean(ics)) if ics else float("nan")


def pair_corr(df: pd.DataFrame, a: str, b: str) -> float:
    """Overall Spearman correlation between two features across all rows (fast sanity check)."""
    s = df[[a, b]].dropna()
    if s.empty:
        return float("nan")
    corr, _ = spearmanr(s[a], s[b])
    return float(corr)


def run_analysis(
    df: pd.DataFrame, groups: Dict[str, Iterable[str]], target: str = TARGET_COL
) -> List[FeatureResult]:
    results: List[FeatureResult] = []

    # Precompute marginal ICs
    marginals = {feat: marginal_ic(df, feat, target) for feats in groups.values() for feat in feats}

    for group, feats in groups.items():
        feats = list(feats)
        if not feats:
            continue
        # Choose base as highest marginal IC within group
        base_feature = max(feats, key=lambda f: marginals.get(f, float("-inf")))
        base_ic = marginals.get(base_feature, float("nan"))

        for feat in feats:
            corr = pair_corr(df, base_feature, feat) if feat != base_feature else 1.0
            inc_ic = (
                incremental_ic(df, base_feature, feat, target)
                if feat != base_feature
                else float("nan")
            )
            results.append(
                FeatureResult(
                    feature=feat,
                    group=group,
                    marginal_ic=marginals.get(feat, float("nan")),
                    base_feature=base_feature,
                    base_ic=base_ic,
                    corr_with_base=corr,
                    incremental_ic_vs_base=inc_ic,
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
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    if "week_date" not in df.index.names:
        raise ValueError("Expected week_date in index; ensure data is indexed by (symbol, week_date).")
    df = df.reset_index()  # easier grouping on week_date

    groups = default_groups()
    results = run_analysis(df, groups, target=TARGET_COL)
    save_summary(results, args.out)


if __name__ == "__main__":
    main()
