"""
CP-based redundancy pruning via LOFO conformal interval widths.

Workflow:
- Load weekly feature panel (optionally date-filtered).
- Load feature manifest (groups, tags, directions).
- Time-safe rolling windows: train -> calibrate -> test.
- Compute baseline PI width and LOFO deltas per feature (non-regime).
- Summarize median/IQR deltas, apply pruning rules within groups, enforce minima.
- Emit summary CSV and frozen feature list.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
try:
    from mapie.regression import SplitConformalRegressor
except ImportError as exc:
    raise SystemExit(
        "MAPIE is required for conformal intervals. Install with `pip install mapie`."
    ) from exc
from sklearn.ensemble import GradientBoostingRegressor


MIN_GROUP_KEEP = {"momentum": 2, "trend": 2, "volatility": 1, "liquidity": 1}


@dataclass
class FeatureMeta:
    feature: str
    group: str
    tag: str  # core | experimental
    direction: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CP-based LOFO feature pruning")
    ap.add_argument("--features", type=Path, required=True, help="Weekly features parquet path.")
    ap.add_argument("--target-col", type=str, default="target_forward_4w_excess")
    ap.add_argument("--manifest", type=Path, default=Path("feature_groups.yaml"))
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--train-weeks", type=int, default=156)
    ap.add_argument("--cal-weeks", type=int, default=26)
    ap.add_argument("--test-weeks", type=int, default=26)
    ap.add_argument("--alpha", type=float, default=0.1, help="Alpha for 1-alpha coverage (e.g., 0.1 -> 90%)")
    ap.add_argument("--out-summary", type=Path, default=Path("data/cp_delta_summary.csv"))
    ap.add_argument("--out-frozen", type=Path, default=Path("data/frozen_features_cp.txt"))
    ap.add_argument("--smoke", action="store_true", help="Smoke test on limited symbols/weeks.")
    return ap.parse_args()


def load_manifest(path: Path) -> Dict[str, FeatureMeta]:
    data = yaml.safe_load(path.read_text())
    metas = {}
    for entry in data.get("groups", []):
        metas[entry["feature"]] = FeatureMeta(
            feature=entry["feature"],
            group=entry["group"],
            tag=entry.get("tag", "experimental"),
            direction=int(entry.get("direction", 1)),
        )
    return metas


def filter_dates(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if start is not None:
        df = df[df.index.get_level_values("week_date") >= start]
    if end is not None:
        df = df[df.index.get_level_values("week_date") <= end]
    return df


def load_panel(path: Path, target_col: str, start: Optional[str], end: Optional[str], smoke: bool) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "week_date" not in df.index.names:
        raise ValueError("Expected index to include week_date")
    df = df.sort_index()
    s = pd.to_datetime(start) if start else None
    e = pd.to_datetime(end) if end else None
    df = filter_dates(df, s, e)
    df = df.replace([np.inf, -np.inf], np.nan)
    if smoke:
        symbols = df.index.get_level_values("symbol").unique()[:30]
        weeks = df.index.get_level_values("week_date").unique()[:120]
        df = df.loc[(df.index.get_level_values("symbol").isin(symbols)) & (df.index.get_level_values("week_date").isin(weeks))]
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} missing.")
    return df


def rolling_windows(dates: List[pd.Timestamp], train: int, cal: int, test: int) -> List[Tuple[List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]]]:
    windows = []
    n = len(dates)
    start = 0
    while start + train + cal + test <= n:
        train_dates = dates[start : start + train]
        cal_dates = dates[start + train : start + train + cal]
        test_dates = dates[start + train + cal : start + train + cal + test]
        if train_dates and cal_dates and test_dates and train_dates[-1] < cal_dates[0] < test_dates[0]:
            windows.append((train_dates, cal_dates, test_dates))
        start += test  # slide by test block
    return windows


def train_mapie(
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    target_col: str,
    alpha: float,
) -> float:
    X_train = train_df[features].values
    y_train = train_df[target_col].values
    X_cal = cal_df[features].values
    y_cal = cal_df[target_col].values
    X_test = test_df[features].values

    est = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    est.fit(X_train, y_train)
    confidence = 1 - alpha
    mapie = SplitConformalRegressor(
        estimator=est,
        confidence_level=confidence,
        prefit=True,
    )
    mapie.conformalize(X_cal, y_cal)
    lower, upper = mapie.predict_interval(X_test)
    widths = upper - lower
    return float(np.median(widths))


def compute_deltas(
    df: pd.DataFrame,
    manifest: Dict[str, FeatureMeta],
    target_col: str,
    alpha: float,
    train_weeks: int,
    cal_weeks: int,
    test_weeks: int,
) -> Dict[str, List[float]]:
    dates = sorted(df.index.get_level_values("week_date").unique())
    windows = rolling_windows(dates, train_weeks, cal_weeks, test_weeks)
    print(f"Window count: {len(windows)}")
    if not windows:
        raise ValueError("No windows constructed; check date range and window lengths.")
    deltas: Dict[str, List[float]] = {}

    for i, (train_dates, cal_dates, test_dates) in enumerate(windows):
        print(f"Window {i+1}: train {train_dates[0]}->{train_dates[-1]}, cal {cal_dates[0]}->{cal_dates[-1]}, test {test_dates[0]}->{test_dates[-1]}")
        feats_available = [
            f for f, meta in manifest.items() if meta.group != "regime" and f in df.columns and f != target_col
        ]
        if not feats_available:
            raise ValueError("No available non-regime features for LOFO.")
        window_df = df.loc[
            df.index.get_level_values("week_date").isin(train_dates + cal_dates + test_dates)
        ]
        window_df = window_df.dropna(subset=feats_available + [target_col])
        train_df = window_df.loc[window_df.index.get_level_values("week_date").isin(train_dates)]
        cal_df = window_df.loc[window_df.index.get_level_values("week_date").isin(cal_dates)]
        test_df = window_df.loc[window_df.index.get_level_values("week_date").isin(test_dates)]
        print(f" n_train={len(train_df)}, n_cal={len(cal_df)}, n_test={len(test_df)}")

        base_width = train_mapie(train_df, cal_df, test_df, feats_available, target_col, alpha)
        # initialize deltas dict lazily
        for feat in feats_available:
            deltas.setdefault(feat, [])
        # Debug: first 3 LOFOs
        for j, feat in enumerate(feats_available):
            subset = [f for f in feats_available if f != feat]
            w_minus = train_mapie(train_df, cal_df, test_df, subset, target_col, alpha)
            if base_width == 0:
                delta = np.nan
            else:
                delta = (w_minus - base_width) / base_width
            deltas[feat].append(delta)
            if j < 3:
                print(f"  LOFO {feat}: base_width={base_width:.6f}, w_minus={w_minus:.6f}, delta={delta:.6f}")
    return deltas


def summarize_deltas(
    deltas: Dict[str, List[float]],
    manifest: Dict[str, FeatureMeta],
    target_col: str,
) -> pd.DataFrame:
    rows = []
    for feat, values in deltas.items():
        if feat == target_col:
            continue
        if feat not in manifest:
            continue
        arr = np.array(values)
        if arr.size == 0:
            med = np.nan
            iqr = np.nan
        else:
            med = float(np.nanmedian(arr))
            iqr = float(np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25))
        meta = manifest[feat]
        rows.append(
            {
                "feature": feat,
                "group": meta.group,
                "core_or_experimental": meta.tag,
                "median_delta": med,
                "iqr_delta": iqr,
            }
        )
    return pd.DataFrame(rows)


def apply_pruning(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    decisions = []
    kept = []
    for group, g in df.groupby("group"):
        min_keep = MIN_GROUP_KEEP.get(group, 0)
        g_sorted = g.sort_values("median_delta", ascending=False)
        # Default decision
        g_sorted["keep_drop"] = "drop"
        g_sorted["notes"] = "rule: median<0.02 & IQR<0.01"
        # Mark drops based on rules
        for idx, row in g_sorted.iterrows():
            med = row["median_delta"]
            iqr = row["iqr_delta"]
            if pd.isna(med):
                reason = "drop: missing delta"
                g_sorted.at[idx, "notes"] = reason
                continue
            if med < 0:
                g_sorted.at[idx, "notes"] = "drop: negative median"
                continue
            if (med < 0.02) and (iqr < 0.01):
                g_sorted.at[idx, "notes"] = "drop: low median & low IQR"
                continue
            g_sorted.at[idx, "keep_drop"] = "keep"
            g_sorted.at[idx, "notes"] = "keep: passes delta rule"
        # Enforce minima
        keepers = g_sorted[g_sorted["keep_drop"] == "keep"]
        if len(keepers) < min_keep:
            shortfall = min_keep - len(keepers)
            candidates = g_sorted[g_sorted["keep_drop"] == "drop"].head(shortfall)
            g_sorted.loc[candidates.index, "keep_drop"] = "keep"
            g_sorted.loc[candidates.index, "notes"] = "keep: enforced group minimum"
        decisions.append(g_sorted)
        kept.extend(g_sorted[g_sorted["keep_drop"] == "keep"]["feature"].tolist())
    out = pd.concat(decisions, ignore_index=True)
    # Regime excluded earlier; kept already excludes them if not in manifest? handled upstream
    return out, kept


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    df = load_panel(args.features, args.target_col, args.start, args.end, args.smoke)

    non_regime_feats = [f for f, meta in manifest.items() if meta.group != "regime"]
    required_cols = non_regime_feats + [args.target_col]
    miss = [c for c in required_cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in panel: {miss}")
    # Drop rows with NaNs in required cols
    na_counts = df[required_cols].isna().sum()
    print("Missingness per required column:")
    print(na_counts.to_string())
    df = df.dropna(subset=required_cols)

    deltas = compute_deltas(
        df,
        manifest,
        target_col=args.target_col,
        alpha=args.alpha,
        train_weeks=args.train_weeks,
        cal_weeks=args.cal_weeks,
        test_weeks=args.test_weeks,
    )
    summary = summarize_deltas(deltas, manifest, args.target_col)
    # Exclude regime if present (none in manifest)
    summary = summary[summary["group"] != "regime"]
    summary_decided, kept = apply_pruning(summary)

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_decided.to_csv(args.out_summary, index=False)

    out_frozen = args.out_frozen
    if args.start and args.end:
        out_frozen = Path(f"data/frozen_features_cp_{args.start}_{args.end}.txt")
    out_frozen.parent.mkdir(parents=True, exist_ok=True)
    with out_frozen.open("w") as f:
        for feat in kept:
            f.write(f"{feat}\n")

    print(f"Saved summary to {args.out_summary}")
    print(f"Saved frozen features to {out_frozen}")


if __name__ == "__main__":
    main()
