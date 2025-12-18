from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------
# Utilities
# -------------------------
def _as_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        if {"symbol", "week_date"}.issubset(df.columns):
            df = df.set_index(["symbol", "week_date"])
        else:
            raise RuntimeError("Features must have MultiIndex or columns ['symbol','week_date'].")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["symbol", "week_date"])
    return df.sort_index()


def _zscore_weekly(series: pd.Series) -> pd.Series:
    def _z(s: pd.Series) -> pd.Series:
        mu = s.mean()
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    return series.groupby(level="week_date", group_keys=False).apply(_z)


def _spearman_ic_by_week(x: pd.Series, y: pd.Series, min_n: int = 50) -> pd.DataFrame:
    weeks = sorted(
        set(x.dropna().index.get_level_values("week_date"))
        & set(y.dropna().index.get_level_values("week_date"))
    )
    rows = []
    for wk in weeks:
        xs = x.xs(wk, level="week_date", drop_level=False)
        ys = y.xs(wk, level="week_date", drop_level=False)
        m = pd.concat([xs.rename("x"), ys.rename("y")], axis=1).dropna()
        if len(m) < min_n:
            continue
        ic, _ = spearmanr(m["x"], m["y"])
        if np.isnan(ic):
            continue
        rows.append({"week_date": wk, "ic": float(ic), "n": int(len(m))})
    return pd.DataFrame(rows)


def _ic_summary(ts: pd.DataFrame) -> Dict[str, float]:
    if ts.empty:
        return {"ic_mean": np.nan, "ic_std": np.nan, "tstat": np.nan, "pos_frac": np.nan, "n_weeks": 0}
    arr = ts["ic"].to_numpy()
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    tstat = float(mean / (std / np.sqrt(n))) if std > 0 else np.nan
    return {
        "ic_mean": mean,
        "ic_std": std,
        "tstat": tstat,
        "pos_frac": float((arr > 0).mean()),
        "n_weeks": int(n),
    }


def _pairwise_corr_fast(df: pd.DataFrame) -> pd.DataFrame:
    # df: rows = observations, cols = features
    return df.corr(method="spearman", min_periods=max(50, int(0.2 * len(df))))


def _dedupe_by_corr(
    feats: List[str],
    rank: pd.DataFrame,
    corr: pd.DataFrame,
    corr_thresh: float,
) -> List[str]:
    """
    Greedy keep-best: iterate features by rank order (best first),
    discard any feature highly correlated with an already-kept feature.
    """
    kept: List[str] = []
    for f in feats:
        ok = True
        for k in kept:
            c = corr.at[f, k]
            if np.isfinite(c) and abs(c) >= corr_thresh:
                ok = False
                break
        if ok:
            kept.append(f)
    return kept


# -------------------------
# Core
# -------------------------
def compute_targets_from_weekly_close(
    wk_close: pd.Series,
    idx_weekly_close: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """
    Builds the same targets as v2:
      - target_1w_excess = fwd1 - idx_fwd1
      - target_4w_excess = fwd4 - idx_fwd4
    wk_close: MultiIndex Series ['symbol','week_date'] -> close
    idx_weekly_close: Series indexed by week_date -> index_close
    """
    wk_close = wk_close.sort_index()
    idx_weekly_close = idx_weekly_close.sort_index()

    fwd1 = wk_close.groupby(level=0).shift(-1) / wk_close - 1.0
    fwd4 = wk_close.groupby(level=0).shift(-4) / wk_close - 1.0

    idx_fwd1 = idx_weekly_close.shift(-1) / idx_weekly_close - 1.0
    idx_fwd4 = idx_weekly_close.shift(-4) / idx_weekly_close - 1.0

    week = wk_close.index.get_level_values("week_date")
    idx1_b = pd.Series(week.map(idx_fwd1).to_numpy(), index=wk_close.index)
    idx4_b = pd.Series(week.map(idx_fwd4).to_numpy(), index=wk_close.index)

    target1 = (fwd1 - idx1_b).rename("target_1w_excess")
    target4 = (fwd4 - idx4_b).rename("target_4w_excess")
    return target1, target4


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Feature selection on weekly features using IC + stability + redundancy pruning.")
    ap.add_argument("--features", type=Path, required=True, help="weekly_features_yahoo.parquet")
    ap.add_argument("--index-weekly", type=Path, required=True, help="index_weekly_close.csv from freeze run")
    ap.add_argument("--out-dir", type=Path, default=Path("results/feature_select_v1"))
    ap.add_argument("--min-coverage", type=float, default=0.80, help="min non-null fraction per feature")
    ap.add_argument("--min-weeks", type=int, default=80, help="min IC weeks to keep a feature")
    ap.add_argument("--min-pos-frac", type=float, default=0.55, help="min fraction of weeks IC>0 after sign-normalization")
    ap.add_argument("--topk-pre", type=int, default=20, help="take top-K by |tstat| before corr pruning")
    ap.add_argument("--corr-thresh", type=float, default=0.85, help="absolute spearman corr threshold to remove redundancy")
    ap.add_argument("--min-n-week", type=int, default=50, help="min symbols per week for IC computation")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feats = _as_multiindex(pd.read_parquet(args.features))

    # Load frozen weekly index close
    idxw = pd.read_csv(args.index_weekly)
    if not {"week_date", "index_close"}.issubset(idxw.columns):
        raise RuntimeError("index_weekly_close.csv must have columns: week_date,index_close")
    idxw["week_date"] = pd.to_datetime(idxw["week_date"])
    idx_close = idxw.set_index("week_date")["index_close"].astype(float).sort_index()

    # Weekly stock close from features (already weekly-aligned)
    if "close" not in feats.columns:
        raise RuntimeError("Features parquet must include 'close' column.")
    wk_close = feats["close"].astype(float).rename("close")

    # Align cutoff to avoid NaNs from forward shifts
    max_week = min(wk_close.index.get_level_values("week_date").max(), idx_close.index.max())
    cutoff = max_week - pd.Timedelta(weeks=4)
    wk_close = wk_close[wk_close.index.get_level_values("week_date") <= cutoff]
    feats = feats[feats.index.get_level_values("week_date") <= cutoff]
    idx_close = idx_close[idx_close.index <= cutoff]

    # Targets
    target1, target4 = compute_targets_from_weekly_close(wk_close, idx_close)

    # Candidate feature columns: drop raw OHLCV-ish cols + anything obviously not a feature
    drop_cols = {"close", "adj_close", "Dividends", "Stock Splits"}
    candidate_cols = [c for c in feats.columns if c not in drop_cols]

    # Coverage filter
    total_rows = len(feats)
    cov = feats[candidate_cols].notna().sum(axis=0) / max(1, total_rows)
    candidate_cols = [c for c in candidate_cols if cov.loc[c] >= args.min_coverage]

    # Build report for each target horizon
    def run_selection(target: pd.Series, tag: str) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
        rows = []
        signed_cols: Dict[str, int] = {}  # +1 or -1
        for col in candidate_cols:
            x = feats[col].astype(float)

            # z-score within week (cross-sectional standardization)
            xz = _zscore_weekly(x)

            # IC timeseries
            ts = _spearman_ic_by_week(xz, target, min_n=args.min_n_week)
            summ = _ic_summary(ts)

            if summ["n_weeks"] < args.min_weeks or not np.isfinite(summ["tstat"]):
                continue

            # Decide sign so “higher score = better future excess return”
            # If ic_mean is negative, flip the feature (sign=-1)
            sign = -1 if summ["ic_mean"] < 0 else 1
            signed_cols[col] = sign

            # recompute pos_frac on sign-adjusted ICs (so stability is about consistency, not direction)
            ts_adj = ts.copy()
            ts_adj["ic_adj"] = ts_adj["ic"] * sign
            pos_frac_adj = float((ts_adj["ic_adj"] > 0).mean())

            rows.append(
                {
                    "feature": col,
                    "coverage": float(cov.loc[col]),
                    "ic_mean": float(summ["ic_mean"]),
                    "tstat": float(summ["tstat"]),
                    "pos_frac_adj": float(pos_frac_adj),
                    "n_weeks": int(summ["n_weeks"]),
                    "sign": int(sign),
                }
            )

        rep = pd.DataFrame(rows)
        if rep.empty:
            return rep, [], signed_cols

        # Stability gate
        rep = rep[rep["pos_frac_adj"] >= args.min_pos_frac].copy()

        # Rank by |tstat|
        rep["abs_tstat"] = rep["tstat"].abs()
        rep = rep.sort_values(["abs_tstat", "pos_frac_adj", "coverage"], ascending=False)

        # Preselect top-K
        pre = rep.head(args.topk_pre)["feature"].tolist()
        if len(pre) <= 1:
            return rep, pre, signed_cols

        # Redundancy pruning: compute Spearman corr on a pooled sample of rows
        sample = feats[pre].astype(float)
        # to control size, sample at most 20000 rows (symbol-weeks)
        if len(sample) > 20000:
            sample = sample.sample(20000, random_state=7)
        corr = _pairwise_corr_fast(sample)

        kept = _dedupe_by_corr(pre, rep.set_index("feature"), corr, args.corr_thresh)
        return rep, kept, signed_cols

    rep_mr, sel_mr, signs = run_selection(target1, "mr_1w")
    rep_tr, sel_tr, signs2 = run_selection(target4, "trend_4w")
    signs.update(signs2)

    # Write outputs
    rep_mr.to_csv(args.out_dir / "feature_report_mr_1w.csv", index=False)
    rep_tr.to_csv(args.out_dir / "feature_report_trend_4w.csv", index=False)

    def write_json(path: Path, features: List[str]) -> None:
        payload = [{"feature": f, "sign": int(signs.get(f, 1))} for f in features]
        with path.open("w") as fp:
            json.dump(payload, fp, indent=2)

    write_json(args.out_dir / "selected_features_mr_1w.json", sel_mr)
    write_json(args.out_dir / "selected_features_trend_4w.json", sel_tr)

    print("Selected MR(1W):", sel_mr)
    print("Selected Trend(4W):", sel_tr)
    print("Wrote:", args.out_dir)

    manifest = {
        "params": {
            "min_coverage": args.min_coverage,
            "min_weeks": args.min_weeks,
            "min_pos_frac": args.min_pos_frac,
            "topk_pre": args.topk_pre,
            "corr_thresh": args.corr_thresh,
            "min_n_week": args.min_n_week,
        },
        "inputs": {
            "features": str(args.features),
            "index_weekly": str(args.index_weekly),
            "sha256": {
                "features": _sha256_file(args.features) if args.features.exists() else None,
                "index_weekly": _sha256_file(args.index_weekly) if args.index_weekly.exists() else None,
            },
        },
        "selected": {
            "mr_1w": sel_mr,
            "trend_4w": sel_tr,
        },
    }
    (args.out_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
