from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_selected_features(path: Path) -> Dict[str, int]:
    """
    Expected JSON format: [{"feature": "ret_1w", "sign": -1}, ...]
    Returns dict: {feature: sign in {-1,+1}}
    """
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


def _as_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        if {"symbol", "week_date"}.issubset(df.columns):
            df = df.set_index(["symbol", "week_date"])
        else:
            raise RuntimeError("Features must have MultiIndex or columns ['symbol','week_date'].")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["symbol", "week_date"])
    return df.sort_index()


def zscore_weekly(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score within each week_date."""
    def _z(s: pd.Series) -> pd.Series:
        mu = s.mean()
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    return series.groupby(level="week_date", group_keys=False).apply(_z)


def build_composite(
    feats: pd.DataFrame,
    selected: Dict[str, int],
    min_feat: int = 2,
) -> pd.Series:
    """
    Apply per-week z-score to each selected feature, apply sign, then average.
    If non-null feature count < min_feat at a row, composite = NaN.
    """
    cols = [c for c in selected.keys() if c in feats.columns]
    if not cols:
        return pd.Series(dtype=float)

    zcols = []
    for c in cols:
        s = feats[c].astype(float)
        z = zscore_weekly(s)
        if selected[c] == -1:
            z = -z
        zcols.append(z.rename(c))

    zdf = pd.concat(zcols, axis=1)
    cnt = zdf.notna().sum(axis=1)
    comp = zdf.mean(axis=1)
    comp[cnt < min_feat] = np.nan
    comp.name = "composite"
    return comp


def rank_weekly(score: pd.Series) -> pd.Series:
    """
    Rank within each week_date as percentile (0..1).
    Deterministic tie handling: average.
    """
    def _r(s: pd.Series) -> pd.Series:
        return s.rank(pct=True, method="average")

    return score.groupby(level="week_date", group_keys=False).apply(_r)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build frozen composite scores from selected features (freeze-grade).")
    ap.add_argument("--features", type=Path, required=True, help="data/weekly_features_yahoo.parquet")
    ap.add_argument("--mr-selected", type=Path, required=True, help="results/.../selected_features_mr_1w.json")
    ap.add_argument("--trend-selected", type=Path, required=True, help="results/.../selected_features_trend_4w.json")
    ap.add_argument("--out", type=Path, default=Path("data/frozen/composites_selected_v1.parquet"))
    ap.add_argument("--manifest", type=Path, default=Path("data/frozen/composites_selected_v1_manifest.json"))
    ap.add_argument("--min-feat", type=int, default=2)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    feats = _as_multiindex(pd.read_parquet(args.features))

    mr_sel = load_selected_features(args.mr_selected)
    tr_sel = load_selected_features(args.trend_selected)

    # Hard fail if anything missing (freeze integrity)
    assert_selected_present(feats, mr_sel, "MR(1W)")
    assert_selected_present(feats, tr_sel, "Trend(4W)")

    # Build composites
    score_mr = build_composite(feats, mr_sel, min_feat=args.min_feat).rename("score_mr_1w")
    score_tr = build_composite(feats, tr_sel, min_feat=args.min_feat).rename("score_trend_4w")

    # Weekly rank percentiles
    rank_mr = rank_weekly(score_mr).rename("rank_mr_1w")
    rank_tr = rank_weekly(score_tr).rename("rank_trend_4w")

    out_df = pd.concat([score_mr, score_tr, rank_mr, rank_tr], axis=1).sort_index()

    # Persist
    out_df.to_parquet(args.out, index=True)

    # Manifest (inputs + selection spec + hashes)
    manifest = {
        "outputs": {
            "composites_parquet": str(args.out),
        },
        "inputs": {
            "features": str(args.features),
            "mr_selected": str(args.mr_selected),
            "trend_selected": str(args.trend_selected),
        },
        "sha256": {
            "features": _sha256_file(args.features) if args.features.exists() else None,
            "mr_selected": _sha256_file(args.mr_selected) if args.mr_selected.exists() else None,
            "trend_selected": _sha256_file(args.trend_selected) if args.trend_selected.exists() else None,
        },
        "params": {
            "min_feat": int(args.min_feat),
            "rank_method": "pct=True, method=average",
            "zscore": "cross-sectional per week_date, ddof=0",
        },
        "selected": {
            "mr_1w": [{"feature": k, "sign": int(v)} for k, v in mr_sel.items()],
            "trend_4w": [{"feature": k, "sign": int(v)} for k, v in tr_sel.items()],
        },
        "shape": {
            "rows": int(out_df.shape[0]),
            "cols": int(out_df.shape[1]),
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote: {args.out}")
    print(f"Wrote: {args.manifest}")
    print(f"Shape: {out_df.shape}")


if __name__ == "__main__":
    main()
