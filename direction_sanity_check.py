import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Direction sanity check for frozen features.")
    ap.add_argument("--features", type=Path, required=True, help="Weekly parquet path.")
    ap.add_argument("--manifest", type=Path, default=Path("feature_groups.yaml"))
    ap.add_argument("--frozen", type=Path, required=True, help="Frozen features txt.")
    ap.add_argument(
        "--target-col",
        type=str,
        required=True,
        choices=["target_forward_1w_excess", "target_forward_4w_excess"],
    )
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Output CSV.")
    return ap.parse_args()


def load_manifest(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    directions = {}
    groups = {}
    for entry in data.get("groups", []):
        directions[entry["feature"]] = int(entry.get("direction", 1))
        groups[entry["feature"]] = entry.get("group", "")
    return directions, groups


def load_frozen(path: Path) -> list:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.features)
    if "week_date" not in df.index.names:
        raise ValueError("Expected index to include week_date")
    df = df.reset_index()
    df = df[(df["week_date"] >= args.start) & (df["week_date"] <= args.end)]

    directions, groups = load_manifest(args.manifest)
    frozen = load_frozen(args.frozen)

    rows = []
    for feat in frozen:
        if feat not in df.columns:
            continue
        weekly_corrs = []
        for _, g in df[["week_date", feat, args.target_col]].dropna().groupby("week_date"):
            if len(g) < 50:
                continue
            ic, _ = spearmanr(g[feat].rank(pct=True), g[args.target_col])
            if np.isnan(ic):
                continue
            weekly_corrs.append(ic)
        if not weekly_corrs:
            continue
        arr = np.array(weekly_corrs)
        med = float(np.median(arr))
        iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        pct_pos = float((arr > 0).mean())
        pct_neg = float((arr < 0).mean())
        suggested = 0
        if med > 0 and pct_pos >= 0.55:
            suggested = 1
        elif med < 0 and pct_neg >= 0.55:
            suggested = -1
        expected = directions.get(feat, 1)
        rows.append(
            {
                "feature": feat,
                "group": groups.get(feat, ""),
                "expected_sign": expected,
                "suggested_sign": suggested,
                "sign_match": expected == suggested if suggested != 0 else False,
                "median_corr": med,
                "iqr_corr": iqr,
                "pct_pos": pct_pos,
                "pct_neg": pct_neg,
                "n_weeks_used": len(arr),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["group", "median_corr"].copy(), ascending=[True, False])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    mismatches = out_df[out_df["sign_match"] == False]
    print(f"Mismatches: {len(mismatches)}")
    if not mismatches.empty:
        print(mismatches.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
