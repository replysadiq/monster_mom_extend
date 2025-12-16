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
    ap.add_argument("--gate-only", type=str, default="", help="Comma separated gate-only features.")
    ap.add_argument("--base-momentum", type=str, default=None, help="Base momentum feature for gate lift.")
    ap.add_argument("--weak-ic-threshold", type=float, default=0.01)
    ap.add_argument("--min-weeks", type=int, default=50)
    ap.add_argument("--out-gate-lift", type=Path, default=None, help="Gate lift output CSV (optional).")
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
    gate_only = [g for g in args.gate_only.split(",") if g] if args.gate_only else []
    weak_thr = args.weak_ic_threshold
    min_weeks = args.min_weeks
    status_rows = []

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
        expected = directions.get(feat, 1)
        if len(arr) < min_weeks:
            suggested = 0
            status = "INSUFFICIENT"
            sign_match = True
        elif abs(med) < weak_thr:
            suggested = 0
            status = "UNRESOLVED"
            sign_match = True
        elif med > 0 and pct_pos >= 0.55:
            suggested = 1
            status = "PASS" if expected == 1 else "FLIP"
            sign_match = expected == 1
        elif med < 0 and pct_neg >= 0.55:
            suggested = -1
            status = "PASS" if expected == -1 else "FLIP"
            sign_match = expected == -1
        else:
            suggested = 0
            status = "UNRESOLVED"
            sign_match = True
        status_rows.append(
            {
                "feature": feat,
                "group": groups.get(feat, ""),
                "expected_sign": expected,
                "suggested_sign": suggested,
                "sign_match": sign_match,
                "status": status,
                "median_corr": med,
                "iqr_corr": iqr,
                "pct_pos": pct_pos,
                "pct_neg": pct_neg,
                "n_weeks_used": len(arr),
            }
        )

    out_df = pd.DataFrame(status_rows)
    out_df = out_df.sort_values(["group", "median_corr"].copy(), ascending=[True, False])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    flips = out_df[out_df["status"] == "FLIP"]
    print(f"FLIP count: {len(flips)}")
    if not flips.empty:
        print(flips.head(10).to_string(index=False))

    # Gate-only lift
    if gate_only:
        if args.base_momentum:
            base_feat = args.base_momentum
        else:
            base_feat = "ret_8w" if "1w" in args.target_col else "ret_3m"
        lift_rows = []
        for gate in gate_only:
            if gate not in df.columns or base_feat not in df.columns:
                continue
            weekly_data = []
            for _, g in df[["week_date", gate, base_feat, args.target_col]].dropna().groupby("week_date"):
                if len(g) < 50:
                    continue
                base_ic, _ = spearmanr(g[base_feat].rank(pct=True), g[args.target_col])
                mask = g[gate].rank(pct=True) >= 0.55
                g_gated = g[mask]
                if len(g_gated) < 50:
                    continue
                gated_ic, _ = spearmanr(g_gated[base_feat].rank(pct=True), g_gated[args.target_col])
                if np.isnan(base_ic) or np.isnan(gated_ic):
                    continue
                weekly_data.append((base_ic, gated_ic))
            if not weekly_data:
                continue
            base_ics, gated_ics = zip(*weekly_data)
            base_med = float(np.median(base_ics))
            gated_med = float(np.median(gated_ics))
            lift_rows.append(
                {
                    "gate_feature": gate,
                    "base_momentum": base_feat,
                    "target_col": args.target_col,
                    "median_base_ic": base_med,
                    "median_gated_ic": gated_med,
                    "lift": gated_med - base_med,
                    "n_weeks_used": len(weekly_data),
                }
            )
        if lift_rows:
            lift_df = pd.DataFrame(lift_rows).sort_values("lift", ascending=False)
            out_gate = args.out_gate_lift
            if out_gate is None:
                out_gate = Path(str(args.out).replace(".csv", "_gate_lift.csv"))
            out_gate.parent.mkdir(parents=True, exist_ok=True)
            lift_df.to_csv(out_gate, index=False)
            print("Top gate lifts:")
            print(lift_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
