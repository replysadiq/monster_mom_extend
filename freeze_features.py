"""
Freeze a minimal, non-redundant feature set based on incremental IC summary.

Rules are deterministic and group-aware:
- Eligibility uses marginal stability and coverage thresholds.
- Base feature per group = strongest stable marginal.
- Add-ons must be non-redundant or show strong incremental t-stat.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Defaults
DEFAULT_SUMMARY = Path("data/incremental_ic_summary.csv")
DEFAULT_OUT_FEATURES = Path("data/frozen_features.txt")
DEFAULT_OUT_REPORT = Path("data/frozen_features_report.csv")


@dataclass
class Decision:
    feature: str
    group: str
    role: str  # base | addon | dropped
    signal_direction: str  # positive | negative
    marginal_ic: float
    marginal_t: float
    marginal_hit_rate: float
    marginal_n_weeks: int
    corr_with_base: float
    incremental_ic_vs_base: float
    incremental_t: float
    incremental_hit_rate: float
    incremental_n_weeks: int
    decision_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze minimal non-redundant feature set.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out_features", type=Path, default=DEFAULT_OUT_FEATURES)
    parser.add_argument("--out_report", type=Path, default=DEFAULT_OUT_REPORT)
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD), optional.")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD), optional.")
    parser.add_argument("--min_weeks_ratio", type=float, default=0.70)
    parser.add_argument("--min_abs_marginal_t", type=float, default=2.0)
    parser.add_argument("--min_hit_rate", type=float, default=0.55)
    parser.add_argument("--redundancy_corr", type=float, default=0.80)
    parser.add_argument("--min_abs_incremental_t", type=float, default=2.0)
    parser.add_argument("--max_features_per_group", type=int, default=2)
    parser.add_argument(
        "--allow_negative",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Keep negative-signed signals (tagged) if True.",
    )
    return parser.parse_args()


def is_eligible(
    row: pd.Series,
    min_weeks: int,
    min_abs_marginal_t: float,
    min_hit_rate: float,
    allow_negative: bool,
) -> bool:
    if not np.isfinite(row["marginal_ic"]):
        return False
    if row["marginal_n_weeks"] < min_weeks:
        return False
    if (abs(row["marginal_t"]) < min_abs_marginal_t) and (row["marginal_hit_rate"] < min_hit_rate):
        return False
    if (not allow_negative) and (row["marginal_ic"] <= 0):
        return False
    return True


def select_features(
    df: pd.DataFrame,
    params: argparse.Namespace,
) -> List[Decision]:
    decisions: List[Decision] = []
    selected_features: Dict[str, List[str]] = {}

    for group, gdf in df.groupby("group"):
        gdf = gdf.copy()
        group_decisions: Dict[str, Decision] = {}

        base_weeks = gdf["base_n_weeks"].max()
        min_weeks = math.ceil(params.min_weeks_ratio * base_weeks)

        # Eligibility check
        eligible_mask = gdf.apply(
            lambda r: is_eligible(
                r, min_weeks, params.min_abs_marginal_t, params.min_hit_rate, params.allow_negative
            ),
            axis=1,
        )
        eligible = gdf[eligible_mask]

        # Default all as dropped (weak/unstable) if not eligible
        for _, row in gdf.iterrows():
            group_decisions[row["feature"]] = Decision(
                feature=row["feature"],
                group=group,
                role="dropped",
                signal_direction="positive" if row["marginal_ic"] > 0 else "negative",
                marginal_ic=row["marginal_ic"],
                marginal_t=row["marginal_t"],
                marginal_hit_rate=row["marginal_hit_rate"],
                marginal_n_weeks=int(row["marginal_n_weeks"]),
                corr_with_base=row["corr_with_base"],
                incremental_ic_vs_base=row["incremental_ic_vs_base"],
                incremental_t=row["incremental_t"],
                incremental_hit_rate=row["incremental_hit_rate"],
                incremental_n_weeks=int(row["incremental_n_weeks"]),
                decision_reason="dropped: weak/unstable marginal",
            )

        if eligible.empty:
            decisions.extend(group_decisions.values())
            continue

        # Choose base: highest |marginal_t|, tie-breaker |marginal_ic|
        eligible_sorted = eligible.copy()
        eligible_sorted["abs_mt"] = eligible_sorted["marginal_t"].abs()
        eligible_sorted["abs_mic"] = eligible_sorted["marginal_ic"].abs()
        base_row = (
            eligible_sorted.sort_values(
                ["abs_mt", "abs_mic", "feature"], ascending=[False, False, True]
            )
            .iloc[0]
            .to_dict()
        )
        base_feature = base_row["feature"]

        # Mark base
        base_decision = group_decisions[base_feature]
        base_decision.role = "base"
        base_decision.decision_reason = "base: strongest stable marginal"
        group_decisions[base_feature] = base_decision
        selected_features.setdefault(group, []).append(base_feature)

        # Process add-on candidates
        candidates = []
        for _, row in eligible.iterrows():
            feat = row["feature"]
            if feat == base_feature:
                continue
            corr = abs(row["corr_with_base"])
            inc_t = row["incremental_t"]
            inc_ic = row["incremental_ic_vs_base"]
            if corr < params.redundancy_corr:
                reason = "addon: low redundancy (corr<{:.2f})".format(params.redundancy_corr)
                candidates.append((feat, inc_t, inc_ic, reason))
            elif abs(inc_t) >= params.min_abs_incremental_t:
                reason = "addon: redundant but incremental_t>= {:.1f}".format(
                    params.min_abs_incremental_t
                )
                candidates.append((feat, inc_t, inc_ic, reason))
            else:
                dec = group_decisions[feat]
                dec.decision_reason = "dropped: redundant and no incremental"
                group_decisions[feat] = dec

        # Rank candidates and select
        max_addons = max(params.max_features_per_group - 1, 0)
        if max_addons > 0 and candidates:
            candidates_sorted = sorted(
                candidates,
                key=lambda x: (
                    -abs(x[1]) if np.isfinite(x[1]) else float("-inf"),
                    -abs(x[2]) if np.isfinite(x[2]) else float("-inf"),
                    x[0],
                ),
            )
            for feat, inc_t, inc_ic, reason in candidates_sorted[:max_addons]:
                dec = group_decisions[feat]
                dec.role = "addon"
                dec.decision_reason = reason
                group_decisions[feat] = dec
                selected_features[group].append(feat)
            # Remaining candidates beyond cap stay dropped with existing reason or add explicit cap reason
            for feat, _, _, _ in candidates_sorted[max_addons:]:
                dec = group_decisions[feat]
                if "dropped" in dec.decision_reason:
                    continue
                dec.decision_reason = "dropped: max_features_per_group reached"
                group_decisions[feat] = dec

        decisions.extend(group_decisions.values())

    return decisions


def write_outputs(decisions: List[Decision], out_features: Path, out_report: Path) -> None:
    # Selected features in order: group then base/addon ordering
    selected = [d.feature for d in decisions if d.role in {"base", "addon"}]
    out_features.parent.mkdir(parents=True, exist_ok=True)
    with out_features.open("w") as f:
        for feat in selected:
            f.write(f"{feat}\n")

    df = pd.DataFrame([d.__dict__ for d in decisions])
    df.sort_values(["group", "role", "feature"], inplace=True)
    df.to_csv(out_report, index=False)
    print(f"Wrote selected features to {out_features}")
    print(f"Wrote report to {out_report}")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary)
    # Override outputs with period suffix if start/end provided
    out_features = args.out_features
    out_report = args.out_report
    if args.start and args.end:
        out_features = Path(f"data/frozen_features_{args.start}_{args.end}.txt")
        out_report = Path(f"data/frozen_features_report_{args.start}_{args.end}.csv")

    required_cols = {
        "group",
        "feature",
        "marginal_ic",
        "marginal_t",
        "marginal_hit_rate",
        "marginal_n_weeks",
        "base_n_weeks",
        "corr_with_base",
        "incremental_ic_vs_base",
        "incremental_t",
        "incremental_hit_rate",
        "incremental_n_weeks",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Summary file missing required columns: {missing}")

    decisions = select_features(df, args)
    write_outputs(decisions, out_features, out_report)


if __name__ == "__main__":
    main()
