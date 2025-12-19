"""Export the latest weekly Top-N signal from frozen composites."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export latest weekly Top-N from frozen composites.")
    ap.add_argument(
        "--composites",
        type=Path,
        default=Path("data/frozen/composites_selected_v1.parquet"),
        help="Frozen composites parquet (MultiIndex symbol, week_date).",
    )
    ap.add_argument(
        "--selection-mode",
        type=str,
        default="trend_confirmed_mr",
        choices=["trend_confirmed_mr"],
        help="Two-stage selection: shortlist by trend_4w then rank by mr_1w.",
    )
    ap.add_argument("--trend-shortlist-size", type=int, default=30, help="Top-M shortlist by trend_4w before MR ranking.")
    ap.add_argument("--topn", type=int, default=10, help="Number of symbols to export.")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/latest_weekly_signal.csv"),
        help="Output CSV path.",
    )
    return ap.parse_args()


def _as_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        if {"symbol", "week_date"}.issubset(df.columns):
            df = df.set_index(["symbol", "week_date"])
        else:
            raise RuntimeError("Composites must have MultiIndex or columns ['symbol','week_date'].")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["symbol", "week_date"])
    return df.sort_index()


def main() -> None:
    args = parse_args()
    comps = _as_multiindex(pd.read_parquet(args.composites))
    latest_week = comps.index.get_level_values("week_date").max()
    latest = comps.xs(latest_week, level="week_date").reset_index()
    latest = latest.dropna(subset=["score_trend_4w", "score_mr_1w"], how="all")

    # Two-stage: shortlist by trend_4w then rank by mr_1w
    if not {"score_trend_4w", "score_mr_1w"}.issubset(latest.columns):
        raise RuntimeError("Composites must contain score_trend_4w and score_mr_1w for export.")

    trend_ranked = latest.sort_values("score_trend_4w", ascending=False)
    shortlist = trend_ranked.head(args.trend_shortlist_size)
    shortlist = shortlist.sort_values("score_mr_1w", ascending=False)
    shortlist["rank"] = shortlist["score_mr_1w"]

    shortlist = shortlist.dropna(subset=["rank"]).head(args.topn)
    shortlist.insert(0, "week_date", pd.to_datetime(latest_week).date())
    shortlist["signal"] = "trend_confirmed_mr"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shortlist[["week_date", "symbol", "rank", "signal"]].to_csv(args.out, index=False)
    print(f"Wrote Top-{args.topn} (trend-confirmed MR) for {latest_week.date()} to {args.out}")


if __name__ == "__main__":
    main()
