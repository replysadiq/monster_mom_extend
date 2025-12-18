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
        "--signal",
        type=str,
        default="rank_trend_4w",
        choices=["rank_trend_4w", "rank_mr_1w", "score_trend_4w", "score_mr_1w"],
        help="Column to rank on (higher is better).",
    )
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
    if args.signal not in comps.columns:
        raise RuntimeError(f"Signal '{args.signal}' not found. Available: {comps.columns.tolist()}")

    latest_week = comps.index.get_level_values("week_date").max()
    latest = comps.xs(latest_week, level="week_date").reset_index()
    latest = latest.dropna(subset=[args.signal])
    latest["rank"] = latest[args.signal]
    latest = latest.sort_values(args.signal, ascending=False).head(args.topn)
    latest.insert(0, "week_date", pd.to_datetime(latest_week).date())
    latest["signal"] = args.signal

    args.out.parent.mkdir(parents=True, exist_ok=True)
    latest[["week_date", "symbol", "rank", "signal"]].to_csv(args.out, index=False)
    print(f"Wrote Top-{args.topn} for {latest_week.date()} to {args.out}")


if __name__ == "__main__":
    main()
