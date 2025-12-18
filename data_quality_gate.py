from __future__ import annotations

import numpy as np
import pandas as pd


def add_close_quality_flags(
    df: pd.DataFrame,
    close_col: str = "close",
    symbol_col: str = "symbol",
    date_col: str = "week_date",
    roll_window: int = 26,
    max_log_jump: float = 0.7,
    max_level_ratio: float = 50.0,
) -> pd.DataFrame:
    """Add deterministic quality flags for close-based weekly datasets."""
    x = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df.copy()
    x = x.sort_values([symbol_col, date_col])

    close = x[close_col].astype("float64")
    finite = np.isfinite(close.to_numpy())
    positive = close.to_numpy() > 0

    x["close_prev"] = x.groupby(symbol_col)[close_col].shift(1)
    prev = x["close_prev"].astype("float64")

    valid_prev = np.isfinite(prev.to_numpy()) & (prev.to_numpy() > 0) & finite & positive
    logret = np.full(len(x), np.nan, dtype="float64")
    logret[valid_prev] = np.log(close.to_numpy()[valid_prev] / prev.to_numpy()[valid_prev])
    x["logret_1w_close"] = logret

    x["roll_med_close"] = (
        x.groupby(symbol_col)[close_col]
        .transform(lambda s: s.shift(1).rolling(roll_window, min_periods=max(10, roll_window // 3)).median())
    )
    med = x["roll_med_close"].astype("float64")
    valid_med = np.isfinite(med.to_numpy()) & (med.to_numpy() > 0) & finite & positive

    ratio = np.full(len(x), np.nan, dtype="float64")
    ratio[valid_med] = close.to_numpy()[valid_med] / med.to_numpy()[valid_med]
    x["ratio_to_roll_med"] = ratio

    reasons = np.array([""] * len(x), dtype=object)
    bad = np.zeros(len(x), dtype=bool)

    bad_1 = (~finite) | (~positive)
    bad |= bad_1
    reasons[bad_1] = "non_finite_or_non_positive_close"

    bad_2 = valid_prev & (np.abs(logret) > max_log_jump)
    bad |= bad_2 & (~bad_1)
    reasons[bad_2 & (~bad_1)] = "excessive_log_jump_vs_prev_close"

    bad_3 = valid_med & ((ratio > max_level_ratio) | (ratio < (1.0 / max_level_ratio)))
    bad |= bad_3 & (~bad_1) & (~bad_2)
    reasons[bad_3 & (~bad_1) & (~bad_2)] = "close_level_outlier_vs_rolling_median"

    x["is_bad_close"] = bad
    x["bad_reason"] = reasons

    if isinstance(df.index, pd.MultiIndex):
        x = x.set_index(df.index.names)

    return x


def quarantine_report(df_flagged: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c
        for c in [
            "symbol",
            "week_date",
            "close",
            "close_prev",
            "roll_med_close",
            "logret_1w_close",
            "ratio_to_roll_med",
            "bad_reason",
        ]
        if c in df_flagged.columns
    ]
    out = df_flagged.reset_index()
    return out.loc[out["is_bad_close"], cols].sort_values(["week_date", "symbol"])
