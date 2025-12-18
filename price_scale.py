from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def choose_scale_factor(median_close: float) -> float:
    """Choose a per-symbol price scale to bring median close into a plausible INR band."""
    if not np.isfinite(median_close) or median_close <= 0:
        return 1.0
    candidates = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    lo, hi = 1.0, 500000.0
    target = 1000.0
    best = candidates[-1]
    best_err = float("inf")
    for f in candidates:
        m = median_close / f
        if lo <= m <= hi:
            err = abs(m - target)
            if err < best_err:
                best_err = err
                best = f
    return best


def apply_price_scale(
    df: pd.DataFrame, symbol_col: str = "symbol", scale_override: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply per-symbol scaling to OHLC columns and return scaled df plus scale series."""
    required_cols = {"open", "high", "low", "close", symbol_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"apply_price_scale missing columns: {missing}")

    if scale_override is not None:
        scale = scale_override.rename("scale")
    else:
        med_close = df.groupby(symbol_col)["close"].median()
        scale = med_close.apply(choose_scale_factor).rename("scale")

    scaled = df.merge(scale, left_on=symbol_col, right_index=True, how="left")
    if scaled["scale"].isna().any():
        missing_syms = scaled.loc[scaled["scale"].isna(), symbol_col].dropna().unique()[:20]
        raise RuntimeError(f"Missing scale for symbols. Example: {missing_syms}")

    for col in ["open", "high", "low", "close"]:
        scaled[col] = scaled[col] / scaled["scale"]

    med_after = scaled.groupby(symbol_col)["close"].median()
    bad = med_after[(med_after < 1) | (med_after > 500000)]
    if len(bad):
        raise RuntimeError(f"OHLC still unscaled for {len(bad)} symbols. Example:\n{bad.head()}")

    scaled = scaled.drop(columns=["scale"])
    return scaled, scale
