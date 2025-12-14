import numpy as np
import pandas as pd
from pathlib import Path


DATA_DIR = Path("data")
OHLVC_PATH = DATA_DIR / "ohlcv.parquet"
INDEX_PATH = DATA_DIR / "index_nifty500.csv"
OUTPUT_PATH = DATA_DIR / "weekly_features.parquet"


def _group_roll(series: pd.Series, window: int, func: str, min_periods: int | None = None) -> pd.Series:
    """Group-wise rolling helper."""
    return (
        series.groupby(series.index.get_level_values("symbol"))
        .rolling(window, min_periods=min_periods or window)
        .agg(func)
        .reset_index(level=0, drop=True)
    )


def _group_ewm(series: pd.Series, span: int) -> pd.Series:
    """Group-wise exponential moving average."""
    return (
        series.groupby(series.index.get_level_values("symbol"))
        .transform(lambda x: x.ewm(span=span, adjust=False).mean())
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load stock and index data."""
    stocks = pd.read_parquet(OHLVC_PATH)
    stocks["date"] = pd.to_datetime(stocks["date"])
    stocks = stocks.sort_values(["symbol", "date"]).set_index(["symbol", "date"])

    index = pd.read_csv(INDEX_PATH)
    index["date"] = pd.to_datetime(index["date"])
    index = index.sort_values("date").set_index("date")
    index.rename(columns={"close": "index_close"}, inplace=True)
    return stocks, index


def compute_index_features(index: pd.DataFrame) -> pd.DataFrame:
    """Compute index-level helpers."""
    idx = index.copy()
    idx["index_ret_1d"] = idx["index_close"].pct_change()
    idx["index_ret_63d"] = idx["index_close"].pct_change(63)
    idx["index_sma_200d"] = idx["index_close"].rolling(200, min_periods=200).mean()
    idx["index_trend_state"] = (idx["index_close"] > idx["index_sma_200d"]).astype(float)
    idx["index_vol_20d"] = idx["index_ret_1d"].rolling(20, min_periods=20).std() * np.sqrt(252)
    idx["index_hv_percentile"] = (
        idx["index_vol_20d"]
        .rolling(252, min_periods=252)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )
    return idx


def compute_stock_features(stocks: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Engineer daily features for each symbol."""
    # Align index columns onto stock rows
    df = stocks.join(index, on="date")

    # Daily return
    df["ret_1d"] = df.groupby(level=0)["close"].pct_change()

    # Momentum returns
    horizons = {
        "ret_1w": 5,
        "ret_2w": 10,
        "ret_1m": 21,
        "ret_8w": 40,
        "ret_3m": 63,
        "ret_6m": 126,
        "ret_12m": 252,
    }
    for name, lag in horizons.items():
        df[name] = df.groupby(level=0)["close"].pct_change(lag)
    df["excess_ret_3m_vs_index"] = df["ret_3m"] - df["index_ret_63d"]

    # Moving averages and distances
    for window in (20, 50, 200):
        sma = _group_roll(df["close"], window, "mean")
        df[f"sma_{window}d"] = sma
        df[f"pct_above_{window}d_sma"] = df["close"] / sma - 1

    high_52w = _group_roll(df["high"], 252, "max")
    df["pct_from_52w_high"] = df["close"] / high_52w - 1

    # ATR and ADX
    df = df.sort_index()
    def _adx(group: pd.DataFrame, period: int = 14) -> pd.Series:
        high = group["high"]
        low = group["low"]
        close = group["close"]
        up = high.diff()
        down = -low.diff()
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=group.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=group.index)
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=period).mean()
        plus_di = 100 * plus_dm.rolling(period, min_periods=period).mean() / atr
        minus_di = 100 * minus_dm.rolling(period, min_periods=period).mean() / atr
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = (100 * dx).rolling(period, min_periods=period).mean()
        return adx

    adx_series = pd.Series(index=df.index, dtype=float)
    for sym, grp in df.groupby(level=0):
        adx_series.loc[grp.index] = _adx(grp)
    df["adx_14"] = adx_series.values

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df.groupby(level=0)["close"].shift()).abs(),
            (df["low"] - df.groupby(level=0)["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = _group_roll(tr, 14, "mean")
    df["atr_pct_14"] = atr14 / df["close"]

    # MACD
    ema12 = _group_ewm(df["close"], 12)
    ema26 = _group_ewm(df["close"], 26)
    macd = ema12 - ema26
    signal = _group_ewm(macd, 9)
    df["macd_hist_12_26_9"] = macd - signal

    # Bollinger
    ma20 = _group_roll(df["close"], 20, "mean")
    std20 = _group_roll(df["close"], 20, "std")
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    df["bollinger_pctb_20_2"] = (df["close"] - lower) / (upper - lower)
    df["bollinger_bw_20_2"] = (upper - lower) / ma20

    # RSI
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df["rsi_14"] = df.groupby(level=0)["close"].transform(_rsi)

    # Donchian position
    roll_high_20 = _group_roll(df["high"], 20, "max")
    roll_low_20 = _group_roll(df["low"], 20, "min")
    denom = roll_high_20 - roll_low_20
    df["donchian_pos_20"] = np.where(denom == 0, np.nan, (df["close"] - roll_low_20) / denom)

    # SuperTrend state (10,3)
    def _supertrend(group: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
        high = group["high"]
        low = group["low"]
        close = group["close"]
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=period).mean()
        hl2 = (high + low) / 2
        upperband = hl2 + multiplier * atr
        lowerband = hl2 - multiplier * atr
        final_upper = upperband.copy()
        final_lower = lowerband.copy()
        for i in range(1, len(group)):
            if upperband.iloc[i] > final_upper.iloc[i - 1]:
                final_upper.iloc[i] = final_upper.iloc[i - 1]
            if lowerband.iloc[i] < final_lower.iloc[i - 1]:
                final_lower.iloc[i] = final_lower.iloc[i - 1]
        direction = pd.Series(1.0, index=group.index)
        supertrend = pd.Series(np.nan, index=group.index)
        for i in range(len(group)):
            if i == 0:
                supertrend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = 1.0
                continue
            if close.iloc[i] > supertrend.iloc[i - 1]:
                direction.iloc[i] = 1.0
            elif close.iloc[i] < supertrend.iloc[i - 1]:
                direction.iloc[i] = -1.0
            else:
                direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1.0:
                supertrend.iloc[i] = min(final_upper.iloc[i], supertrend.iloc[i - 1])
            else:
                supertrend.iloc[i] = max(final_lower.iloc[i], supertrend.iloc[i - 1])
        return direction

    supertrend_series = pd.Series(index=df.index, dtype=float)
    for sym, grp in df.groupby(level=0):
        supertrend_series.loc[grp.index] = _supertrend(grp).values
    df["supertrend_state_10_3"] = supertrend_series.values

    # EMA slope
    ema20 = _group_ewm(df["close"], 20)
    df["ema_slope_20d"] = ema20.groupby(level=0).pct_change(5)

    # Volatility metrics
    df["hist_vol_20d"] = df.groupby(level=0)["ret_1d"].transform(
        lambda x: x.rolling(20, min_periods=20).std() * np.sqrt(252)
    )
    df["hist_vol_60d"] = df.groupby(level=0)["ret_1d"].transform(
        lambda x: x.rolling(60, min_periods=60).std() * np.sqrt(252)
    )

    # Garman-Klass (20d)
    log_hl = np.log(df["high"] / df["low"]) ** 2
    log_co = np.log(df["close"] / df.groupby(level=0)["close"].shift()) ** 2
    gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    df["gk_vol_20d"] = _group_roll(gk_var, 20, "mean").pow(0.5) * np.sqrt(252)

    # HV percentile 1y (based on hist_vol_20d)
    df["hv_percentile_1y"] = df.groupby(level=0)["hist_vol_20d"].transform(
        lambda x: x.rolling(252, min_periods=252).apply(lambda w: pd.Series(w).rank(pct=True).iloc[-1])
    )

    # Beta vs index (60d)
    idx_var_60 = index["index_ret_1d"].rolling(60, min_periods=60).var()
    idx_var_by_date = idx_var_60.reindex(df.index.get_level_values("date"))
    cov_60 = (
        df.groupby(level=0, group_keys=False)
        .apply(lambda g: g["ret_1d"].rolling(60, min_periods=60).cov(g["index_ret_1d"]))
    )
    df["beta_60d"] = cov_60.values / idx_var_by_date.values

    # Liquidity
    traded_value = df["close"] * df["volume"]
    df["adv_20d"] = _group_roll(traded_value, 20, "mean")
    df["adv_60d"] = _group_roll(traded_value, 60, "mean")
    df["adv_median_60d"] = (
        traded_value.groupby(level=0)
        .rolling(60, min_periods=60)
        .median()
        .reset_index(level=0, drop=True)
    )
    vol_mean_20 = _group_roll(df["volume"], 20, "mean")
    vol_std_20 = _group_roll(df["volume"], 20, "std")
    df["turnover_ratio_20d"] = df["volume"] / vol_mean_20
    df["volume_zscore_20d"] = (df["volume"] - vol_mean_20) / vol_std_20

    return df


def build_weekly_panel(df: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily features to weekly and compute forward targets."""
    weekly = df.copy()
    weekly["week_date"] = weekly.index.get_level_values("date").to_period("W-FRI").to_timestamp("W-FRI")

    # Take last observation per week
    weekly_panel = (
        weekly.reset_index()
        .sort_values(["symbol", "date"])
        .groupby(["symbol", "week_date"])
        .last()
    )

    # Weekly close for forward returns
    weekly_close = weekly_panel["close"]
    weekly_panel["fwd_4w_ret"] = weekly_close.groupby(level=0).shift(-4) / weekly_close - 1

    # Index weekly close and forward return
    index_weekly = (
        index.reset_index()
        .assign(week_date=lambda d: d["date"].dt.to_period("W-FRI").dt.to_timestamp("W-FRI"))
        .groupby("week_date")
        .last()
    )
    index_weekly["fwd_4w_ret_index"] = index_weekly["index_close"].shift(-4) / index_weekly["index_close"] - 1

    weekly_panel = weekly_panel.join(index_weekly["fwd_4w_ret_index"], on="week_date")
    weekly_panel["target_forward_4w_excess"] = weekly_panel["fwd_4w_ret"] - weekly_panel["fwd_4w_ret_index"]

    # Keep relevant columns
    feature_cols = [
        c
        for c in weekly_panel.columns
        if c
        not in {
            "date",
            "open",
            "high",
            "low",
            "volume",
            "fwd_4w_ret",
            "fwd_4w_ret_index",
            "index_close",
            "index_ret_1d",
            "index_ret_63d",
            "index_sma_200d",
            "index_trend_state",
            "index_vol_20d",
            "index_hv_percentile",
        }
    ]
    weekly_panel = weekly_panel[feature_cols]
    weekly_panel = weekly_panel.dropna()
    return weekly_panel


def main() -> None:
    stocks, index = load_data()
    index_feat = compute_index_features(index)
    daily_feat = compute_stock_features(stocks, index_feat)
    weekly = build_weekly_panel(daily_feat, index_feat)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(OUTPUT_PATH)
    print(f"Weekly feature panel saved to {OUTPUT_PATH} with shape {weekly.shape}")


if __name__ == "__main__":
    main()
