# Feature Freezing Utilities

Generate weekly features and freeze a minimal, non-redundant set based on incremental IC diagnostics.

Quick run:
```bash
python freeze_features.py --summary data/incremental_ic_summary.csv
```

# Weekly Momentum Backtest

A simple weekly rebalanced Top-N momentum backtest for Nifty 500-style data.

Example (full window):
```bash
python backtest_weekly_momentum.py \
  --start 2015-09-26 --end 2025-10-23 \
  --initial_capital 1000000 \
  --top_n 10 \
  --min_adv 20000000 \
  --w12m 0.6 --w8w 0.4 \
  --trend_gate 200d \
  --sizing equal \
  --commission_bps 5 --slippage_bps 10 \
  --features data/weekly_features_2015-09-26_2025-10-23.parquet \
  --frozen_features data/frozen_features_2015-09-26_2025-10-23.txt \
  --out_prefix backtest
```

Notes:
- Rebalances on W-FRI features; default execution at next trading day open (no lookahead).
- Costs include commission_bps + slippage_bps; optional impact via `impact_k/sqrt(adv_60d)`.
- Trend gate options: none, 200d (pct_above_200d_sma > 0), 50d_200d (both 50/200D above 0).
- Sizing options: equal-weight (default) or atr_inverse (uses atr_pct_14).
- Outputs in `results/`: equity_curve.csv, trades.csv, holdings.csv, summary.json.

# CP-based Feature Pruning (LOFO + conformal)

Run conformal LOFO pruning on a weekly features parquet (no production changes):
```bash
python cp_prune_features.py \
  --features data/weekly_features_2015-09-26_2025-10-23.parquet \
  --manifest feature_groups.yaml \
  --target-col target_forward_4w_excess \
  --start 2015-09-26 --end 2025-10-23 \
  --train-weeks 156 --cal-weeks 26 --test-weeks 26 \
  --alpha 0.1 \
  --out-summary data/cp_delta_summary.csv \
  --out-frozen data/frozen_features_cp_2015-09-26_2025-10-23.txt
```
This computes conformal interval width deltas for each feature via LOFO, applies the pruning rules per group (with minimum keeps), and writes a frozen feature list for offline review. Use `--smoke` to run a small subset for quick checks.
