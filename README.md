# Monster Momentum – Frozen Composite Signal System

## 1. What this system is

This repository implements a **production-grade, weekly, cross-sectional momentum signal pipeline** for Indian equities (NIFTY 500 universe).

The system is **not a black-box ML model**.
It is a **frozen, deterministic composite signal** built from carefully selected technical and liquidity features, designed to rank stocks weekly and support systematic portfolio construction.

The output is a **ranked signal**, not a return forecast.

---

## 2. Core principles

This system is explicitly built on the following principles:

### a) Cross-sectional momentum (not time-series prediction)

* Stocks are ranked **relative to each other** every week
* The strategy exploits **persistent leadership**, not absolute price prediction

### b) Feature discipline

* All features are:

  * observable at decision time
  * weekly aligned (W-FRI)
  * non-leaking (no forward information)
* Features are **frozen before backtesting**

### c) Composite > single indicator

* Individual indicators are noisy
* Cross-sectional z-scored composites reduce variance
* Redundancy is explicitly pruned

### d) Reproducibility first

* Every critical artifact is hashed (SHA256)
* Feature selection, composites, and backtests are **fully reproducible**
* No “silent fallbacks” or auto-healing logic

---

## 3. System architecture (high level)

```
Raw OHLCV (Yahoo)
        ↓
Weekly Feature Engineering
        ↓
Feature Selection (IC + stability + redundancy)
        ↓
Frozen Composite Construction
        ↓
Weekly Ranking Signal
        ↓
Portfolio Construction (Top-N)
        ↓
Risk Controls (costs, drawdowns, regime gates – optional)
```

---

## 4. What exactly was built

### 4.1 Feature selection (frozen)

Features were selected **separately for each horizon** using:

* Mean Spearman IC
* IC t-statistic
* Positive IC fraction
* Minimum coverage
* Correlation pruning (Spearman)

Outputs:

* `results/feature_select_v1/selected_features_mr_1w.json`
* `results/feature_select_v1/selected_features_trend_4w.json`

These lists are **frozen inputs**.

### 4.2 Frozen composite signal

A deterministic composite is constructed as:

1. Cross-sectional z-score per week
2. Sign-aligned (momentum vs mean-reversion)
3. Equal-weight average
4. Minimum feature count enforced

Outputs:

* `score_mr_1w`
* `score_trend_4w`
* `rank_mr_1w`
* `rank_trend_4w`

Stored in:

```
data/frozen/composites_selected_v1.parquet
```

With full provenance:

```
data/frozen/composites_selected_v1_manifest.json
```

---

## 5. How the system was tested

### 5.1 Backtest design

* Weekly rebalance (W-FRI close)
* Top-10 ranked stocks
* Equal weight
* Full turnover
* No leverage
* Close-to-close returns

### 5.2 Transaction cost stress testing

Costs applied as:

```
net_return = gross_return − (bps × weekly_turnover)
```

Tested at:

* 0 bps (ideal)
* 30 bps (realistic)
* 50 bps (conservative)

### 5.3 Results summary (trend_4w signal)

| Cost   | Sharpe | Max DD | Status                  |
| ------ | ------ | ------ | ----------------------- |
| 0 bps  | ~1.34  | ~-22%  | Research pass           |
| 30 bps | ~1.05  | ~-23%  | Deployable              |
| 50 bps | ~0.86  | ~-26%  | Tradable (reduced size) |

### 5.4 Rolling stability diagnostics

* Rolling 26-week Sharpe computed
* ~75–80% of weeks positive
* No prolonged negative regime
* Drawdowns coincide with known momentum crashes

This confirms **signal persistence**, not curve-fit luck.

---

## 6. How this system should be used

### Intended use

* **Signal engine**, not a complete trading system
* Weekly batch generation
* Feeds downstream portfolio logic

### Typical usage

1. Generate weekly ranks
2. Select top-N stocks
3. Apply execution logic externally (broker / OMS)
4. Apply risk controls before capital deployment

This system intentionally **does not** handle:

* order placement
* intraday execution
* slippage modeling at tick level

---

## 7. What this system is NOT

* ❌ Not a price prediction model
* ❌ Not ML-heavy or neural
* ❌ Not adaptive during live trading
* ❌ Not optimized for intraday trading

This is a **robust, slow-moving alpha signal**.

---

## 8. What is intentionally deferred (next actions)

These are **deliberate next layers**, not missing pieces:

### 8.1 Market regime gating

* Trade only if index 20W SMA > 40W SMA
* Goal: reduce drawdowns during momentum crashes

### 8.2 Position sizing

* Volatility targeting
* Cap per-name exposure
* Liquidity-aware sizing

### 8.3 Capital allocation

* Satellite allocation
* Combine with orthogonal signals
* Dynamic exposure scaling based on rolling Sharpe

### 8.4 Monitoring & kill-switches

* Rolling Sharpe < 0 for N weeks → pause
* Turnover spike alerts
* Breadth collapse alerts

---

## 9. Why conformal prediction is not used (yet)

Conformal prediction is designed for **calibrated return forecasts**.

This system:

* produces **rankings**, not point forecasts
* decisions are ordinal (top-N), not threshold-based

Conformal prediction becomes relevant **only if**:

* you move to supervised return prediction
* or want formal abstention guarantees

At this stage, it adds complexity without benefit.

---

## 10. Status

**Current status:**
✔ Research complete
✔ Signal frozen
✔ Cost-robust
✔ Ready for controlled deployment with risk gates

Any change to:

* features
* horizons
* composite logic

→ must result in a **new versioned freeze**.

---

## 11. Usage examples (how to run this system)

### 11.1 One-time batch run (research / backfill)

```bash
# 1. Build frozen composites (only when features/selection change)
python build_frozen_composites.py \
  --features data/weekly_features_yahoo.parquet \
  --mr-selected results/feature_select_v1/selected_features_mr_1w.json \
  --trend-selected results/feature_select_v1/selected_features_trend_4w.json \
  --out data/frozen/composites_selected_v1.parquet \
  --manifest data/frozen/composites_selected_v1_manifest.json

# 2. Run weekly backtest (example with costs)
python backtest_weekly_from_frozen_composites.py \
  --features data/weekly_features_yahoo.parquet \
  --composites data/frozen/composites_selected_v1.parquet \
  --signal rank_trend_4w \
  --topn 10 \
  --fee_bps_per_turnover 30 \
  --out-dir results/backtest_selected_v1_cost30
```

---

## 12. How this should be run in practice (weekly automation)

**This system is weekly by construction.** All features, composites, and ranks are aligned to **W-FRI (weekly Friday close)**.

### 13. Friday vs Monday — which is correct?

**Recommended:** run after Friday close, execute Friday close or Monday open.

* Signal date: Friday
* Execution: Friday close or Monday open
* Data cutoff: Friday close

**Monday runs:** allowed only if using last Friday’s data; do not recompute with Monday prices.

---

## 14. Recommended production schedule

**Friday (cleanest)**
```
Friday 3:45–4:00 PM IST
```
Steps: update data → build composites → generate ranks → send orders.

**Monday open (operationally convenient)**
```
Monday 8:30–9:00 AM IST
```
Steps: load last Friday’s features/composites → generate ranks → place orders at open.

---

## 15. Example: automated weekly run (cron)

```cron
0 18 * * 5 cd /home/user/monster_mom_extend && \
python build_frozen_composites.py \
  --features data/weekly_features_yahoo.parquet \
  --mr-selected results/feature_select_v1/selected_features_mr_1w.json \
  --trend-selected results/feature_select_v1/selected_features_trend_4w.json && \
python backtest_weekly_from_frozen_composites.py \
  --features data/weekly_features_yahoo.parquet \
  --composites data/frozen/composites_selected_v1.parquet \
  --signal rank_trend_4w \
  --topn 10 \
  --out-dir results/live_signal
```

---

## 16. Practical recommendation (clear answer)

* Treat this as a **Friday signal** regardless of when you run it.
* Do not recompute features mid-week or with Monday prices.

---

## 17. Next actions

* Add market regime gating
* Add position sizing
* Add live signal export
* Add monitoring & kill-switches

---

## 18. Files of interest (frozen v1)

* Feature selection: `results/feature_select_v1/selected_features_*.json`, `results/feature_select_v1/selection_manifest.json`
* Composites: `data/frozen/composites_selected_v1.parquet` (ignored), manifest: `data/frozen/composites_selected_v1_manifest.json`
* Freeze pointer: `frozen/selected_v1_pointer.json`
* Backtest (default): `results/backtest_selected_v1/`

Any change to features or composites must be versioned and re-frozen.
