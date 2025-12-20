# AI Price Patterns — Showcase

This folder contains ready-to-use demo examples for presenting the SDK and CLI.

## Product Capabilities (Summary)

**SDK (Python):**

- **Pattern Search**: Find historical analogues (`Client.search`)
- **Metrics**: Aggregated metrics and risk analysis (`Client.get_pattern_metrics`)
- **Grid Stats**: Statistics and recommendations for grid trading (`Client.get_grid_stats`)
- **Recalc**: Historical replay of a specific point in time (`Client.recalc_patterns`)
- **Batch**: Batch search across a watchlist (`Client.batch_search`)
- **Backtest**: Walk-forward backtesting (`Client.backtest`)
- **RL Episodes**: Parallel historical episodes for a given context (`Client.get_rl_episodes`)
- **RL Training Batch**: Flat tensors for offline RL training (`Client.get_rl_training_batch`)
- **Regimes**: Current market regime detection (used in audit/scan)
- **Audit**: Loss attribution by market regimes (`BacktestAuditor.analyze_losses`)

**CLI (`aipp`):**

- Commands: `search`, `metrics`, `grid`, `recalc`, `batch`, `backtest`, `backtest-specific`, `audit`, `scan`, `rl-episodes`, `rl-training-batch`.
- `scan`: Heartbeat for a watchlist (batch-search + regime + GO/NO-GO decision).
- `scan --watch --every N`: Continuous monitoring mode.

## Quick Start

1. Install the package (editable mode for development):

```bash
pip install -e .
```

2. Configure the API:

```bash
export AIPP_BASE_URL="https://aipricepatterns.com/api/rust"
export AIPP_API_KEY=""  # if needed
```

---

## Recommended Playbooks

These playbooks describe how to run the existing demos as a coherent product story for different audiences.

### 1) Operator Playbook (Daily Heartbeat)

Goal: Continuously scan a watchlist and provide simple GO/NO-GO signals with context.

**Run (CLI):**

```bash
# One-shot scan
aipp scan \
	--symbols BTCUSDT,ETHUSDT,SOLUSDT \
	--interval 1h \
	--q 60 \
	--f 24 \
	--limit 16 \
	--no-stability
```

**Expected Output:**

- A per-symbol shortlist of the best matching patterns (similarity-ranked).
- Regime context and a quick “why/why not” summary.
- A reproducible command line for scheduling via cron.

**Python Equivalent:**

```bash
python3 python-sdk/showcase/03_scan_watchlist.py
```

### 2) Trader Playbook (Idea → Reality Check → Attribution)

Goal: Take a candidate pattern signal, validate it with realistic costs, and understand failure modes.

**Run (Python Showcase):**

```bash
# 1) Retrieve similar historical patterns for a symbol/interval
python3 python-sdk/showcase/01_search.py

# 2) Walk-forward backtest + regime attribution (audit)
python3 python-sdk/showcase/02_backtest_and_audit.py

# 3) Investor-grade “reality check” with friction (fees + slippage)
export AIPP_FEE_PCT="0.04"       # 0.04 = 0.04% per trade
export AIPP_SLIPPAGE_PCT="0.02"  # 0.02 = 0.02% per trade
python3 python-sdk/showcase/05_backtesting_reality_check.py
```

Expected output artifacts:

- A concrete set of historical analogs (“why this looks like those times”).
- A walk-forward backtest report card.
- A regime audit explaining where/why losses concentrate.
- A delta between “no-cost fantasy” vs “with-cost reality”.

Optional (when the regime fits):

```bash
python3 python-sdk/showcase/04_grid_trading_intel.py
```

### 3) Investor playbook (5-minute narrative)

Goal: show an end-to-end, falsifiable workflow: watchlist → validation → risk attribution → (optional) RL context.

Run:

```bash
# One-pager narrative
python3 python-sdk/showcase/00_investor_onepager.py

# Optional: RL “parallel universes” (context-aware episode sampling)
python3 python-sdk/showcase/06_rl_parallel_universes.py

# Optional: train a specialist policy + sanity baselines (requires extra deps)
python3 -m pip install numpy gymnasium stable-baselines3
python3 python-sdk/showcase/07_rl_train_specialist_sb3.py
```

Expected output artifacts:

- A concise narrative the investor can follow (not just raw metrics).
- A “reality check” emphasis: fees/slippage, regime attribution, and reproducible commands.
- RL section framed as retrieval + confidence gating (“trade only when similar enough”), not black-box magic.

### Backtesting: “reality check” с комиссиями

По умолчанию демо показывают walk-forward backtest. Для инвесторской подачи полезно
включать реалистичную фрикцию (комиссия + проскальзывание).

**Investor one-pager** ([00_investor_onepager.py](00_investor_onepager.py)) использует:

```bash
export AIPP_FEE_PCT="0.04"       # % per trade, например 0.04 = 0.04%
export AIPP_SLIPPAGE_PCT="0.02"  # % per trade
```

Legacy (оставлено для совместимости): `AIPP_DEMO_FEE_PCT` / `AIPP_DEMO_SLIPPAGE_PCT`.

**Reality check** ([05_backtesting_reality_check.py](05_backtesting_reality_check.py)) использует:

```bash
export AIPP_FEE_PCT="0.04"
export AIPP_SLIPPAGE_PCT="0.02"
```

Legacy (оставлено для совместимости): `AIPP_BT_FEE_PCT` / `AIPP_BT_SLIPPAGE_PCT`.

3. Прогнать демо:

```bash
python3 python-sdk/showcase/00_investor_onepager.py
python3 python-sdk/showcase/01_search.py
python3 python-sdk/showcase/02_backtest_and_audit.py
python3 python-sdk/showcase/03_scan_watchlist.py
python3 python-sdk/showcase/04_grid_trading_intel.py
python3 python-sdk/showcase/05_backtesting_reality_check.py
python3 python-sdk/showcase/06_rl_parallel_universes.py
python3 python-sdk/showcase/07_rl_train_specialist_sb3.py

# optional: make reward more “realistic”
export AIPP_RL_TRADE_COST_PCT="0.04"  # 0.04 = 0.04% per trade
export AIPP_RL_DD_PENALTY="0.10"      # penalty per drawdown magnitude increase

# optional: sanity baseline gating (use suggestedAction only if similarity is high)
export AIPP_RL_SUGGESTED_MIN_SIMILARITY="0.90"

# optional: how many episodes to use in the sanity summary
export AIPP_RL_SANITY_EPISODES="10"

# optional: if gating is always ON/OFF (narrow similarity range), refetch broader episodes for sanity only
export AIPP_RL_SANITY_MIN_SIMILARITY="0.60"   # used only for sanity evaluation fetch
export AIPP_RL_SANITY_FETCH_EPISODES="120"    # pool size for sanity evaluation fetch

python3 python-sdk/showcase/07_rl_train_specialist_sb3.py

```

Примечание: `07_rl_train_specialist_sb3.py` требует дополнительных пакетов:

```bash
python3 -m pip install numpy gymnasium stable-baselines3
```

## CLI demo (быстро, без python)

```bash
aipp --help
aipp scan --symbols BTCUSDT,ETHUSDT --interval 1h --no-stability
```
