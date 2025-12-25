# AI Price Patterns Python SDK (`aipp`)

The official Python client for [AI Price Patterns](https://aipricepatterns.com) - the search engine for financial price action.

[Source Code](https://github.com/sergio12S/aipp) | [Bug Tracker](https://github.com/sergio12S/aipp/issues)

This library allows Quants and Traders to programmatically access our pattern matching engine, run walk-forward backtests, and integrate "Market Memory" into their existing ML and Reinforcement Learning pipelines.

## Installation

```bash
# Basic installation
pip install aipricepatterns

# With Reinforcement Learning support
pip install "aipricepatterns[rl]"

# With Research/Notebook support
pip install "aipricepatterns[research]"
```

## 🏢 For Institutional Partners

AIPP provides a forensic, evidence-based approach to pattern recognition for Hedge Funds, Proprietary Trading Firms, and Quantitative Alpha Teams.

See our **[Institutional Partnership Portal](./pitch/)** for:
- **Pitch Deck**: High-level value proposition and technical edge.
- **Audit Reports**: Leakage-free integrity audits (Look-ahead bias verification).
- **Case Studies**: Verified performance during extreme events (SVB Crisis, Alpha Discovery).
- **Impact Analysis**: Quantitative and Qualitative Sharpe-lift proof.

## Detailed Documentation

To get the most out of the SDK, check out our specialized guides:

- [**Overview & Concepts**](./docs/overview.md) – "Market Memory" and trading workflows.
- [**Search & Discovery**](./docs/search_and_discovery.md) – Using scanners and cross-asset matching.
- [**Validation & Risk**](./docs/validation_and_risk.md) – Backtesting and regime-based audits.
- [**Execution & Grid Intel**](./docs/execution_and_grid.md) – Dynamic grid bots and price zones.
- [**ML & RL Integration**](./docs/ml_and_rl.md) – Episode sampling and feature factories.
- [**Examples & Case Studies**](./docs/examples_guide.md) – Real-world forensic audits and institutional proofs.
- [**Research & Quant Workflows**](./docs/research_guide.md) – Calibration, stress-testing, and regime analysis.

## Quick Start

### 1. Initialize Client

```python
from aipricepatterns import Client

# Connect to the public API (or your on-premise instance)
# Default base_url is https://aipricepatterns.com/api/rust
client = Client()
```

### 2. Live Signals & Pattern Search

Get the latest high-probability signals from our background scanner.

```python
# Get latest scanners findings
signals = client.get_signals()
for s in signals['signals']:
    print(f"{s['symbol']} {s['direction']} (prob: {s['up_prob']:.2f})")
```

Find historical analogues for a specific price action (with cross-asset support).

```python
# Search for patterns similar to the last 60 hours of BTCUSDT across all assets
results = client.search(symbol="BTCUSDT", interval="1h", q=60, top_k=5, cross_asset=True)

print(f"Found {len(results['matches'])} matches.")
for match in results['matches']:
    print(f"Date: {match['date']}, Asset: {match['symbol']}, Similarity: {match['similarity']:.2f}%")
```

### 3. Run a Walk-Forward Backtest (Validation)

Validate the predictive power of the pattern engine on historical data.

```python
# Run a simulation on the last 50,000 bars
bt = client.backtest(
    symbol="BTCUSDT",
    interval="1h",
    q=60,
    f=24,
    min_prob=0.6,  # Only trade if >60% probability
    include_stats=True
)

# Convert to Pandas for analysis
df = client.backtest_to_df(bt)
equity = client.equity_curve_to_df(bt)

print(f"Total Return: {bt['stats']['totalReturnPct']:.2f}%")
print(f"Sharpe Ratio: {bt['stats']['sharpeRatio']:.2f}")

# Plot equity curve (using Plotly)
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity['equity'], name='Equity'))
fig.show()
```

### 4. Reinforcement Learning (Market Memory)

Train agents on historical episodes retrieved by the pattern engine.

```python
# Fetch 1000 historical episodes similar to the current market state
episodes = client.get_rl_simple(
    symbol="BTCUSDT",
    interval="1h",
    num_episodes=1000
)

# Use these episodes to "warm up" your RL agent
# See examples/rl/ for full implementation
```

## Project Structure

- `src/aipricepatterns`: Core SDK source code.
- `examples/`: Practical scripts and notebooks for traders.
- `research/`: Advanced quant research notebooks (Plotly-based).
- `showcase/`: Ready-to-use demo playbooks for different market scenarios.

## CLI (`aipp`)

When installed, the package provides a console command `aipp`.

Examples:

Tip: you can set default friction for backtests via env vars:

```bash
export AIPP_FEE_PCT="0.04"
export AIPP_SLIPPAGE_PCT="0.02"
```

```bash
# Get live signals
aipp signals

# Pattern search (with cross-asset)
aipp search --symbol BTCUSDT --interval 1h --q 60 --f 30 --limit 10 --cross-asset

# Watchlist heartbeat (GO/NO-GO)
aipp --base-url https://aipricepatterns.com/api/rust scan \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT \
    --interval 1h --q 60 --f 24 --limit 16 \
    --block-regimes BEARISH_MOMENTUM,STABLE_DOWNTREND \
    --no-stability \
    --grid-hint --grid-hint-human

# Repeat scan every 60 seconds
aipp --base-url https://aipricepatterns.com/api/rust scan --symbols BTCUSDT,ETHUSDT --interval 1h --watch --every 60

# Backtest (walk-forward) with realistic costs
aipp --base-url https://aipricepatterns.com/api/rust backtest \
    --symbol BTCUSDT --interval 1h --q 24 --f 12 --step 24 \
    --fee-pct 0.04 --slippage-pct 0.02

# Micro-validation for a specific moment (timestamp in ms)
aipp --base-url https://aipricepatterns.com/api/rust backtest-specific \
    --symbol BTCUSDT --interval 1h --q 24 --f 12 \
    --timestamp 1710000000000 \
    --fee-pct 0.04 --slippage-pct 0.02

# RL: get "parallel universe" episodes for context-aware RL (recommended: anchor-ts)
aipp --base-url https://aipricepatterns.com/api/rust rl-episodes \
    --symbol BTCUSDT --interval 1h \
    --anchor-ts 1678406400000 \
    --forecast-horizon 24 --num-episodes 50 --min-similarity 0.80

# RL: get flattened (s, r, s', d) arrays for offline RL
aipp rl-training-batch \
    --symbol BTCUSDT --interval 1h \
    --query-length 40 --forecast-horizon 24 \
    --batch-size 1000 --min-similarity 0.70

# Dataset Management
aipp dataset status --symbol BTCUSDT
aipp dataset stats

# Regime Analysis
aipp regime latest --symbol BTCUSDT --interval 4h

# ANN Index Operations
aipp ann status
```

## Features

- **Live Signals:** Background scanner discovers high-probability setups across all pairs.
- **Pattern Search:** Find nearest neighbors in high-dimensional space using HNSW (cross-asset supported).
- **Walk-Forward Backtesting:** Strict, look-ahead bias free simulation with institutional metrics.
- **Dataset & ANN Management:** Programmatically expand history or manage vector indices.
- **Regime Awareness:** Detect current market environments (e.g. VOLATILE_UPTREND).

## Requirements

- Python 3.7+
- pandas
- requests
