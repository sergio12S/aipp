# AI Price Patterns Python SDK

The official Python client for [AI Price Patterns](https://aipricepatterns.com) - the search engine for financial price action.

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

## Quick Start

### 1. Initialize Client

```python
from aipricepatterns import Client

# Connect to the public API (or your on-premise instance)
# Default base_url is https://aipricepatterns.com/api/rust
client = Client()
```

### 2. Find Similar Patterns (Idea Generation)

Find historical analogues for the current BTC price action.

```python
# Search for patterns similar to the last 60 hours of BTCUSDT
results = client.search(symbol="BTCUSDT", interval="1h", q=60, top_k=5)

print(f"Found {len(results['matches'])} matches.")
for match in results['matches']:
    print(f"Date: {match['date']}, Similarity: {match['similarity']:.2f}%")
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
# Pattern search
aipp --base-url https://aipricepatterns.com/api/rust search --symbol BTCUSDT --interval 1h --q 60 --f 30 --limit 20

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
aipp --base-url https://aipricepatterns.com/api/rust rl-training-batch \
    --symbol BTCUSDT --interval 1h \
    --query-length 40 --forecast-horizon 24 \
    --batch-size 1000 --min-similarity 0.70
```

## Features

- **Pattern Search:** Find nearest neighbors in high-dimensional space using HNSW.
- **Walk-Forward Backtesting:** Strict, look-ahead bias free simulation.
- **Pandas Integration:** Native support for DataFrames.
- **Regime Awareness:** Detect current regimes and use them in `aipp scan` / `aipp audit`.

## Requirements

- Python 3.7+
- pandas
- requests
