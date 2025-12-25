# AI Price Patterns: Market Memory for Traders

AI Price Patterns is a "Search Engine for Financial Price Action." Unlike traditional technical analysis (which uses fixed indicators) or typical ML (which is often black-box), our engine focuses on **Market Memory**.

## What is Market Memory?

Markets are driven by human behavior and institutional algorithms. These forces create repeatable price action signatures. When a specific "pattern" of volatility and trend appears today, it is highly likely that similar conditions have occurred in the past.

Our SDK allows you to:
1. **Discover** these analogues in historical data.
2. **Quantify** the outcomes following those analogues.
3. **Execute** relative to those historical probabilities.

## Core Trading Workflows

The SDK and CLI are designed around four pillars of professional trading:

### 1. Discovery (Idea Generation)
Use the background scanner or manual search to find "frequent flyers"—patterns that are currently forming and have high similarity to historical winners.
- **Tools**: `Client.get_signals()`, `Client.search()`.

### 2. Validation (Falsification)
Before trading an idea, we "falsify" it using walk-forward backtesting. If a pattern looks good but has historically failed when accounting for fees and slippage, we discard it.
- **Tools**: `Client.backtest()`, `BacktestAuditor`.

### 3. Execution (Precision)
Once a pattern is validated, we use **Grid Intel** to determine the optimal execution strategy. This includes volatility-adjusted grid steps and regime-aware bias (e.g., Mean Reversion vs. Momentum).
- **Tools**: `Client.get_grid_stats()`.

### 4. Integration (Machine Learning)
For quant teams, the engine acts as a **Feature Factory**. Instead of feeding raw prices into a model, you feed it "Market Context"—the outcomes of the top-K similar historical episodes.
- **Tools**: `Client.get_rl_episodes()`, `Client.get_rl_training_batch()`.

---

[Next: Search & Discovery Patterns](./search_and_discovery.md) | [View Examples](./examples_guide.md) | [Advanced Research](./research_guide.md)
