# Validation & Risk: The Reality Check

Institutional trading is not about having a "winning" signal; it's about knowing the failure modes of that signal.

## 1. Walk-Forward Backtesting

Our engine uses a **Walk-Forward** approach. This means at every step in the simulation, the engine only "knows" what happened before that point. It mimics real-time trading perfectly.

### The "Reality Check" (Friction)

Trading in a vacuum is easy. Real-world trading involves **fees** and **slippage**.
- Even a 0.04% fee per trade can turn a winning strategy into a losing one over hundreds of trades.
- Our SDK enforces these costs in backtests to give you a "ground truth" performance metric.

### Institutional Metrics
We provide standard metrics used by hedge funds:
- **Sharpe/Sortino Ratio**: Risk-adjusted returns.
- **Profit Factor**: Gross Profit vs. Gross Loss.
- **Max Drawdown**: The largest "pain point" of the strategy.
- **Expectancy**: What you can expect to make *per trade*.

## 2. Regime Audit (Failure Attribution)

When a backtest shows losses, it's rarely random. Usually, a specific pattern fails in a specific market environment (e.g., trying to trade a Mean Reversion pattern during a Trending Breakout).

### The `BacktestAuditor`
The auditor maps every losing trade to the market regime that existed at that moment.
- **Actionable Output**: "Avoid trading this pattern in BEARISH_MOMENTUM regimes."
- **Why it matters**: Instead of discarding a strategy, you can "filter" it by disabling it in its weak regimes.

### Usage:
- **CLI**: `aipp audit --symbol BTCUSDT --days 90`
- **SDK**: `BacktestAuditor(client).analyze_losses(...)`

---

[Back to Overview](./overview.md) | [Next: Execution & Grid Intel](./execution_and_grid.md)
