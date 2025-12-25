# Execution & Grid Intel: Precise Entries

The "last mile" of a trade is execution. Pattern matching doesn't just give you a direction; it provides statistical bounds for price action.

## 1. Grid Trading Guidance

Standard grid bots use fixed steps (e.g., buy every 1% down). AI Price Patterns provides **Dynamic Grid Intel** based on historical volatility and "Market Context."

### Key Metrics:
- **Suggested Step Pct**: Volatility-adjusted distance between grid levels.
- **Sigma (σ) Levels**: Historical "standard deviation" bands. If price hits a 2σ or 3σ level, it's often a high-probability reversal point.
- **Bias**: Bullish, Bearish, or Neutral. This determines whether your grid should be weighted towards buys or sells.

### Usage:
- **CLI**: `aipp grid --symbol BTCUSDT`
- **SDK**: `client.get_grid_stats()`

## 2. Regime-Aware Positioning

Market regimes determine the "personality" of the trade:
- **MEAN_REVERSION**: Best for grid bots. Tight levels, high frequency.
- **MOMENTUM / TRENDING**: Riskier for grids. Wide levels, bias-aware positioning required.
- **VOLATILITY**: High risk. Reduce position size, widen stop-losses.

### The "Price Context" Zone
We provide a qualitative "Zone" to help you understand where the current price sits relative to its pattern:
- `OVERBOUGHT` / `OVERSOLD` (2σ+ extremes)
- `NEUTRAL_ACCUMULATION`
- `TREND_CONFIRMED`

---

[Back to Overview](./overview.md) | [Next: Machine Learning & RL](./ml_and_rl.md)
