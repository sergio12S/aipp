# Search & Discovery: Finding Repeating Patterns

The foundation of pattern-based trading is the ability to find "historical twins" for the current market state.

## 1. Live Signals Feed

The engine runs background scanners across multiple assets and timeframes (1h, 4h, 1d). It looks for patterns with high similarity to historical data that also exhibit clear directional probability.

### Why it matters:
Traders use the signals feed to discover opportunities they aren't actively watching. It’s an automated "idea generator" that filters thousands of possibilities into a manageable shortlist.

### Usage:
- **CLI**: `aipp signals`
- **SDK**: `client.get_signals()`

```python
signals = client.get_signals()
# Results contain directional probability (up_prob) and similarity scores.
```

## 2. Cross-Asset Discovery

Markets are interconnected. A price action signature that appeared in BTC might have strong predictive power even if its historical twins are found in ETH, SOL, or even Equities.

### Why it matters:
By enabling `cross_asset=True`, you expand your lookback library from a single asset to the entire market history. This significantly increases "statistical significance" and provides a broader context for the current move.

### Usage:
- **CLI**: `aipp search --symbol BTCUSDT --cross-asset`
- **SDK**: `client.search(symbol="BTCUSDT", cross_asset=True)`

## 3. Pattern Filtering (The Quant Way)

When searching, the SDK allows you to apply advanced filters. For example, you might only care about matches that:
- Have similarity > 0.85
- Occurred within a specific volatility regime
- Resulted in moves larger than X%

### Key Parameters:
- `q` (Query size): How many bars back to look.
- `f` (Forecast horizon): How far into the future the engine "looks" to score the outcome.
- `top_k`: Number of matches to return for voting.

---

[Back to Overview](./overview.md) | [Next: Validation & Risk](./validation_and_risk.md)
