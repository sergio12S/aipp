# Research Guide: Advanced Quant Workflows

The `research/` directory is where we push the "Market Memory" approach to its limits. These notebooks are designed for quants and advanced traders who want to build robust, evidence-based systems.

## 1. Calibration & Gating (The PnL Sweet Spot)

The most common question in pattern trading is: *"What similarity threshold should I use?"*

### [02_rl_episode_gating_research.ipynb](../research/02_rl_episode_gating_research.ipynb)
This notebook performs "Confidence Gating" research. It sweeps through different similarity thresholds and plots the resulting PnL distributions.

**Trader Value**: You can visually identify the threshold where "Similarity" transforms into "Profitability." For example, you might find that patterns with >0.87 similarity have a significantly higher Expectancy than those at 0.80.

### [05_similarity_calibration.ipynb](../research/05_similarity_calibration.ipynb)
Verifies the **reliability** of the engine. It checks if higher similarity actually leads to higher win rates (monotonicity).

---

## 2. Regime Conditioning (Optimizing for Context)

Different strategies work in different market "moods."

### [06_regime_conditioned_gating.ipynb](../research/06_regime_conditioned_gating.ipynb)
This notebook calculates the optimal similarity threshold *per regime*.

**Trader Value**: You might discover that in a `STABLE_UPTREND`, a similarity of 0.75 is enough to produce alpha, but in a `VOLATILE_RANGE`, you need a stricter 0.90 threshold to survive the noise.

---

## 3. Stress Testing & Robustness

Before deploying an agent or a grid bot, you need to know when it breaks.

### [07_cost_slippage_stress_grid.ipynb](../research/07_cost_slippage_stress_grid.ipynb)
Performs a 2D stress test over (Fees × Slippage). It produces a "Robustness Heatmap."

**Trader Value**: Identifies the "Surge Zone"—the maximum cost environment your strategy can handle while remaining positive. If your exchange fees increase, you'll know exactly how it affects your bottom line.

### [09_cross_asset_transfer.ipynb](../research/09_cross_asset_transfer.ipynb)
Tests if an agent trained on BTC can remain profitable on ETH or SOL without retraining.

**Quant Value**: Measures the "Generalization" of the pattern logic. High transferability indicates that the patterns are capturing universal market dynamics rather than asset-specific noise.

---

## 4. Failure Analysis (The "Worst Case" Gallery)

### [10_failure_case_gallery.ipynb](../research/10_failure_case_gallery.ipynb)
Instead of looking at a single number (like Profit Factor), this notebook generates "Episode Cards" for the top-N worst-performing patterns.

**Trader Value**: Building intuition. By studying exactly *how* a pattern failed (e.g., "It looked like a breakout but failed at a 2.5σ extreme"), you can add your own manual or automated filters to your terminal.

---

[Back to Overview](./overview.md) | [Back to Home](../README.md)
