# Examples Guide: Real-World Case Studies

The `examples/` directory contains institutional-grade case studies and integration scripts. This guide explains their purpose and the value they provide for traders and quants.

## 1. Institutional Integrity (Trust but Verify)

In quantitative finance, the biggest risk is **Data Leakage** (look-ahead bias). If a model "sees" the future during training, it will fail in live trading.

### [institutional_proof.py](../examples/institutional_proof.py)
This is the "Gold Standard" demo. It performs a multi-regime backtest across:
- 🐻 **Bear Market 2022**
- 📈 **Recovery 2023**
- 🚀 **Bull Market 2024**

**Trader Value**: It proves that the "Market Memory" approach works consistently across different market personalities, delivering Alpha relative to the benchmark.

### [check_leakage.py](../examples/check_leakage.py)
A specialized audit tool that picks a random point in history and verifies that the engine *never* returns a match from the relative "future."

**Quant Value**: Provides absolute confidence that backtests are "honest" and reproducible.

---

## 2. Forensic Analysis (Event-Driven)

### [svb_audit.py](../examples/svb_audit.py)
A deep dive into the **Silicon Valley Bank (SVB) crisis** (March 2023). During peak panic, the pattern engine identified that the current price action was highly similar to previous historical "bottoms" and "reversals."

**Trader Value**: Shows how to use the SDK to stay calm during black-swan events. While news/Twitter were bearish, the patterns were objectively bullish.

---

## 3. High-Performance Integration (ML/RL)

The `examples/rl/` folder contains the "last mile" for automated fund management.

### [rl_sb3_training.py](../examples/rl/rl_sb3_training.py)
Integrates the SDK with **Stable Baselines 3**, the industry standard for reinforcement learning.

**Value for Fund Managers**:
- **Curriculum Learning**: Train agents on simple patterns first, then move to complex regimes.
- **Warming Up**: Instead of an agent learning by trial-and-error for weeks, it "warms up" on the most similar historical episodes in minutes.

---

## 4. Interactive Decision Pipelines

### [trader_decision_workflow.ipynb](../examples/trader_decision_workflow.ipynb)
A Jupyter notebook that combines local CSV data with SDK signals.

**Trader Value**: A complete "Daily Workspace" where you can ingest your own indicators, find historical analogues, and generate a final GO/NO-GO decision.

---

[Back to Overview](./overview.md) | [Back to Home](../README.md)
