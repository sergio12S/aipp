# 📚 AI Price Patterns SDK — Examples

This directory contains examples and case studies demonstrating how to use the AI Price Patterns SDK for institutional-grade market analysis, backtesting, and reinforcement learning.

## 🚀 Quick Start

1. **Install the SDK** (from the root of the repository):

   ```bash
   pip install -e .
   ```

2. **Set up the API URL** (optional, defaults to production):
   ```bash
   export AIPP_BASE_URL="https://aipricepatterns.com/api/rust"
   ```

---

## 🏛️ Institutional Demos & Case Studies

### [institutional_proof.py](institutional_proof.py)

The "Gold Standard" demonstration. It runs a multi-period backtest across different market regimes (Bear 2022, Recovery 2023, Bull 2024) and performs an automated data leakage audit to prove the integrity of the engine.

### [svb_audit.py](svb_audit.py)

A forensic case study of the **Silicon Valley Bank (SVB) crisis** and USDC depeg (March 2023). It shows how the pattern engine identified contrarian opportunities during peak market panic.

### [check_leakage.py](check_leakage.py)

A utility script that uses the `BacktestValidator` to ensure the engine strictly respects time causality (no look-ahead bias).

---

## 📓 Interactive Notebooks

### [institutional_demo.ipynb](institutional_demo.ipynb)

A comprehensive narrative demo for institutional clients. Covers pattern matching, metrics visualization, and backtesting workflows.

### [trader_decision_workflow.ipynb](trader_decision_workflow.ipynb)

A practical end-to-end pipeline for daily trading decisions. Shows how to combine local OHLCV data with SDK signals for risk management and paper backtesting.

---

## 🤖 Reinforcement Learning (RL)

The `rl/` directory contains advanced scripts for training trading agents using Stable Baselines3 and custom Q-learning, all fully integrated with the `aipricepatterns` SDK:

- **[rl_sb3_training.py](rl/rl_sb3_training.py)**: Standard training loop using A2C/PPO with Stable Baselines3.
- **[rl_simple_training.py](rl/rl_simple_training.py)**: A lightweight Q-learning implementation for one-decision-per-episode training.
- **[rl_context_aware_training.py](rl/rl_context_aware_training.py)**: Training with additional market context.
- **[cross_regime_test.py](rl/cross_regime_test.py)**: Training on one regime (e.g., Bear) and testing on another (e.g., Bull).
- **[plot_results.py](rl/plot_results.py)**: Visualization of agent performance, entry/exit points, and cumulative PnL.

---

## 📊 Data Requirements

For notebooks using local CSV data, place your files in:
`python-sdk/examples/data/<SYMBOL>.csv`

Expected schema: `time, open, high, low, close, volume`
