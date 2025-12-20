# AI Price Patterns — Research Notebooks

This folder is for quant research workflows (analysis, charts, statistics, and ML experiments) built on top of the same SDK/API used in `python-sdk/showcase/`.

## Philosophy

- Reuse the same endpoints and assumptions as the production-oriented examples.
- Prefer reproducible notebooks: explicit parameters, explicit friction (fees/slippage), explicit time windows.
- Focus on “research artifacts”: charts, distributions, regime breakdowns, and simple baselines.

## Notebooks

- `01_quant_onepager_plotly.ipynb` — a notebook version of the investor one-pager with Plotly visuals:

  - watchlist scan table
  - equity curves (no-cost vs with-cost)
  - trade return distribution
  - regime loss attribution bar chart

- `02_rl_episode_gating_research.ipynb` — RL “retrieval → statistics” research:
  - similarity distribution
  - similarity vs outcome scatter
  - confidence-gating curve (threshold vs coverage + PnL quantiles)
  - optional scikit-learn classifier cell (guarded if not installed)
- `03_rl_anchor_sweep_generalization.ipynb` — walk-forward sweep over anchors (default 300) to measure gating/threshold stability and drift over time.
- `04_retrieval_only_policy_backtest.ipynb` — retrieval-only “policy backtest” over anchors: suggestedAction vs baselines, with optional confidence gating and trade costs.
- `05_similarity_calibration.ipynb` — similarity calibration: reliability diagram (similarity→winrate/avgPnL), monotonicity checks, and gating choice (threshold vs top‑k vs smooth weights).
- `06_regime_conditioned_gating.ipynb` — regime-conditioned thresholds: optimal threshold table per regime + regime-aware vs single-threshold policy comparison.
- `07_cost_slippage_stress_grid.ipynb` — cost robustness stress-test: heatmap over (fee, slippage) and a “robustness zone” where expected PnL stays positive.
- `08_episode_feature_ablation.ipynb` — episode feature ablation: PPO performance across observation sets + sanity check for “just copying suggestedAction”.
- `09_cross_asset_transfer.ipynb` — cross-asset transfer: train PPO on one asset, test on another; produces a transfer matrix (train×test) to show generalization.
- `10_failure_case_gallery.ipynb` — failure case gallery: top‑N worst episodes with compact “episode cards” (price/returns/pos/drawdown) and a worst-vs-best breakdown.

## Setup

1. Install the SDK in editable mode:

```bash
pip install -e .
```

2. Configure API:

```bash
export AIPP_BASE_URL="https://aipricepatterns.com/api/rust"
export AIPP_API_KEY=""  # if needed
```

3. Notebook deps (for charts):

```bash
pip install ".[research]"
```
