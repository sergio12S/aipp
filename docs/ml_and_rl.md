# Machine Learning & RL Integration

For quantitative teams, AI Price Patterns serves as a **Feature Factory** and **Environment Simulator**.

## 1. Feature Extraction (Contextual RL)

Traditional RL agents only see raw price data (OHLC). A **Context-Aware Agent** sees the price *and* its historical analogues.

### Why it matters:
By including "Market Context" (e.g., the outcomes of the top-5 historical twins), the agent's state space becomes significantly more informative. It doesn't have to "guess" if a pattern is bullish; it has the historical distribution right in its input vector.

### Usage:
- `client.get_rl_episodes()`: Fetches full trajectories for training.
- `client.get_rl_training_batch()`: Fetches flat tensors optimized for `PyTorch` or `TensorFlow`.

## 2. Generating Parallel Universes

Our `get_rl_episodes` endpoint allows you to sample "Parallel Universes"—historical trajectories that look exactly like the current moment.

### How it works:
Instead of training on a single, linear history, you train on the **N most similar historical paths**.
- **Diverse Sampling**: Samples different historical periods to ensure the agent is robust.
- **Regime Gating**: Train specialized agents (e.g., a "Range Agent" trained only on Mean Reversion episodes).

## 3. High-Performance ANN Search

The SDK provides direct access to our **ANN (Approximate Nearest Neighbor)** index via the same infrastructure that powers the UI.

### Why it matters:
If you have your own proprietary price vectors, you can upsert them into our index and search for analogues using our high-performance Rust backend.

### Shared Operations:
- `client.ann_search()`: Direct vector search in the HNSW index.
- `client.ann_upsert()`: Add your own vectors/labels to the searchable pool.

---

[Back to Overview](./overview.md) | [Back to Home](../README.md)
