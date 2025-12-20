#!/usr/bin/env python3
"""
Simple RL Training v2 - правильное разделение train/test

Логика:
1. Запрос на текущий момент (4 Dec 2025) - найди похожие паттерны с similarity > 0.85
2. API возвращает все похожие паттерны из истории
3. Сортируем по timestamp и делим на train (80%) / test (20%)
"""

import numpy as np
import requests
from collections import defaultdict
from datetime import datetime
import time
import os

# =============================================================================
# Configuration
# =============================================================================

BASE_URL = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
QUERY_LENGTH = 40
FORECAST_HORIZON = 24

# Training config
NUM_EPOCHS = 300
MIN_SIMILARITY = 0.80  # Больше данных
LEARNING_RATE = 0.0001  # Ещё меньше LR
NUM_EPISODES = 1000  # Больше похожих паттернов
L2_REGULARIZATION = 0.001  # Меньше регуляризация

# Train/test split
TRAIN_RATIO = 0.8  # 80% train, 20% test

# Текущий момент (4 Dec 2025, 18:00 UTC)
CURRENT_TIMESTAMP = 1733335200000  # 4 Dec 2025 18:00 UTC


# =============================================================================
# Simple Q-Learning Agent
# =============================================================================


class SimpleQAgent:
    """
    Simple Q-learning agent with linear function approximation.
    State: Z-normalized returns (40 values)
    Actions: 0=HOLD, 1=LONG, 2=SHORT
    """

    def __init__(
        self, state_dim: int, n_actions: int = 3, lr: float = 0.01, l2_reg: float = 0.01
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = lr
        self.l2_reg = l2_reg

        # Linear weights for Q-function: Q(s,a) = w[a] @ s + b[a]
        self.weights = np.random.randn(n_actions, state_dim) * 0.01
        self.biases = np.zeros(n_actions)

        # Exploration
        self.epsilon = 0.3  # Меньше начальная exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        # Stats
        self.action_counts = defaultdict(int)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Compute Q-values for all actions."""
        return self.weights @ state + self.biases

    def predict(self, state: np.ndarray, explore: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if explore and np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            q_values = self.get_q_values(state)
            action = np.argmax(q_values)

        self.action_counts[action] += 1
        return action

    def learn(self, state: np.ndarray, action: int, reward: float):
        """Update Q-function using simple gradient descent with L2 regularization."""
        q_values = self.get_q_values(state)
        target = reward
        error = target - q_values[action]

        # Gradient update with L2 regularization
        self.weights[action] += self.lr * (
            error * state - self.l2_reg * self.weights[action]
        )
        self.biases[action] += self.lr * error

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# =============================================================================
# Data Fetching
# =============================================================================


def fetch_similar_episodes(timestamp: int, num_episodes: int = 500) -> list:
    """
    Запрос похожих паттернов на указанный момент времени.
    Возвращает все похожие эпизоды из всей истории.
    """
    payload = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "timestamp": timestamp,  # Ищем паттерны похожие на этот момент
        "queryLength": QUERY_LENGTH,
        "forecastHorizon": FORECAST_HORIZON,
        "numEpisodes": num_episodes,
        "minSimilarity": MIN_SIMILARITY,
    }

    try:
        response = requests.post(f"{BASE_URL}/api/rl/simple", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("episodes", [])
    except Exception as e:
        print(f"Error fetching episodes: {e}")
        return []


def split_train_test(episodes: list, train_ratio: float = 0.8) -> tuple:
    """
    Разделение эпизодов на train/test по временной метке.
    Сортируем по timestamp, берём первые 80% на train, остальные на test.
    """
    # Сортировка по timestamp (от старых к новым)
    sorted_episodes = sorted(episodes, key=lambda x: x["timestamp"])

    split_idx = int(len(sorted_episodes) * train_ratio)

    train_episodes = sorted_episodes[:split_idx]
    test_episodes = sorted_episodes[split_idx:]

    return train_episodes, test_episodes


def ts_to_date(ts: int) -> str:
    """Convert timestamp to readable date."""
    return datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M")


# =============================================================================
# Training Loop
# =============================================================================


def train_agent(agent: SimpleQAgent, train_episodes: list, epochs: int = 100):
    """Train agent on training episodes."""

    print(f"\n🎓 Training on {len(train_episodes)} episodes for {epochs} epochs")

    for epoch in range(epochs):
        epoch_rewards = []

        # Shuffle episodes each epoch
        np.random.shuffle(train_episodes)

        for ep in train_episodes:
            obs = np.array(ep["observation"])
            action = agent.predict(obs, explore=True)

            if action == 1:  # LONG
                reward = ep["rewardLong"]
            elif action == 2:  # SHORT
                reward = ep["rewardShort"]
            else:  # HOLD
                reward = 0

            agent.learn(obs, action, reward)
            epoch_rewards.append(reward)

        agent.decay_epsilon()

        if (epoch + 1) % 10 == 0:
            mean_reward = np.mean(epoch_rewards)
            print(
                f"  Epoch {epoch + 1:3d}: reward={mean_reward:+.3f}%, ε={agent.epsilon:.2f}"
            )

    return agent


def evaluate_agent(
    agent: SimpleQAgent, test_episodes: list, n_random_runs: int = 100
) -> dict:
    """Evaluate agent on test episodes with multiple random baseline runs."""

    print(f"\n📊 Evaluating on {len(test_episodes)} episodes")
    print(f"   Running {n_random_runs} random baselines for statistical comparison")

    results = {
        "agent": {"rewards": [], "actions": []},
        "random_runs": [],  # Multiple random runs
        "optimal": {"rewards": []},  # Optimal (oracle) baseline
    }

    for ep in test_episodes:
        obs = np.array(ep["observation"])

        # Agent prediction (no exploration)
        agent_action = agent.predict(obs, explore=False)
        if agent_action == 1:
            agent_reward = ep["rewardLong"]
        elif agent_action == 2:
            agent_reward = ep["rewardShort"]
        else:
            agent_reward = 0

        results["agent"]["rewards"].append(agent_reward)
        results["agent"]["actions"].append(agent_action)

        # Optimal (oracle) - always picks the best action
        optimal_reward = max(ep["rewardLong"], ep["rewardShort"], 0)
        results["optimal"]["rewards"].append(optimal_reward)

    # Multiple random baseline runs
    for _ in range(n_random_runs):
        run_rewards = []
        for ep in test_episodes:
            random_action = np.random.randint(3)
            if random_action == 1:
                run_rewards.append(ep["rewardLong"])
            elif random_action == 2:
                run_rewards.append(ep["rewardShort"])
            else:
                run_rewards.append(0)
        results["random_runs"].append(sum(run_rewards))

    return results


def print_results(results: dict, test_episodes: list):
    """Print evaluation results."""

    agent_rewards = results["agent"]["rewards"]
    optimal_rewards = results["optimal"]["rewards"]
    random_pnls = results["random_runs"]

    print("\n" + "=" * 60)
    print("📈 EVALUATION RESULTS")
    print("=" * 60)

    # Test period info
    timestamps = [ep["timestamp"] for ep in test_episodes]
    print(
        f"\nTest period: {ts_to_date(min(timestamps))} → {ts_to_date(max(timestamps))}"
    )
    print(f"Test episodes: {len(test_episodes)}")

    # Agent stats
    agent_mean = np.mean(agent_rewards)
    agent_std = np.std(agent_rewards)
    agent_total = np.sum(agent_rewards)
    agent_wins = sum(1 for r in agent_rewards if r > 0)
    agent_win_rate = agent_wins / len(agent_rewards) * 100

    print(f"\n🤖 AGENT:")
    print(f"   Mean Reward: {agent_mean:+.3f}% (std: {agent_std:.3f}%)")
    print(f"   Total PnL:   {agent_total:+.2f}%")
    print(f"   Win Rate:    {agent_win_rate:.1f}%")

    # Random stats (averaged over 100 runs)
    random_mean_pnl = np.mean(random_pnls)
    random_std_pnl = np.std(random_pnls)
    random_min_pnl = np.min(random_pnls)
    random_max_pnl = np.max(random_pnls)

    print(f"\n🎲 RANDOM (100 runs):")
    print(f"   Mean PnL:    {random_mean_pnl:+.2f}% (std: {random_std_pnl:.2f}%)")
    print(f"   Range:       {random_min_pnl:+.2f}% to {random_max_pnl:+.2f}%")

    # Optimal (oracle) stats
    optimal_total = np.sum(optimal_rewards)
    print(f"\n🔮 OPTIMAL (oracle):")
    print(f"   Total PnL:   {optimal_total:+.2f}%")

    # Comparison
    percentile = sum(1 for r in random_pnls if agent_total > r) / len(random_pnls) * 100
    print(f"\n📊 COMPARISON:")
    print(f"   Agent vs Random Mean: {agent_total - random_mean_pnl:+.2f}%")
    print(
        f"   Agent percentile:     {percentile:.1f}% (beats {percentile:.1f}% of random runs)"
    )
    print(
        f"   Agent vs Optimal:     {agent_total / optimal_total * 100:.1f}% of optimal"
    )

    # Action distribution
    agent_actions = results["agent"]["actions"]
    print(f"\n🎯 Agent Action Distribution:")
    print(
        f"   HOLD:  {agent_actions.count(0):3d} ({agent_actions.count(0) / len(agent_actions) * 100:.1f}%)"
    )
    print(
        f"   LONG:  {agent_actions.count(1):3d} ({agent_actions.count(1) / len(agent_actions) * 100:.1f}%)"
    )
    print(
        f"   SHORT: {agent_actions.count(2):3d} ({agent_actions.count(2) / len(agent_actions) * 100:.1f}%)"
    )


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 60)
    print("🚀 RL TRAINING v2 - Correct Train/Test Split")
    print("=" * 60)

    # Current moment
    print(f"\n📅 Current moment: {ts_to_date(CURRENT_TIMESTAMP)}")
    print(
        f"🔍 Searching for patterns similar to NOW with similarity > {MIN_SIMILARITY}"
    )

    # Step 1: Fetch all similar episodes
    print(f"\n⏳ Fetching {NUM_EPISODES} similar episodes...")
    start_time = time.time()

    episodes = fetch_similar_episodes(CURRENT_TIMESTAMP, NUM_EPISODES)

    fetch_time = time.time() - start_time
    print(f"✅ Fetched {len(episodes)} episodes in {fetch_time:.1f}s")

    if len(episodes) < 10:
        print("❌ Not enough episodes for training!")
        return

    # Show episode time range
    timestamps = [ep["timestamp"] for ep in episodes]
    print(
        f"   Time range: {ts_to_date(min(timestamps))} → {ts_to_date(max(timestamps))}"
    )
    print(f"   Avg similarity: {np.mean([ep['similarity'] for ep in episodes]):.4f}")

    # Step 2: Split into train/test by timestamp
    train_episodes, test_episodes = split_train_test(episodes, TRAIN_RATIO)

    train_timestamps = [ep["timestamp"] for ep in train_episodes]
    test_timestamps = [ep["timestamp"] for ep in test_episodes]

    print(
        f"\n📊 Train/Test Split ({TRAIN_RATIO * 100:.0f}/{(1 - TRAIN_RATIO) * 100:.0f}):"
    )
    print(
        f"   Train: {len(train_episodes)} episodes ({ts_to_date(min(train_timestamps))} → {ts_to_date(max(train_timestamps))})"
    )
    print(
        f"   Test:  {len(test_episodes)} episodes ({ts_to_date(min(test_timestamps))} → {ts_to_date(max(test_timestamps))})"
    )

    # Step 3: Train agent
    agent = SimpleQAgent(
        state_dim=QUERY_LENGTH, n_actions=3, lr=LEARNING_RATE, l2_reg=L2_REGULARIZATION
    )

    train_start = time.time()
    train_agent(agent, train_episodes, epochs=NUM_EPOCHS)
    train_time = time.time() - train_start

    print(f"\n⏱️  Training completed in {train_time:.1f}s")

    # Step 4: Evaluate
    results = evaluate_agent(agent, test_episodes)
    print_results(results, test_episodes)

    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
