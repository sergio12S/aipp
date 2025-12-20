#!/usr/bin/env python3
"""
RL Training with Stable Baselines 3
PPO agent for pattern-based trading
"""

import numpy as np
import requests
from datetime import datetime
import time
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
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
MIN_SIMILARITY = 0.80
NUM_EPISODES = 1000
TRAIN_RATIO = 0.8

# Test different time periods:
# 4 Dec 2024, 18:00 UTC = 1733335200000
# 1 Jul 2024, 00:00 UTC = 1719792000000
# 1 Jan 2024, 00:00 UTC = 1704067200000
# 1 Jan 2023, 00:00 UTC = 1672531200000

CURRENT_TIMESTAMP = 1672531200000  # 1 January 2023


# =============================================================================
# Custom Gym Environment
# =============================================================================


class PatternTradingEnv(gym.Env):
    """
    Custom Environment for pattern-based trading.

    Observation: Z-normalized returns (40 values)
    Action: 0=HOLD, 1=LONG, 2=SHORT
    Reward: Percentage return based on action
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, episodes: list):
        super().__init__()

        self.episodes = episodes
        self.current_idx = 0

        # Action space: HOLD, LONG, SHORT
        self.action_space = spaces.Discrete(3)

        # Observation space: Z-normalized returns
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(QUERY_LENGTH,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick random episode
        self.current_idx = np.random.randint(len(self.episodes))
        ep = self.episodes[self.current_idx]

        obs = np.array(ep["observation"], dtype=np.float32)
        return obs, {}

    def step(self, action):
        ep = self.episodes[self.current_idx]

        # Calculate reward based on action
        if action == 1:  # LONG
            reward = ep["rewardLong"]
        elif action == 2:  # SHORT
            reward = ep["rewardShort"]
        else:  # HOLD
            reward = 0.0

        # Episode is done after one decision
        terminated = True
        truncated = False

        # Next observation (not used since episode ends)
        obs = np.array(ep["observation"], dtype=np.float32)

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass


# =============================================================================
# Data Fetching
# =============================================================================


def fetch_similar_episodes(timestamp: int, num_episodes: int = 500) -> list:
    """Fetch similar patterns from API."""
    payload = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "timestamp": timestamp,
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


def split_train_test(
    episodes: list, current_ts: int, train_ratio: float = 0.8
) -> tuple:
    """
    Split episodes by timestamp, EXCLUDING future data.

    1. Filter out episodes from the future (timestamp > current_ts)
    2. Sort remaining by timestamp
    3. Split into train/test
    """
    # CRITICAL: Remove future episodes to prevent data leakage
    valid_episodes = [ep for ep in episodes if ep["timestamp"] < current_ts]

    if len(valid_episodes) < len(episodes):
        print(
            f"⚠️  Removed {len(episodes) - len(valid_episodes)} future episodes (data leakage prevention)"
        )

    sorted_episodes = sorted(valid_episodes, key=lambda x: x["timestamp"])
    split_idx = int(len(sorted_episodes) * train_ratio)
    return sorted_episodes[:split_idx], sorted_episodes[split_idx:]


def ts_to_date(ts: int) -> str:
    return datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M")


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_on_test(model, test_episodes: list, n_random_runs: int = 100):
    """Evaluate trained model on test episodes."""

    print(f"\n📊 Evaluating on {len(test_episodes)} test episodes")

    agent_rewards = []
    agent_actions = []
    optimal_rewards = []

    for ep in test_episodes:
        obs = np.array(ep["observation"], dtype=np.float32)

        # Model prediction
        action, _ = model.predict(obs, deterministic=True)

        if action == 1:
            reward = ep["rewardLong"]
        elif action == 2:
            reward = ep["rewardShort"]
        else:
            reward = 0.0

        agent_rewards.append(reward)
        agent_actions.append(action)

        # Optimal (oracle)
        optimal_rewards.append(max(ep["rewardLong"], ep["rewardShort"], 0))

    # Multiple random baselines
    random_pnls = []
    for _ in range(n_random_runs):
        run_rewards = []
        for ep in test_episodes:
            action = np.random.randint(3)
            if action == 1:
                run_rewards.append(ep["rewardLong"])
            elif action == 2:
                run_rewards.append(ep["rewardShort"])
            else:
                run_rewards.append(0)
        random_pnls.append(sum(run_rewards))

    # Print results
    print("\n" + "=" * 60)
    print("📈 EVALUATION RESULTS")
    print("=" * 60)

    timestamps = [ep["timestamp"] for ep in test_episodes]
    print(
        f"\nTest period: {ts_to_date(min(timestamps))} → {ts_to_date(max(timestamps))}"
    )
    print(f"Test episodes: {len(test_episodes)}")

    agent_total = sum(agent_rewards)
    agent_mean = np.mean(agent_rewards)
    agent_wins = sum(1 for r in agent_rewards if r > 0)
    agent_win_rate = agent_wins / len(agent_rewards) * 100

    print(f"\n🤖 AGENT (PPO):")
    print(f"   Mean Reward: {agent_mean:+.3f}%")
    print(f"   Total PnL:   {agent_total:+.2f}%")
    print(f"   Win Rate:    {agent_win_rate:.1f}%")

    random_mean = np.mean(random_pnls)
    random_std = np.std(random_pnls)

    print(f"\n🎲 RANDOM (100 runs):")
    print(f"   Mean PnL:    {random_mean:+.2f}% (std: {random_std:.2f}%)")
    print(f"   Range:       {min(random_pnls):+.2f}% to {max(random_pnls):+.2f}%")

    optimal_total = sum(optimal_rewards)
    print(f"\n🔮 OPTIMAL (oracle):")
    print(f"   Total PnL:   {optimal_total:+.2f}%")

    percentile = sum(1 for r in random_pnls if agent_total > r) / len(random_pnls) * 100
    print(f"\n📊 COMPARISON:")
    print(f"   Agent vs Random Mean: {agent_total - random_mean:+.2f}%")
    print(f"   Agent percentile:     {percentile:.1f}%")
    print(
        f"   Agent vs Optimal:     {agent_total / optimal_total * 100:.1f}% of optimal"
    )

    print(f"\n🎯 Action Distribution:")
    print(
        f"   HOLD:  {agent_actions.count(0):3d} ({agent_actions.count(0) / len(agent_actions) * 100:.1f}%)"
    )
    print(
        f"   LONG:  {agent_actions.count(1):3d} ({agent_actions.count(1) / len(agent_actions) * 100:.1f}%)"
    )
    print(
        f"   SHORT: {agent_actions.count(2):3d} ({agent_actions.count(2) / len(agent_actions) * 100:.1f}%)"
    )

    return {
        "agent_pnl": agent_total,
        "random_mean": random_mean,
        "optimal_pnl": optimal_total,
        "percentile": percentile,
    }


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 60)
    print("🚀 RL TRAINING with Stable Baselines 3")
    print("=" * 60)

    print(f"\n📅 Current moment: {ts_to_date(CURRENT_TIMESTAMP)}")
    print(f"🔍 Searching for patterns with similarity > {MIN_SIMILARITY}")

    # Fetch data
    print(f"\n⏳ Fetching {NUM_EPISODES} similar episodes...")
    episodes = fetch_similar_episodes(CURRENT_TIMESTAMP, NUM_EPISODES)
    print(f"✅ Fetched {len(episodes)} episodes")

    if len(episodes) < 20:
        print("❌ Not enough episodes!")
        return

    # Split train/test (excludes future data!)
    train_episodes, test_episodes = split_train_test(
        episodes, CURRENT_TIMESTAMP, TRAIN_RATIO
    )

    if len(train_episodes) < 10 or len(test_episodes) < 5:
        print(f"❌ Not enough valid episodes after removing future data!")
        print(f"   Train: {len(train_episodes)}, Test: {len(test_episodes)}")
        return

    train_ts = [ep["timestamp"] for ep in train_episodes]
    test_ts = [ep["timestamp"] for ep in test_episodes]

    print(f"\n📊 Train/Test Split:")
    print(
        f"   Train: {len(train_episodes)} episodes ({ts_to_date(min(train_ts))} → {ts_to_date(max(train_ts))})"
    )
    print(
        f"   Test:  {len(test_episodes)} episodes ({ts_to_date(min(test_ts))} → {ts_to_date(max(test_ts))})"
    )

    # Create environment
    env = DummyVecEnv([lambda: PatternTradingEnv(train_episodes)])

    # Train PPO
    print(f"\n🎓 Training PPO agent...")
    start_time = time.time()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=64,
        batch_size=32,
        n_epochs=10,
        gamma=0.0,  # No discounting - immediate reward
        ent_coef=0.1,  # Encourage exploration
        verbose=0,
        policy_kwargs=dict(
            net_arch=[64, 64]  # Two hidden layers
        ),
    )

    # Train for many timesteps
    total_timesteps = len(train_episodes) * 500
    print(f"   Training for {total_timesteps} timesteps...")

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    train_time = time.time() - start_time
    print(f"\n⏱️  Training completed in {train_time:.1f}s")

    # Evaluate
    results = evaluate_on_test(model, test_episodes)

    # Also try A2C and DQN
    print("\n" + "=" * 60)
    print("🔄 Comparing with other algorithms...")
    print("=" * 60)

    # A2C
    print("\n🎓 Training A2C...")
    env_a2c = DummyVecEnv([lambda: PatternTradingEnv(train_episodes)])
    model_a2c = A2C(
        "MlpPolicy",
        env_a2c,
        learning_rate=0.0007,
        gamma=0.0,
        ent_coef=0.1,
        verbose=0,
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    model_a2c.learn(total_timesteps=total_timesteps, progress_bar=True)

    print("\n📊 A2C Results:")
    results_a2c = evaluate_on_test(model_a2c, test_episodes)

    # DQN
    print("\n🎓 Training DQN...")
    env_dqn = DummyVecEnv([lambda: PatternTradingEnv(train_episodes)])
    model_dqn = DQN(
        "MlpPolicy",
        env_dqn,
        learning_rate=0.0001,
        buffer_size=10000,
        learning_starts=100,
        batch_size=32,
        gamma=0.0,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=0,
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    model_dqn.learn(total_timesteps=total_timesteps, progress_bar=True)

    print("\n📊 DQN Results:")
    results_dqn = evaluate_on_test(model_dqn, test_episodes)

    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"\n{'Algorithm':<10} {'PnL':>10} {'vs Random':>12} {'Percentile':>12}")
    print("-" * 46)
    print(
        f"{'PPO':<10} {results['agent_pnl']:>+10.2f}% {results['agent_pnl'] - results['random_mean']:>+12.2f}% {results['percentile']:>11.1f}%"
    )
    print(
        f"{'A2C':<10} {results_a2c['agent_pnl']:>+10.2f}% {results_a2c['agent_pnl'] - results_a2c['random_mean']:>+12.2f}% {results_a2c['percentile']:>11.1f}%"
    )
    print(
        f"{'DQN':<10} {results_dqn['agent_pnl']:>+10.2f}% {results_dqn['agent_pnl'] - results_dqn['random_mean']:>+12.2f}% {results_dqn['percentile']:>11.1f}%"
    )
    print(f"{'Random':<10} {results['random_mean']:>+10.2f}%")
    print(f"{'Optimal':<10} {results['optimal_pnl']:>+10.2f}%")

    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
