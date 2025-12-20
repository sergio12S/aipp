#!/usr/bin/env python3
"""
Check if agent is just going LONG always (following the market)
"""

import numpy as np
import os
from datetime import datetime, timezone
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from aipricepatterns import Client

# Config
# For production use: https://aipricepatterns.com/api/rust
# For local use: http://localhost:8787
BASE_URL = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
API_KEY = os.getenv("AIPP_API_KEY")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "1h")
QUERY_LENGTH = 40
FORECAST_HORIZON = 24
MIN_SIMILARITY = 0.80
NUM_EPISODES = 1000
TRAIN_RATIO = 0.8

# Default to a more recent timestamp if not provided
# Dec 4, 2024
DEFAULT_TS = 1733335200000
CURRENT_TIMESTAMP = int(os.getenv("AIPP_RL_ANCHOR_TS", str(DEFAULT_TS)))

client = Client(base_url=BASE_URL, api_key=API_KEY)


class PatternTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, episodes: list):
        super().__init__()
        self.episodes = episodes
        self.current_idx = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(QUERY_LENGTH,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = np.random.randint(len(self.episodes))
        ep = self.episodes[self.current_idx]
        obs = np.array(ep["observation"], dtype=np.float32)
        return obs, {}

    def step(self, action):
        ep = self.episodes[self.current_idx]
        if action == 1:
            reward = ep["rewardLong"]
        elif action == 2:
            reward = ep["rewardShort"]
        else:
            reward = 0.0
        terminated = True
        obs = np.array(ep["observation"], dtype=np.float32)
        return obs, reward, terminated, False, {}


def fetch_episodes(timestamp, num=1000):
    try:
        resp = client.get_rl_simple(
            symbol=SYMBOL,
            interval=INTERVAL,
            timestamp=timestamp,
            query_length=QUERY_LENGTH,
            forecast_horizon=FORECAST_HORIZON,
            num_episodes=num,
            min_similarity=MIN_SIMILARITY,
        )
        return resp.get("episodes", [])
    except Exception as e:
        print(f"Error fetching episodes from {BASE_URL}: {e}")
        return []


def split(episodes, current_ts, ratio=0.8):
    valid = [ep for ep in episodes if ep["timestamp"] < current_ts]
    valid = sorted(valid, key=lambda x: x["timestamp"])
    idx = int(len(valid) * ratio)
    return valid[:idx], valid[idx:]


def main():
    print("=" * 60)
    print("🔍 STRATEGY ANALYSIS")
    print("=" * 60)

    # Get data
    print(
        f"Fetching episodes for {SYMBOL} {INTERVAL} at {datetime.fromtimestamp(CURRENT_TIMESTAMP / 1000, tz=timezone.utc)}..."
    )
    episodes = fetch_episodes(CURRENT_TIMESTAMP, NUM_EPISODES)

    if not episodes:
        print(
            f"❌ No episodes found! Check if {BASE_URL} is reachable and has data for {SYMBOL}."
        )
        return

    train_eps, test_eps = split(episodes, CURRENT_TIMESTAMP, TRAIN_RATIO)

    if not train_eps:
        print(
            "❌ No training episodes after split! Try a different timestamp or lower MIN_SIMILARITY."
        )
        # Fallback: if split by timestamp failed, just do a random split
        print("   Falling back to random split...")
        np.random.shuffle(episodes)
        idx = int(len(episodes) * TRAIN_RATIO)
        train_eps, test_eps = episodes[:idx], episodes[idx:]

    test_eps = sorted(test_eps, key=lambda x: x["timestamp"])

    print(f"Total episodes: {len(episodes)}")
    print(f"Train episodes: {len(train_eps)}")
    print(f"Test episodes:  {len(test_eps)}")

    # Train A2C
    print("\n🎓 Training A2C...")
    env = DummyVecEnv([lambda: PatternTradingEnv(train_eps)])
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=0.0007,
        n_steps=32,
        gamma=0.0,
        ent_coef=0.1,
        verbose=0,
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    model.learn(total_timesteps=len(train_eps) * 500, progress_bar=True)

    # Evaluate different strategies
    print("\n📊 Comparing Strategies on TEST data:")
    print("-" * 60)

    # Agent
    agent_pnl = 0
    agent_actions = []
    for ep in test_eps:
        obs = np.array(ep["observation"], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        agent_actions.append(int(action))
        if action == 1:
            agent_pnl += ep["rewardLong"]
        elif action == 2:
            agent_pnl += ep["rewardShort"]

    # Always LONG (Buy & Hold equivalent)
    always_long_pnl = sum(ep["rewardLong"] for ep in test_eps)

    # Always SHORT
    always_short_pnl = sum(ep["rewardShort"] for ep in test_eps)

    # Random
    random_pnls = []
    for _ in range(1000):
        pnl = 0
        for ep in test_eps:
            action = np.random.randint(3)
            if action == 1:
                pnl += ep["rewardLong"]
            elif action == 2:
                pnl += ep["rewardShort"]
        random_pnls.append(pnl)
    random_mean = np.mean(random_pnls)
    random_std = np.std(random_pnls)

    # Optimal
    optimal_pnl = sum(max(ep["rewardLong"], ep["rewardShort"], 0) for ep in test_eps)

    # Count agent actions
    long_count = agent_actions.count(1)
    short_count = agent_actions.count(2)
    hold_count = agent_actions.count(0)

    print(f"\n{'Strategy':<20} {'PnL':>12} {'Notes':>30}")
    print("=" * 62)
    print(f"{'🤖 A2C Agent':<20} {agent_pnl:>+11.2f}% {'':>30}")
    print(f"{'📈 Always LONG':<20} {always_long_pnl:>+11.2f}% {'(Buy & Hold)':>30}")
    print(f"{'📉 Always SHORT':<20} {always_short_pnl:>+11.2f}% {'':>30}")
    print(
        f"{'🎲 Random (mean)':<20} {random_mean:>+11.2f}% {'(±' + f'{random_std:.1f}%)':>29}"
    )
    print(f"{'🔮 Optimal':<20} {optimal_pnl:>+11.2f}% {'(Oracle)':>30}")

    print(f"\n📊 Agent Action Distribution:")
    print(f"   LONG:  {long_count:3d} ({long_count / len(test_eps) * 100:5.1f}%)")
    print(f"   SHORT: {short_count:3d} ({short_count / len(test_eps) * 100:5.1f}%)")
    print(f"   HOLD:  {hold_count:3d} ({hold_count / len(test_eps) * 100:5.1f}%)")

    print(f"\n🎯 KEY INSIGHT:")
    if abs(agent_pnl - always_long_pnl) < 5:
        print(f"   ⚠️  Agent is essentially 'Always LONG' strategy!")
        print(
            f"   Agent PnL: {agent_pnl:+.2f}% vs Always LONG: {always_long_pnl:+.2f}%"
        )
        print(f"   Difference: {abs(agent_pnl - always_long_pnl):.2f}%")
        print(f"\n   This means the agent learned to BUY in a BULL market.")
        print(f"   It doesn't show real predictive ability!")
    else:
        print(f"   ✅ Agent shows differentiated behavior from naive strategies")
        print(f"   Agent PnL: {agent_pnl:+.2f}%")
        print(f"   Always LONG: {always_long_pnl:+.2f}%")
        print(f"   Difference: {agent_pnl - always_long_pnl:+.2f}%")


if __name__ == "__main__":
    main()
