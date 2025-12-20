#!/usr/bin/env python3
"""
Cross-market regime testing:
- Train on BEAR market, test on BULL market
- Train on BULL market, test on BEAR market
"""

import numpy as np
import requests
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import os

# Config
BASE_URL = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
QUERY_LENGTH = 40
FORECAST_HORIZON = 24
MIN_SIMILARITY = 0.80
NUM_EPISODES = 1000


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
    payload = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "timestamp": timestamp,
        "queryLength": QUERY_LENGTH,
        "forecastHorizon": FORECAST_HORIZON,
        "numEpisodes": num,
        "minSimilarity": MIN_SIMILARITY,
    }
    resp = requests.post(f"{BASE_URL}/api/rl/simple", json=payload, timeout=60)
    return resp.json().get("episodes", [])


def get_valid_episodes(episodes, current_ts):
    """Get episodes before current timestamp, sorted by time."""
    valid = [ep for ep in episodes if ep["timestamp"] < current_ts]
    return sorted(valid, key=lambda x: x["timestamp"])


def evaluate(model, test_eps):
    """Evaluate model and return (agent_pnl, long_pnl, short_pnl, action_dist)"""
    agent_pnl = 0
    actions = []
    for ep in test_eps:
        obs = np.array(ep["observation"], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))
        if action == 1:
            agent_pnl += ep["rewardLong"]
        elif action == 2:
            agent_pnl += ep["rewardShort"]

    long_pnl = sum(ep["rewardLong"] for ep in test_eps)
    short_pnl = sum(ep["rewardShort"] for ep in test_eps)

    return agent_pnl, long_pnl, short_pnl, actions


def main():
    print("=" * 70)
    print("🔄 CROSS-MARKET REGIME TESTING")
    print("=" * 70)

    # Define market periods
    # Bear market: mid-2022 (after ATH crash)
    BEAR_TS = 1656633600000  # July 1, 2022
    # Bull market: early 2024 (before halving rally)
    BULL_TS = 1704067200000  # Jan 1, 2024

    print("\n📅 Market Periods:")
    print("   BEAR: July 2022 (BTC crashed from 69k to 20k)")
    print("   BULL: Jan 2024 (BTC recovering, pre-halving)")

    # Fetch episodes for both periods
    print("\n⏳ Fetching episodes...")
    bear_episodes = fetch_episodes(BEAR_TS, NUM_EPISODES)
    bull_episodes = fetch_episodes(BULL_TS, NUM_EPISODES)

    bear_valid = get_valid_episodes(bear_episodes, BEAR_TS)
    bull_valid = get_valid_episodes(bull_episodes, BULL_TS)

    print(f"   BEAR period: {len(bear_valid)} valid episodes")
    print(f"   BULL period: {len(bull_valid)} valid episodes")

    # Split each into train/test (80/20)
    bear_split = int(len(bear_valid) * 0.8)
    bull_split = int(len(bull_valid) * 0.8)

    bear_train, bear_test = bear_valid[:bear_split], bear_valid[bear_split:]
    bull_train, bull_test = bull_valid[:bull_split], bull_valid[bull_split:]

    print(f"\n📊 Data splits:")
    print(f"   BEAR: train={len(bear_train)}, test={len(bear_test)}")
    print(f"   BULL: train={len(bull_train)}, test={len(bull_test)}")

    # =========================================================================
    # TEST 1: Train on BEAR, test on BEAR (in-sample regime)
    # =========================================================================
    print("\n" + "=" * 70)
    print("🐻 TEST 1: Train BEAR → Test BEAR")
    print("=" * 70)

    env = DummyVecEnv([lambda: PatternTradingEnv(bear_train)])
    model_bear = A2C(
        "MlpPolicy",
        env,
        learning_rate=0.0007,
        n_steps=32,
        gamma=0.0,
        ent_coef=0.1,
        verbose=0,
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    model_bear.learn(total_timesteps=len(bear_train) * 500, progress_bar=True)

    agent_pnl, long_pnl, short_pnl, actions = evaluate(model_bear, bear_test)
    print(f"\n   Agent:        {agent_pnl:+.2f}%")
    print(f"   Always LONG:  {long_pnl:+.2f}%")
    print(f"   Always SHORT: {short_pnl:+.2f}%")
    print(
        f"   Actions: LONG={actions.count(1)}, SHORT={actions.count(2)}, HOLD={actions.count(0)}"
    )

    # =========================================================================
    # TEST 2: Train on BEAR, test on BULL (out-of-regime)
    # =========================================================================
    print("\n" + "=" * 70)
    print("🐻→🐂 TEST 2: Train BEAR → Test BULL (regime change!)")
    print("=" * 70)

    agent_pnl, long_pnl, short_pnl, actions = evaluate(model_bear, bull_test)
    print(f"\n   Agent:        {agent_pnl:+.2f}%")
    print(f"   Always LONG:  {long_pnl:+.2f}%")
    print(f"   Always SHORT: {short_pnl:+.2f}%")
    print(
        f"   Actions: LONG={actions.count(1)}, SHORT={actions.count(2)}, HOLD={actions.count(0)}"
    )

    if agent_pnl < long_pnl:
        print(
            f"\n   ⚠️  Bear-trained model UNDERPERFORMS on bull market by {long_pnl - agent_pnl:.2f}%"
        )

    # =========================================================================
    # TEST 3: Train on BULL, test on BULL (in-sample regime)
    # =========================================================================
    print("\n" + "=" * 70)
    print("🐂 TEST 3: Train BULL → Test BULL")
    print("=" * 70)

    env = DummyVecEnv([lambda: PatternTradingEnv(bull_train)])
    model_bull = A2C(
        "MlpPolicy",
        env,
        learning_rate=0.0007,
        n_steps=32,
        gamma=0.0,
        ent_coef=0.1,
        verbose=0,
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    model_bull.learn(total_timesteps=len(bull_train) * 500, progress_bar=True)

    agent_pnl, long_pnl, short_pnl, actions = evaluate(model_bull, bull_test)
    print(f"\n   Agent:        {agent_pnl:+.2f}%")
    print(f"   Always LONG:  {long_pnl:+.2f}%")
    print(f"   Always SHORT: {short_pnl:+.2f}%")
    print(
        f"   Actions: LONG={actions.count(1)}, SHORT={actions.count(2)}, HOLD={actions.count(0)}"
    )

    # =========================================================================
    # TEST 4: Train on BULL, test on BEAR (out-of-regime)
    # =========================================================================
    print("\n" + "=" * 70)
    print("🐂→🐻 TEST 4: Train BULL → Test BEAR (regime change!)")
    print("=" * 70)

    agent_pnl, long_pnl, short_pnl, actions = evaluate(model_bull, bear_test)
    print(f"\n   Agent:        {agent_pnl:+.2f}%")
    print(f"   Always LONG:  {long_pnl:+.2f}%")
    print(f"   Always SHORT: {short_pnl:+.2f}%")
    print(
        f"   Actions: LONG={actions.count(1)}, SHORT={actions.count(2)}, HOLD={actions.count(0)}"
    )

    if agent_pnl < short_pnl:
        print(
            f"\n   ⚠️  Bull-trained model UNDERPERFORMS on bear market by {short_pnl - agent_pnl:.2f}%"
        )

    print("\n" + "=" * 70)
    print("📋 CONCLUSION")
    print("=" * 70)
    print("""
If agent performs well ONLY in the same regime it was trained on,
but fails when market regime changes → it learned MARKET BIAS, not patterns.

True pattern recognition should work across regimes because patterns
are about local price structures, not overall market direction.
""")


if __name__ == "__main__":
    main()
