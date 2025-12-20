#!/usr/bin/env python3
"""
RL Training Data API Client Example

This script demonstrates how to use the rlx-search RL API for training
Reinforcement Learning agents on similar historical market patterns.

Requirements:
    pip install requests numpy pandas

Usage:
    1. Start rlx-search server: cargo run
    2. Run this script: python rl_training_example.py
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

BASE_URL = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")


class RLXSearchClient:
    """Client for rlx-search RL Training Data API"""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()

    def get_episodes(
        self,
        current_state: List[float],
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        forecast_horizon: int = 24,
        num_episodes: int = 50,
        min_similarity: float = 0.75,
        include_actions: bool = False,
        reward_type: str = "returns",
        sampling_strategy: str = "uniform",
    ) -> Dict[str, Any]:
        """
        Fetch similar historical episodes for RL training.

        Args:
            current_state: Normalized returns for the current pattern
            symbol: Trading pair (default: BTCUSDT)
            interval: Candle interval (default: 1h)
            forecast_horizon: Episode length in bars
            num_episodes: Number of episodes to return
            min_similarity: Minimum similarity threshold (0-1)
            include_actions: Whether to include action suggestions
            reward_type: "returns", "sharpe", or "sortino"
            sampling_strategy: "uniform", "diverse", or "hard"

        Returns:
            Dict with meta, episodes, and statistics
        """
        response = self.session.post(
            f"{self.base_url}/api/rl/episodes",
            json={
                "symbol": symbol,
                "interval": interval,
                "currentState": current_state,
                "forecastHorizon": forecast_horizon,
                "numEpisodes": num_episodes,
                "minSimilarity": min_similarity,
                "includeActions": include_actions,
                "rewardType": reward_type,
                "samplingStrategy": sampling_strategy,
            },
        )
        response.raise_for_status()
        return response.json()

    def get_training_batch(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        query_length: int = 40,
        forecast_horizon: int = 24,
        batch_size: int = 500,
        min_similarity: float = 0.70,
    ) -> Dict[str, Any]:
        """
        Fetch batch data optimized for training.

        Returns flattened arrays that can be reshaped for use in RL frameworks.
        """
        response = self.session.post(
            f"{self.base_url}/api/rl/training-batch",
            json={
                "symbol": symbol,
                "interval": interval,
                "queryLength": query_length,
                "forecastHorizon": forecast_horizon,
                "batchSize": batch_size,
                "minSimilarity": min_similarity,
            },
        )
        response.raise_for_status()
        return response.json()

    def get_regimes(
        self, symbol: str = "BTCUSDT", interval: str = "1h"
    ) -> Dict[str, Any]:
        """Get available market regimes for regime-based training."""
        response = self.session.get(
            f"{self.base_url}/api/rl/regimes",
            params={"symbol": symbol, "interval": interval},
        )
        response.raise_for_status()
        return response.json()

    def get_patterns(
        self, symbol: str = "BTCUSDT", interval: str = "1h", q: int = 40, f: int = 24
    ) -> Dict[str, Any]:
        """Get current market pattern for use as currentState."""
        response = self.session.get(
            f"{self.base_url}/api/patterns/metrics",
            params={"symbol": symbol, "interval": interval, "q": q, "f": f, "limit": 5},
        )
        response.raise_for_status()
        return response.json()


def demo_episode_based_training():
    """Demonstrate episode-based training workflow."""
    print("\n" + "=" * 60)
    print("Demo: Episode-Based RL Training")
    print("=" * 60)

    client = RLXSearchClient()

    # 1. Get current market state
    print("\n1. Fetching current market pattern...")
    try:
        patterns = client.get_patterns(q=40, f=24)
        # Extract normalized returns from matches for current state
        if patterns.get("matchMetrics"):
            # Use recent price data to create current state
            current_state = [0.01, -0.02, 0.015, 0.008, -0.01, 0.02, 0.005, -0.015]
            print(f"   Using sample state: {len(current_state)} bars")
        else:
            current_state = [0.01, -0.02, 0.015, 0.008, -0.01, 0.02, 0.005, -0.015]
    except Exception as e:
        print(f"   Warning: Could not fetch live data: {e}")
        print("   Using sample state...")
        current_state = [0.01, -0.02, 0.015, 0.008, -0.01, 0.02, 0.005, -0.015]

    # 2. Fetch similar episodes
    print("\n2. Fetching similar historical episodes...")
    try:
        result = client.get_episodes(
            current_state=current_state,
            forecast_horizon=24,
            num_episodes=20,
            min_similarity=0.75,
            include_actions=True,
            reward_type="returns",
        )

        meta = result.get("meta", {})
        episodes = result.get("episodes", [])
        stats = result.get("statistics", {})

        print(f"\n   Meta:")
        print(f"   - Symbol: {meta.get('symbol')}")
        print(f"   - Interval: {meta.get('interval')}")
        print(f"   - Total Episodes: {meta.get('totalEpisodes')}")
        print(f"   - Avg Similarity: {meta.get('avgSimilarity', 0):.4f}")
        print(f"   - Regime Type: {meta.get('regimeType')}")
        print(f"   - Regime Confidence: {meta.get('regimeConfidence', 0):.4f}")

        if stats:
            print(f"\n   Statistics:")
            rd = stats.get("returnDistribution", {})
            print(f"   - Mean Return: {rd.get('mean', 0):.2f}%")
            print(f"   - Win Rate: {stats.get('winRate', 0):.1f}%")
            print(f"   - Avg Max Drawdown: {stats.get('avgMaxDrawdown', 0):.2f}%")

        if episodes:
            print(f"\n   Sample Episode (first of {len(episodes)}):")
            ep = episodes[0]
            print(f"   - ID: {ep.get('id')}")
            print(f"   - Similarity: {ep.get('similarity', 0):.4f}")
            print(
                f"   - Terminal Return: {ep.get('outcome', {}).get('terminalReturn', 0):.2f}%"
            )
            print(
                f"   - Sharpe Ratio: {ep.get('outcome', {}).get('sharpeRatio', 0):.2f}"
            )
            print(f"   - Transitions: {len(ep.get('transitions', []))} steps")

    except requests.exceptions.ConnectionError:
        print("   Error: Could not connect to rlx-search server.")
        print("   Make sure the server is running: cargo run")
        return
    except Exception as e:
        print(f"   Error: {e}")
        return


def demo_batch_training():
    """Demonstrate batch training data retrieval."""
    print("\n" + "=" * 60)
    print("Demo: Batch Training Data")
    print("=" * 60)

    client = RLXSearchClient()

    print("\n1. Fetching training batch...")
    try:
        result = client.get_training_batch(
            query_length=40, forecast_horizon=24, batch_size=100, min_similarity=0.70
        )

        meta = result.get("meta", {})
        data = result.get("data", {})

        if data:
            shapes = data.get("shapes", {})
            n = shapes.get("numEpisodes", 0)
            q = shapes.get("queryLength", 0)
            f = shapes.get("forecastHorizon", 0)

            print(f"\n   Batch Info:")
            print(f"   - Episodes: {n}")
            print(f"   - Query Length: {q}")
            print(f"   - Forecast Horizon: {f}")

            # Reshape arrays
            if n > 0 and data.get("states"):
                states = np.array(data["states"]).reshape(n, q)
                rewards = np.array(data["rewards"]).reshape(n, f)
                similarities = np.array(data["similarities"])

                print(f"\n   Array Shapes:")
                print(f"   - States: {states.shape}")
                print(f"   - Rewards: {rewards.shape}")
                print(f"   - Similarities: {similarities.shape}")

                print(f"\n   Statistics:")
                print(f"   - Avg Similarity: {similarities.mean():.4f}")
                print(f"   - Avg Episode Return: {rewards.sum(axis=1).mean():.2f}%")
                print(f"   - Return Std: {rewards.sum(axis=1).std():.2f}%")
        else:
            print("   No data returned.")

    except requests.exceptions.ConnectionError:
        print("   Error: Could not connect to rlx-search server.")
        return
    except Exception as e:
        print(f"   Error: {e}")
        return


def demo_regime_analysis():
    """Demonstrate regime-based training."""
    print("\n" + "=" * 60)
    print("Demo: Market Regimes")
    print("=" * 60)

    client = RLXSearchClient()

    print("\n1. Fetching available regimes...")
    try:
        result = client.get_regimes()
        regimes = result.get("regimes", [])

        print(f"\n   Found {len(regimes)} regimes:")
        for r in regimes:
            print(f"\n   [{r.get('id')}]")
            print(f"   - Description: {r.get('description')}")
            print(f"   - Frequency: {r.get('frequency', 0) * 100:.1f}%")
            print(f"   - Avg Return: {r.get('avgReturn', 0):.2f}%")
            print(f"   - Avg Similarity: {r.get('avgSimilarity', 0):.2f}")
            print(f"   - Sample Count: {r.get('sampleCount', 0)}")

    except requests.exceptions.ConnectionError:
        print("   Error: Could not connect to rlx-search server.")
        return
    except Exception as e:
        print(f"   Error: {e}")
        return


def demo_stable_baselines_integration():
    """Show how to integrate with Stable-Baselines3."""
    print("\n" + "=" * 60)
    print("Demo: Stable-Baselines3 Integration (Pseudo-code)")
    print("=" * 60)

    code = """
# Example integration with Stable-Baselines3

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np

# 1. Initialize client
client = RLXSearchClient()

# 2. Get current market state
patterns = client.get_patterns(q=40, f=24)
current_state = extract_returns(patterns)  # Your preprocessing

# 3. Fetch similar episodes
episodes = client.get_episodes(
    current_state=current_state,
    num_episodes=50,
    min_similarity=0.80
)

# 4. Fill replay buffer with similar experiences
buffer = ReplayBuffer(buffer_size=10000)
for ep in episodes["episodes"]:
    state = np.array(ep["initialState"]["returns"])
    for trans in ep["transitions"]:
        next_state = state[1:].tolist() + [trans["ret"]]
        buffer.add(
            obs=state,
            next_obs=np.array(next_state),
            action=0,  # Your action encoding
            reward=trans["reward"],
            done=trans["step"] == len(ep["transitions"]) - 1,
            infos={"similarity": ep["similarity"]}
        )
        state = np.array(next_state)

# 5. Pre-train on similar episodes
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, replay_buffer=buffer)

# 6. Get action for current state
action, _ = model.predict(current_state)
print(f"Recommended: {['HOLD', 'LONG', 'SHORT'][action]}")
"""
    print(code)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" RLX-Search: RL Training Data API Demo")
    print("=" * 60)
    print("\nThis demo shows how to use the RL API for training agents")
    print("on similar historical patterns.")
    print("\nMake sure rlx-search is running: cargo run")

    try:
        # Check connection
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        print(f"\n✓ Server is running at {BASE_URL}")
    except:
        print(f"\n✗ Cannot connect to server at {BASE_URL}")
        print("  Start server with: cd rlx-search && cargo run")
        print("\nShowing integration examples anyway...\n")

    # Run demos
    demo_episode_based_training()
    demo_batch_training()
    demo_regime_analysis()
    demo_stable_baselines_integration()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Integrate with your RL framework")
    print("2. Use episodes for replay buffer pre-filling")
    print("3. Experiment with different regimes for curriculum learning")
    print()
