#!/usr/bin/env python3
"""
Visualize RL trading results:
- Price chart with entry/exit points
- Cumulative PnL on secondary Y-axis
"""

import numpy as np
import requests
from datetime import datetime, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import os

# =============================================================================
# Configuration
# =============================================================================

BASE_URL = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
QUERY_LENGTH = 40
FORECAST_HORIZON = 24
MIN_SIMILARITY = 0.80
NUM_EPISODES = 1000
TRAIN_RATIO = 0.8

# July 1, 2024 00:00 UTC
CURRENT_TIMESTAMP = 1719792000000


# =============================================================================
# Custom Gym Environment
# =============================================================================


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
        truncated = False
        obs = np.array(ep["observation"], dtype=np.float32)
        return obs, reward, terminated, truncated, {}


# =============================================================================
# Helper Functions
# =============================================================================


def ts_to_date(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def fetch_similar_episodes(timestamp: int, num_episodes: int = 500) -> list:
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
    valid_episodes = [ep for ep in episodes if ep["timestamp"] < current_ts]
    if len(valid_episodes) < len(episodes):
        print(f"⚠️  Removed {len(episodes) - len(valid_episodes)} future episodes")
    sorted_episodes = sorted(valid_episodes, key=lambda x: x["timestamp"])
    split_idx = int(len(sorted_episodes) * train_ratio)
    return sorted_episodes[:split_idx], sorted_episodes[split_idx:]


def fetch_price_data(start_ts: int, end_ts: int) -> list:
    """Fetch OHLC price data from the API."""
    # Use a simple endpoint to get klines
    payload = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": start_ts,
        "endTime": end_ts,
    }
    try:
        response = requests.get(f"{BASE_URL}/api/klines", params=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching prices: {e}")
        return []


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 60)
    print("📊 RL TRADING VISUALIZATION")
    print("=" * 60)

    print(f"\n📅 Current moment: {ts_to_date(CURRENT_TIMESTAMP)}")

    # Fetch episodes
    print(f"\n⏳ Fetching episodes...")
    episodes = fetch_similar_episodes(CURRENT_TIMESTAMP, NUM_EPISODES)
    print(f"✅ Fetched {len(episodes)} episodes")

    if len(episodes) < 20:
        print("❌ Not enough episodes!")
        return

    # Split train/test
    train_episodes, test_episodes = split_train_test(
        episodes, CURRENT_TIMESTAMP, TRAIN_RATIO
    )
    print(f"\n📊 Train: {len(train_episodes)}, Test: {len(test_episodes)}")

    # Train A2C (best performer)
    print(f"\n🎓 Training A2C agent...")
    env = DummyVecEnv([lambda: PatternTradingEnv(train_episodes)])

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

    total_timesteps = len(train_episodes) * 500
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    print("✅ Training complete!")

    # Evaluate and collect trade data
    print(f"\n📊 Generating trades on test data...")

    trades = []
    cumulative_pnl = []
    running_pnl = 0.0

    # Sort test episodes by timestamp
    sorted_test = sorted(test_episodes, key=lambda x: x["timestamp"])

    for ep in sorted_test:
        obs = np.array(ep["observation"], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)

        if action == 1:  # LONG
            reward = ep["rewardLong"]
            direction = "LONG"
        elif action == 2:  # SHORT
            reward = ep["rewardShort"]
            direction = "SHORT"
        else:  # HOLD
            reward = 0.0
            direction = "HOLD"

        running_pnl += reward

        # Entry timestamp (end of query period)
        entry_ts = ep["timestamp"]
        # Exit timestamp (after forecast horizon)
        exit_ts = entry_ts + FORECAST_HORIZON * 3600 * 1000  # hours to ms

        trades.append(
            {
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "action": int(action),
                "direction": direction,
                "reward": reward,
                "cumulative_pnl": running_pnl,
                # Approximate entry/exit prices from observation
                "entry_price": ep.get("entryPrice", 0),
                "exit_price": ep.get("exitPrice", 0),
            }
        )

        cumulative_pnl.append(running_pnl)

    # Stats
    total_trades = len([t for t in trades if t["action"] != 0])
    long_trades = len([t for t in trades if t["action"] == 1])
    short_trades = len([t for t in trades if t["action"] == 2])
    winning_trades = len([t for t in trades if t["reward"] > 0])

    print(f"\n📈 Trade Summary:")
    print(f"   Total trades: {total_trades}")
    print(f"   Long: {long_trades}, Short: {short_trades}")
    print(f"   Win rate: {winning_trades / len(trades) * 100:.1f}%")
    print(f"   Final PnL: {running_pnl:.2f}%")

    # ==========================================================================
    # PLOTTING (PLOTLY)
    # ==========================================================================

    print(f"\n📊 Creating visualization...")

    # Prepare data
    timestamps = [
        datetime.fromtimestamp(t["entry_ts"] / 1000, tz=timezone.utc) for t in trades
    ]
    pnl_values = [t["cumulative_pnl"] for t in trades]

    # We need price data - let's use entry prices if available, otherwise simulate
    if trades[0].get("entry_price", 0) > 0:
        prices = [t["entry_price"] for t in trades]
    else:
        base_price = 60000  # Approximate BTC price mid-2024
        prices = [base_price]
        for i, t in enumerate(trades[1:], 1):
            prev_price = prices[-1]
            if trades[i - 1]["action"] == 1:  # Previous was LONG
                price_change = trades[i - 1]["reward"] / 100
            elif trades[i - 1]["action"] == 2:  # Previous was SHORT
                price_change = -trades[i - 1]["reward"] / 100
            else:
                price_change = 0
            prices.append(prev_price * (1 + price_change))

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add BTC Price trace
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=prices,
            name="BTC Price",
            line=dict(color="blue", width=1.5),
            opacity=0.7,
        ),
        secondary_y=False,
    )

    # Add Cumulative PnL trace
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=pnl_values,
            name="Cumulative PnL (%)",
            line=dict(color="green", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 255, 0, 0.1)",
        ),
        secondary_y=True,
    )

    # Add entry markers
    long_entries_x = [timestamps[i] for i, t in enumerate(trades) if t["action"] == 1]
    long_entries_y = [prices[i] for i, t in enumerate(trades) if t["action"] == 1]

    short_entries_x = [timestamps[i] for i, t in enumerate(trades) if t["action"] == 2]
    short_entries_y = [prices[i] for i, t in enumerate(trades) if t["action"] == 2]

    fig.add_trace(
        go.Scatter(
            x=long_entries_x,
            y=long_entries_y,
            mode="markers",
            name="LONG Entry",
            marker=dict(
                symbol="triangle-up",
                size=12,
                color="green",
                line=dict(width=1, color="black"),
            ),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=short_entries_x,
            y=short_entries_y,
            mode="markers",
            name="SHORT Entry",
            marker=dict(
                symbol="triangle-down",
                size=12,
                color="red",
                line=dict(width=1, color="black"),
            ),
        ),
        secondary_y=False,
    )

    # Add profit/loss indicators (circles around entries)
    win_x = [
        timestamps[i]
        for i, t in enumerate(trades)
        if t["action"] != 0 and t["reward"] > 0
    ]
    win_y = [
        prices[i] for i, t in enumerate(trades) if t["action"] != 0 and t["reward"] > 0
    ]

    loss_x = [
        timestamps[i]
        for i, t in enumerate(trades)
        if t["action"] != 0 and t["reward"] <= 0
    ]
    loss_y = [
        prices[i] for i, t in enumerate(trades) if t["action"] != 0 and t["reward"] <= 0
    ]

    fig.add_trace(
        go.Scatter(
            x=win_x,
            y=win_y,
            mode="markers",
            name="Winning Trade",
            marker=dict(
                symbol="circle-open", size=18, color="lime", line=dict(width=2)
            ),
            showlegend=True,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=loss_x,
            y=loss_y,
            mode="markers",
            name="Losing Trade",
            marker=dict(
                symbol="circle-open", size=18, color="darkred", line=dict(width=2)
            ),
            showlegend=True,
        ),
        secondary_y=False,
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=(
                f"RL Trading Results - A2C Agent<br>"
                f"<sup>Test Period: {ts_to_date(sorted_test[0]['timestamp'])} → {ts_to_date(sorted_test[-1]['timestamp'])} | "
                f"Total PnL: {running_pnl:.2f}% | Win Rate: {winning_trades / len(trades) * 100:.1f}%</sup>"
            ),
            font=dict(size=20),
        ),
        xaxis_title="Date",
        yaxis_title="BTC Price (USD)",
        yaxis2_title="Cumulative PnL (%)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.add_hline(
        y=0, line_dash="dash", line_color="gray", opacity=0.5, secondary_y=True
    )

    # Show plot
    fig.show()


if __name__ == "__main__":
    main()
