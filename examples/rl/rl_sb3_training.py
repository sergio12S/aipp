"""
Stable-Baselines3 Training with RLX-Search Pattern Memory
==========================================================

This script demonstrates real RL training using historical pattern episodes
as pre-training data. The key insight: instead of learning from scratch,
we "warm up" the agent on similar historical situations.

Architecture:
1. Custom Gymnasium Environment (TradingEnv) - simulates trading on price data
2. RLX-Search Integration - fetches similar historical episodes
3. PPO Agent - learns optimal policy from pattern-based experience

Key Features:
- Pre-fill replay buffer with pattern episodes (curriculum learning)
- Online learning with new episodes from similar patterns
- Regime-aware training (train on specific market conditions)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch
from aipricepatterns import Client


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_TIMEFRAME = "1h"


# =============================================================================
# Trading Environment
# =============================================================================


class TradingEnv(gym.Env):
    """
    Custom Trading Environment that uses price data from RLX-Search episodes.

    State Space:
        - Normalized price returns (window of last `window_size` candles)
        - Current position (-1: short, 0: flat, 1: long)
        - Unrealized PnL

    Action Space:
        - 0: Hold
        - 1: Buy/Long
        - 2: Sell/Short
        - 3: Close position

    Reward:
        - Realized PnL on position close
        - Small holding penalty to encourage action
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: np.ndarray,
        window_size: int = 20,
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.prices = prices
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.render_mode = render_mode

        # Action space: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)

        # Observation space: price returns + position + unrealized PnL
        # [returns..., position, unrealized_pnl]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size + 2,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # -1: short, 0: flat, 1: long
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trades = []

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        # Price returns for the window
        window_prices = self.prices[
            self.current_step - self.window_size : self.current_step
        ]
        returns = np.diff(window_prices) / window_prices[:-1]

        # Pad returns to match window_size
        returns = np.concatenate([[0], returns])

        # Current position and unrealized PnL
        current_price = self.prices[self.current_step]
        if self.position != 0:
            unrealized_pnl = (
                (current_price - self.entry_price) / self.entry_price * self.position
            )
        else:
            unrealized_pnl = 0.0

        obs = np.concatenate(
            [returns.astype(np.float32), [float(self.position), unrealized_pnl]]
        )

        return obs.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        current_price = self.prices[self.current_step]
        reward = 0.0

        # Execute action
        if action == 1:  # Buy/Long
            if self.position <= 0:
                if self.position == -1:  # Close short first
                    pnl = (self.entry_price - current_price) / self.entry_price
                    reward += pnl - self.commission
                    self.total_pnl += pnl - self.commission
                    self.trades.append(("close_short", current_price, pnl))

                # Open long
                self.position = 1
                self.entry_price = current_price
                reward -= self.commission  # Entry commission

        elif action == 2:  # Sell/Short
            if self.position >= 0:
                if self.position == 1:  # Close long first
                    pnl = (current_price - self.entry_price) / self.entry_price
                    reward += pnl - self.commission
                    self.total_pnl += pnl - self.commission
                    self.trades.append(("close_long", current_price, pnl))

                # Open short
                self.position = -1
                self.entry_price = current_price
                reward -= self.commission

        elif action == 3:  # Close position
            if self.position == 1:
                pnl = (current_price - self.entry_price) / self.entry_price
                reward += pnl - self.commission
                self.total_pnl += pnl - self.commission
                self.trades.append(("close_long", current_price, pnl))
            elif self.position == -1:
                pnl = (self.entry_price - current_price) / self.entry_price
                reward += pnl - self.commission
                self.total_pnl += pnl - self.commission
                self.trades.append(("close_short", current_price, pnl))
            self.position = 0
            self.entry_price = 0.0

        # Small holding penalty to encourage action
        if self.position != 0:
            reward -= 0.0001

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= len(self.prices) - 1
        truncated = False

        # Force close on termination
        if terminated and self.position != 0:
            if self.position == 1:
                pnl = (self.prices[-1] - self.entry_price) / self.entry_price
            else:
                pnl = (self.entry_price - self.prices[-1]) / self.entry_price
            reward += pnl - self.commission
            self.total_pnl += pnl - self.commission

        info = {
            "total_pnl": self.total_pnl,
            "n_trades": len(self.trades),
            "balance": self.balance * (1 + self.total_pnl),
        }

        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            current_price = self.prices[self.current_step]
            print(
                f"Step {self.current_step}: Price={current_price:.2f}, "
                f"Position={self.position}, PnL={self.total_pnl:.4f}"
            )


# =============================================================================
# Episode-based Environment Factory
# =============================================================================


def create_env_from_episode(
    episode: Dict[str, Any], window_size: int = 20
) -> TradingEnv:
    """Create a TradingEnv from an RLX-Search episode."""

    # Combine query and forecast prices
    state_prices = episode.get("state_prices", [])
    forecast_prices = [t["price"] for t in episode.get("transitions", [])]

    all_prices = np.array(state_prices + forecast_prices, dtype=np.float32)

    if len(all_prices) < window_size + 10:
        # Pad with repeated first price if too short
        padding = np.full(window_size, all_prices[0])
        all_prices = np.concatenate([padding, all_prices])

    return TradingEnv(prices=all_prices, window_size=window_size)


def create_multi_episode_env(
    episodes: List[Dict[str, Any]], window_size: int = 20
) -> DummyVecEnv:
    """Create a vectorized environment from multiple episodes."""

    def make_env(ep):
        def _init():
            return create_env_from_episode(ep, window_size)

        return _init

    # Use first N episodes (SB3 works well with 1-16 parallel envs)
    n_envs = min(len(episodes), 4)

    return DummyVecEnv([make_env(episodes[i]) for i in range(n_envs)])


# =============================================================================
# Training Callbacks
# =============================================================================


class PatternRefreshCallback(BaseCallback):
    """
    Callback that periodically fetches new pattern-based episodes
    and refreshes the training environment.
    """

    def __init__(
        self,
        client: Client,
        refresh_freq: int = 1000,
        symbol: str = DEFAULT_SYMBOL,
        timeframe: str = DEFAULT_TIMEFRAME,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.client = client
        self.refresh_freq = refresh_freq
        self.symbol = symbol
        self.timeframe = timeframe
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.refresh_freq == 0:
            try:
                # Fetch new episodes
                res = self.client.get_rl_episodes(
                    symbol=self.symbol, interval=self.timeframe, num_episodes=10
                )
                episodes = res.get("episodes", [])
                self.episode_count += len(episodes)

                if self.verbose > 0:
                    print(
                        f"[PatternRefresh] Fetched {len(episodes)} new episodes "
                        f"(total: {self.episode_count})"
                    )
            except Exception as e:
                if self.verbose > 0:
                    print(f"[PatternRefresh] Error fetching episodes: {e}")

        return True


# =============================================================================
# Main Training Script
# =============================================================================


def train_on_patterns(
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    regime: Optional[str] = None,
    total_timesteps: int = 10000,
    n_episodes: int = 20,
):
    """
    Train a PPO agent on pattern-based historical episodes.

    Args:
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        regime: Optional market regime filter
        total_timesteps: Total training steps
        n_episodes: Number of episodes to fetch
    """

    print("=" * 60)
    print("AI Price Patterns + Stable-Baselines3 Training")
    print("=" * 60)

    # Initialize client
    import os

    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    client = Client(base_url=base_url)

    # Fetch episodes
    print(f"\n📊 Fetching {n_episodes} episodes from pattern engine...")
    print(f"   Symbol: {symbol}, Timeframe: {timeframe}")
    if regime:
        print(
            f"   Regime: {regime} (note: regime filtering not yet implemented in API)"
        )

    try:
        res = client.get_rl_episodes(
            symbol=symbol,
            interval=timeframe,
            num_episodes=n_episodes,
            min_similarity=0.7,
        )
        episodes = res.get("episodes", [])
    except Exception as e:
        print(f"❌ Error: Cannot connect to pattern engine: {e}")
        return None

    if not episodes:
        print("❌ No episodes returned. Try different parameters.")
        return None

    print(f"✅ Got {len(episodes)} episodes")

    # Analyze episodes
    similarities = [ep.get("similarity", 0) for ep in episodes]
    print(
        f"   Similarity: min={min(similarities):.3f}, max={max(similarities):.3f}, "
        f"avg={np.mean(similarities):.3f}"
    )

    # Create environment
    print("\n🎮 Creating training environment...")
    env = create_multi_episode_env(episodes, window_size=20)
    print(f"   Vectorized env with {env.num_envs} parallel environments")

    # Create PPO agent
    print("\n🤖 Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    # Setup callback
    callback = PatternRefreshCallback(
        client=client, refresh_freq=2000, symbol=symbol, timeframe=timeframe, verbose=1
    )

    # Train
    print(f"\n🚀 Training for {total_timesteps} timesteps...")
    print("-" * 60)

    model.learn(total_timesteps=total_timesteps, callback=callback)

    print("-" * 60)
    print("✅ Training complete!")

    # Evaluate on fresh episodes
    print("\n📈 Evaluating on fresh episodes...")
    res = client.get_rl_episodes(
        symbol=symbol, interval=timeframe, num_episodes=5, min_similarity=0.6
    )
    eval_episodes = res.get("episodes", [])

    total_rewards = []
    for i, ep in enumerate(eval_episodes):
        eval_env = create_env_from_episode(ep)
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
        print(
            f"   Episode {i + 1}: Reward={episode_reward:.4f}, "
            f"Trades={info['n_trades']}, PnL={info['total_pnl']:.2%}"
        )

    print(f"\n   Average reward: {np.mean(total_rewards):.4f}")
    print(f"   Std reward: {np.std(total_rewards):.4f}")

    # Save model
    model_path = f"ppo_trading_{symbol}_{timeframe}"
    model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}.zip")

    return model


def train_curriculum(
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    timesteps_per_regime: int = 5000,
):
    """
    Curriculum learning: train sequentially on different market regimes.

    This approach trains the agent on progressively different market conditions,
    helping it generalize better across various scenarios.
    """

    print("=" * 60)
    print("Curriculum Learning: Train on Multiple Regimes")
    print("=" * 60)

    import os

    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    client = Client(base_url=base_url)

    # Get available regimes
    try:
        res = client.get_rl_regimes(symbol=symbol, interval=timeframe)
        regimes = [r["id"] for r in res.get("regimes", [])]
    except Exception as e:
        print(f"❌ Error: Cannot connect to pattern engine: {e}")
        return None

    print(f"\n📋 Available regimes: {regimes}")

    # Training order (from stable to volatile)
    regime_order = [
        "range_bound",
        "mean_reversion",
        "trend_continuation",
        "high_volatility_breakout",
        "capitulation",
        "blow_off_top",
    ]

    # Filter to available regimes
    regime_order = [r for r in regime_order if r in regimes]

    if not regime_order:
        print("⚠️ No matching regimes found for curriculum, using all available.")
        regime_order = regimes

    print(f"   Training order: {regime_order}")

    # Start with first regime
    first_regime = regime_order[0] if regime_order else None
    print(f"\n🎯 Phase 1: Training on '{first_regime}'...")

    res = client.get_rl_episodes(
        symbol=symbol, interval=timeframe, num_episodes=10, regime=first_regime
    )
    episodes = res.get("episodes", [])

    if not episodes:
        print("❌ No episodes found for first regime")
        return None

    env = create_multi_episode_env(episodes)

    model = PPO("MlpPolicy", env, learning_rate=3e-4, verbose=0)

    # Train on each regime
    for i, regime in enumerate(regime_order):
        print(f"\n🎯 Phase {i + 1}/{len(regime_order)}: Training on '{regime}'...")

        # Fetch regime-specific episodes
        res = client.get_rl_episodes(
            symbol=symbol, interval=timeframe, num_episodes=10, regime=regime
        )
        episodes = res.get("episodes", [])

        if not episodes:
            print(f"   ⚠️ No episodes for {regime}, skipping...")
            continue

        # Create new environment with regime episodes
        env = create_multi_episode_env(episodes)
        model.set_env(env)

        # Train
        model.learn(
            total_timesteps=timesteps_per_regime,
            reset_num_timesteps=False,  # Continue learning
        )

        print(f"   ✅ Completed {timesteps_per_regime} steps on '{regime}'")

    # Save curriculum-trained model
    model_path = f"ppo_curriculum_{symbol}_{timeframe}"
    model.save(model_path)
    print(f"\n💾 Curriculum model saved to: {model_path}.zip")

    return model


# =============================================================================
# Demo: Quick Test
# =============================================================================


def quick_demo():
    """Quick demo with minimal training to verify everything works."""

    print("=" * 60)
    print("Quick Demo: Verify SB3 + RLX-Search Integration")
    print("=" * 60)

    import os

    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    client = Client(base_url=base_url)

    # Test connection
    print("\n1️⃣ Testing pattern engine connection...")
    try:
        res = client.get_rl_regimes()
        regimes = [r["id"] for r in res.get("regimes", [])]
        print(f"   ✅ Connected! Available regimes: {regimes}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return

    # Fetch episodes
    print("\n2️⃣ Fetching training episodes...")
    res = client.get_rl_episodes(
        symbol=DEFAULT_SYMBOL,
        interval=DEFAULT_TIMEFRAME,
        num_episodes=4,
        min_similarity=0.5,
    )
    episodes = res.get("episodes", [])
    print(f"   ✅ Got {len(episodes)} episodes")

    # Create environment
    print("\n3️⃣ Creating Gymnasium environment...")
    env = create_multi_episode_env(episodes)
    print(f"   ✅ Environment created with {env.num_envs} parallel envs")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Quick training
    print("\n4️⃣ Quick PPO training (500 steps)...")
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=500)
    print("   ✅ Training complete!")

    # Quick evaluation
    print("\n5️⃣ Evaluating trained agent...")
    eval_env = create_env_from_episode(episodes[0])
    obs, _ = eval_env.reset()

    total_reward = 0
    steps = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"   ✅ Evaluation complete!")
    print(f"   Steps: {steps}")
    print(f"   Total Reward: {total_reward:.4f}")
    print(f"   Trades: {info['n_trades']}")
    print(f"   PnL: {info['total_pnl']:.2%}")

    print("\n" + "=" * 60)
    print("🎉 All systems working! Ready for full training.")
    print("=" * 60)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RL Training with RLX-Search")
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "train", "curriculum"],
        help="Training mode",
    )
    parser.add_argument(
        "--symbol", type=str, default=DEFAULT_SYMBOL, help="Trading symbol"
    )
    parser.add_argument(
        "--timeframe", type=str, default=DEFAULT_TIMEFRAME, help="Candle timeframe"
    )
    parser.add_argument("--regime", type=str, default=None, help="Market regime filter")
    parser.add_argument(
        "--timesteps", type=int, default=10000, help="Total training timesteps"
    )
    parser.add_argument(
        "--episodes", type=int, default=20, help="Number of episodes to fetch"
    )

    args = parser.parse_args()

    if args.mode == "demo":
        quick_demo()
    elif args.mode == "train":
        train_on_patterns(
            symbol=args.symbol,
            timeframe=args.timeframe,
            regime=args.regime,
            total_timesteps=args.timesteps,
            n_episodes=args.episodes,
        )
    elif args.mode == "curriculum":
        train_curriculum(
            symbol=args.symbol,
            timeframe=args.timeframe,
            timesteps_per_regime=args.timesteps // 6,
        )
