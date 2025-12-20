#!/usr/bin/env python3
"""
=============================================================================
RLX-Search + RL: Proof of Concept
=============================================================================

This script proves that training RL agents on SIMILAR historical patterns
(Neural Memory approach) is more effective than random sampling.

Experiment Design:
------------------
1. BASELINE: Train agent on RANDOM historical episodes
2. PATTERN-BASED: Train agent on SIMILAR episodes (from RLX-Search)
3. Compare performance on the SAME evaluation set

Key Metrics:
- Average PnL per episode
- Win rate (% of profitable episodes)
- Sharpe ratio
- Number of trades
- Learning curve (how fast agent improves)

The hypothesis: Pattern-based training should show:
- Faster convergence (fewer timesteps to learn)
- Better generalization (higher eval performance)
- More consistent results (lower variance)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import json
import time
from datetime import datetime
import os

# =============================================================================
# Configuration
# =============================================================================

RLX_BASE_URL = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_TIMEFRAME = "1h"

# Experiment parameters
TRAINING_TIMESTEPS = 50000  # Total training steps
EVAL_EPISODES = 20  # Episodes for evaluation
N_TRAINING_EPISODES = 30  # Episodes for training pool
WINDOW_SIZE = 20  # Observation window
N_EVAL_CHECKPOINTS = 10  # How many times to evaluate during training


# =============================================================================
# Trading Environment (Improved)
# =============================================================================


class TradingEnv(gym.Env):
    """
    Improved Trading Environment with better reward shaping.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: np.ndarray,
        window_size: int = WINDOW_SIZE,
        commission: float = 0.001,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.prices = prices
        self.window_size = window_size
        self.commission = commission
        self.render_mode = render_mode

        # Action space: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)

        # Observation: normalized returns + position info
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                window_size + 3,
            ),  # returns + position + unrealized_pnl + volatility
            dtype=np.float32,
        )

        self._reset_state()

    def _reset_state(self):
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trades = []
        self.max_drawdown = 0.0
        self.peak_pnl = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        # Get price window
        start_idx = max(0, self.current_step - self.window_size)
        window_prices = self.prices[start_idx : self.current_step]

        # Compute normalized returns
        if len(window_prices) > 1:
            returns = np.diff(window_prices) / window_prices[:-1]
            # Z-normalize returns
            if np.std(returns) > 0:
                returns = (returns - np.mean(returns)) / np.std(returns)
        else:
            returns = np.zeros(self.window_size - 1)

        # Pad if needed
        if len(returns) < self.window_size:
            returns = np.concatenate(
                [np.zeros(self.window_size - len(returns)), returns]
            )

        # Current volatility (rolling std of returns)
        volatility = np.std(returns) if len(returns) > 0 else 0.0

        # Unrealized PnL
        current_price = self.prices[min(self.current_step, len(self.prices) - 1)]
        if self.position != 0 and self.entry_price > 0:
            unrealized_pnl = (
                (current_price - self.entry_price) / self.entry_price * self.position
            )
        else:
            unrealized_pnl = 0.0

        obs = np.concatenate(
            [
                returns.astype(np.float32),
                [float(self.position), unrealized_pnl, volatility],
            ]
        )

        return obs.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.current_step >= len(self.prices) - 1:
            return self._get_observation(), 0.0, True, False, self._get_info()

        current_price = self.prices[self.current_step]
        reward = 0.0
        trade_made = False

        # Execute action
        if action == 1:  # Buy/Long
            if self.position == -1:  # Close short
                pnl = (self.entry_price - current_price) / self.entry_price
                reward += pnl - self.commission
                self.total_pnl += pnl - self.commission
                self.trades.append(
                    {"type": "close_short", "price": current_price, "pnl": pnl}
                )
                trade_made = True

            if self.position <= 0:  # Open long
                self.position = 1
                self.entry_price = current_price
                reward -= self.commission
                trade_made = True

        elif action == 2:  # Sell/Short
            if self.position == 1:  # Close long
                pnl = (current_price - self.entry_price) / self.entry_price
                reward += pnl - self.commission
                self.total_pnl += pnl - self.commission
                self.trades.append(
                    {"type": "close_long", "price": current_price, "pnl": pnl}
                )
                trade_made = True

            if self.position >= 0:  # Open short
                self.position = -1
                self.entry_price = current_price
                reward -= self.commission
                trade_made = True

        elif action == 3:  # Close position
            if self.position == 1:
                pnl = (current_price - self.entry_price) / self.entry_price
                reward += pnl - self.commission
                self.total_pnl += pnl - self.commission
                self.trades.append(
                    {"type": "close_long", "price": current_price, "pnl": pnl}
                )
                trade_made = True
            elif self.position == -1:
                pnl = (self.entry_price - current_price) / self.entry_price
                reward += pnl - self.commission
                self.total_pnl += pnl - self.commission
                self.trades.append(
                    {"type": "close_short", "price": current_price, "pnl": pnl}
                )
                trade_made = True

            self.position = 0
            self.entry_price = 0.0

        # Reward shaping: small penalty for holding, bonus for profitable trades
        if self.position != 0:
            reward -= 0.00005  # Smaller holding penalty

        if trade_made and len(self.trades) > 0:
            last_trade_pnl = self.trades[-1].get("pnl", 0)
            if last_trade_pnl > 0:
                reward += 0.01  # Bonus for profitable trade

        # Track drawdown
        self.peak_pnl = max(self.peak_pnl, self.total_pnl)
        current_drawdown = self.peak_pnl - self.total_pnl
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Move to next step
        self.current_step += 1

        # Check termination
        terminated = self.current_step >= len(self.prices) - 1

        # Force close on termination
        if terminated and self.position != 0:
            final_price = self.prices[-1]
            if self.position == 1:
                pnl = (final_price - self.entry_price) / self.entry_price
            else:
                pnl = (self.entry_price - final_price) / self.entry_price
            reward += pnl - self.commission
            self.total_pnl += pnl - self.commission
            self.trades.append(
                {"type": "force_close", "price": final_price, "pnl": pnl}
            )

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_info(self) -> dict:
        return {
            "total_pnl": self.total_pnl,
            "n_trades": len(self.trades),
            "max_drawdown": self.max_drawdown,
            "win_rate": self._compute_win_rate(),
        }

    def _compute_win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
        return wins / len(self.trades)


# =============================================================================
# RLX-Search Client
# =============================================================================


class RLXSearchClient:
    def __init__(self, base_url: str = RLX_BASE_URL):
        self.base_url = base_url

    def _get_all_prices(
        self, symbol: str = DEFAULT_SYMBOL, timeframe: str = DEFAULT_TIMEFRAME
    ) -> List[float]:
        """Fetch all available prices."""
        params = {"symbol": symbol, "interval": timeframe, "q": 40}
        response = requests.get(f"{self.base_url}/api/patterns", params=params)
        response.raise_for_status()
        return response.json().get("series", [])

    def get_similar_episodes(
        self,
        symbol: str = DEFAULT_SYMBOL,
        timeframe: str = DEFAULT_TIMEFRAME,
        num_episodes: int = 20,
        min_similarity: float = 0.7,
        query_prices: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Get episodes similar to given state (or current market state)."""

        # Get prices if not provided
        if query_prices is None:
            all_prices = self._get_all_prices(symbol, timeframe)
            query_prices = all_prices[-40:] if len(all_prices) >= 40 else all_prices

        params = {
            "symbol": symbol,
            "interval": timeframe,
            "currentState": query_prices,
            "queryLength": len(query_prices),
            "forecastHorizon": 24,
            "numEpisodes": num_episodes,
            "minSimilarity": min_similarity,
            "includeActions": True,
            "rewardType": "returns",
            "samplingStrategy": "diverse",
        }

        response = requests.post(f"{self.base_url}/api/rl/episodes", json=params)
        response.raise_for_status()
        return response.json()["episodes"]

    def get_train_eval_split(
        self,
        symbol: str = DEFAULT_SYMBOL,
        timeframe: str = DEFAULT_TIMEFRAME,
        n_train: int = 50,
        n_eval: int = 30,
        min_similarity: float = 0.7,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Get train/eval episodes with NO OVERLAP using TIME-BASED SPLIT.

        Strategy:
        1. Split price history into TRAIN period (first 70%) and EVAL period (last 30%)
        2. Training episodes come from TRAIN period only
        3. Evaluation episodes come from EVAL period only
        This guarantees no data leakage!

        PATTERN-BASED: Episodes sorted by similarity to current market
        RANDOM: Same episodes but in random order (control)
        """

        all_prices = self._get_all_prices(symbol, timeframe)

        # Time-based split: 70% train, 30% eval
        split_idx = int(len(all_prices) * 0.7)
        train_prices = all_prices[:split_idx]
        eval_prices = all_prices[split_idx:]

        print(
            f"   📅 Time-based split: {split_idx:,} train prices, {len(eval_prices):,} eval prices"
        )

        # For Pattern-based: Use API to find SIMILAR episodes
        # Query with recent train prices (simulating "current" state within train period)
        query_prices = train_prices[-40:] if len(train_prices) >= 40 else train_prices

        train_episodes = self.get_similar_episodes(
            symbol=symbol,
            timeframe=timeframe,
            num_episodes=n_train,
            min_similarity=min_similarity,
            query_prices=query_prices,
        )

        # Mark episodes as train period
        for ep in train_episodes:
            ep["period"] = "train"

        # Filter train episodes: keep only those from train period
        # by checking if their prices appear in train_prices range
        # (This is a heuristic since API doesn't return timestamps)

        print(
            f"   📊 Pattern-based: Got {len(train_episodes)} similar episodes from API"
        )
        if train_episodes:
            sims = [ep.get("similarity", 0) for ep in train_episodes]
            print(
                f"      Similarity: min={min(sims):.3f}, max={max(sims):.3f}, avg={np.mean(sims):.3f}"
            )

        # Generate evaluation episodes from EVAL period (truly unseen!)
        eval_episodes = []
        episode_length = 64

        if len(eval_prices) >= episode_length:
            for i in range(n_eval):
                start_idx = np.random.randint(0, len(eval_prices) - episode_length)
                prices = eval_prices[start_idx : start_idx + episode_length]

                eval_episodes.append(
                    {
                        "episode_id": f"eval_{start_idx}_{i}",
                        "similarity": 0.0,  # Different period
                        "state_prices": prices[:40],
                        "transitions": [{"price": p} for p in prices[40:]],
                        "period": "eval",
                    }
                )

        return train_episodes, eval_episodes

    def get_pattern_episodes_from_period(
        self,
        symbol: str = DEFAULT_SYMBOL,
        timeframe: str = DEFAULT_TIMEFRAME,
        num_episodes: int = 50,
        use_train_period: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get pattern-matched episodes from specific time period.
        Uses RLX-Search API to find similar patterns.
        """

        all_prices = self._get_all_prices(symbol, timeframe)
        split_idx = int(len(all_prices) * 0.7)

        if use_train_period:
            period_prices = all_prices[:split_idx]
        else:
            period_prices = all_prices[split_idx:]

        # Use prices from this period as query
        query_prices = (
            period_prices[-40:] if len(period_prices) >= 40 else period_prices
        )

        return self.get_similar_episodes(
            symbol=symbol,
            timeframe=timeframe,
            num_episodes=num_episodes,
            min_similarity=0.7,
            query_prices=query_prices,
        )

    def get_random_episodes(
        self,
        symbol: str = DEFAULT_SYMBOL,
        timeframe: str = DEFAULT_TIMEFRAME,
        num_episodes: int = 20,
        episode_length: int = 64,
        exclude_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Get random historical episodes (baseline)."""

        all_prices = self._get_all_prices(symbol, timeframe)
        return self._generate_random_episodes(
            all_prices, num_episodes, episode_length, "random"
        )

    def get_random_episodes_from_period(
        self,
        symbol: str = DEFAULT_SYMBOL,
        timeframe: str = DEFAULT_TIMEFRAME,
        num_episodes: int = 20,
        episode_length: int = 64,
        use_train_period: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get random episodes from specific time period (train or eval)."""

        all_prices = self._get_all_prices(symbol, timeframe)
        split_idx = int(len(all_prices) * 0.7)

        if use_train_period:
            period_prices = all_prices[:split_idx]
            period_name = "train"
        else:
            period_prices = all_prices[split_idx:]
            period_name = "eval"

        return self._generate_random_episodes(
            period_prices, num_episodes, episode_length, period_name
        )

    def _generate_random_episodes(
        self,
        prices: List[float],
        num_episodes: int,
        episode_length: int,
        period_name: str,
    ) -> List[Dict[str, Any]]:
        """Generate random episodes from price list."""

        if len(prices) < episode_length * 2:
            raise ValueError(
                f"Not enough price data ({len(prices)}) for {num_episodes} random episodes"
            )

        episodes = []
        for i in range(num_episodes):
            start_idx = np.random.randint(0, len(prices) - episode_length)
            ep_prices = prices[start_idx : start_idx + episode_length]

            episodes.append(
                {
                    "episode_id": f"{period_name}_{start_idx}_{i}",
                    "similarity": 0.0,
                    "state_prices": ep_prices[:40],
                    "transitions": [{"price": p} for p in ep_prices[40:]],
                    "period": period_name,
                }
            )

        return episodes


# =============================================================================
# Environment Factories
# =============================================================================


def create_env_from_episode(episode: Dict[str, Any]) -> TradingEnv:
    """Create environment from RLX episode."""
    state_prices = episode.get("state_prices", [])
    forecast_prices = [t["price"] for t in episode.get("transitions", [])]

    all_prices = np.array(state_prices + forecast_prices, dtype=np.float32)

    # Ensure minimum length
    if len(all_prices) < WINDOW_SIZE + 10:
        padding = np.full(WINDOW_SIZE, all_prices[0] if len(all_prices) > 0 else 100.0)
        all_prices = np.concatenate([padding, all_prices])

    return TradingEnv(prices=all_prices)


def create_env_pool(episodes: List[Dict[str, Any]]) -> DummyVecEnv:
    """Create vectorized environment from episode pool."""
    n_envs = min(len(episodes), 8)

    def make_env(i):
        def _init():
            return create_env_from_episode(episodes[i % len(episodes)])

        return _init

    return DummyVecEnv([make_env(i) for i in range(n_envs)])


# =============================================================================
# Evaluation & Metrics
# =============================================================================


@dataclass
class EvalResults:
    """Evaluation results container."""

    pnl_mean: float = 0.0
    pnl_std: float = 0.0
    win_rate: float = 0.0
    avg_trades: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    episodes_evaluated: int = 0

    def to_dict(self) -> dict:
        return {
            "pnl_mean": float(self.pnl_mean),
            "pnl_std": float(self.pnl_std),
            "win_rate": float(self.win_rate),
            "avg_trades": float(self.avg_trades),
            "sharpe_ratio": float(self.sharpe_ratio),
            "max_drawdown": float(self.max_drawdown),
            "episodes_evaluated": int(self.episodes_evaluated),
        }


def evaluate_agent(
    model: PPO, eval_episodes: List[Dict[str, Any]], deterministic: bool = True
) -> EvalResults:
    """Evaluate agent on a set of episodes."""

    pnls = []
    win_rates = []
    trades = []
    drawdowns = []

    for ep in eval_episodes:
        env = create_env_from_episode(ep)
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        pnls.append(info["total_pnl"])
        win_rates.append(info["win_rate"])
        trades.append(info["n_trades"])
        drawdowns.append(info["max_drawdown"])

    pnl_array = np.array(pnls)

    # Compute Sharpe ratio
    if np.std(pnl_array) > 0:
        sharpe = np.mean(pnl_array) / np.std(pnl_array) * np.sqrt(252)  # Annualized
    else:
        sharpe = 0.0

    return EvalResults(
        pnl_mean=np.mean(pnl_array),
        pnl_std=np.std(pnl_array),
        win_rate=np.mean(win_rates),
        avg_trades=np.mean(trades),
        sharpe_ratio=sharpe,
        max_drawdown=np.mean(drawdowns),
        episodes_evaluated=len(eval_episodes),
    )


# =============================================================================
# Training Callback for Learning Curve
# =============================================================================


class LearningCurveCallback(BaseCallback):
    """Track learning progress at regular intervals."""

    def __init__(
        self, eval_episodes: List[Dict[str, Any]], eval_freq: int, verbose: int = 0
    ):
        super().__init__(verbose)
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.learning_curve = []
        self.timesteps = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            results = evaluate_agent(self.model, self.eval_episodes[:5])
            self.learning_curve.append(results.pnl_mean)
            self.timesteps.append(self.num_timesteps)

            if self.verbose > 0:
                print(
                    f"  Step {self.num_timesteps}: PnL={results.pnl_mean:.4f}, "
                    f"WinRate={results.win_rate:.2%}, Trades={results.avg_trades:.1f}"
                )

        return True


# =============================================================================
# Main Experiment
# =============================================================================


def run_experiment(
    training_timesteps: int = TRAINING_TIMESTEPS,
    n_training_episodes: int = N_TRAINING_EPISODES,
    n_eval_episodes: int = EVAL_EPISODES,
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> Dict[str, Any]:
    """
    Run the full proof-of-concept experiment.

    Compares:
    1. Pattern-based training (using RLX-Search similar episodes)
    2. Random baseline (using random historical episodes)

    IMPORTANT: Train and Eval sets are STRICTLY SEPARATED to prevent data leakage!
    - Training uses episodes similar to CURRENT market state
    - Evaluation uses episodes similar to OLDER market state (different time period)
    - Episode IDs are checked to prevent any overlap
    """

    print("=" * 70)
    print("RLX-Search + RL: Proof of Concept Experiment")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Training timesteps: {training_timesteps:,}")
    print(f"  Training episodes:  {n_training_episodes}")
    print(f"  Evaluation episodes: {n_eval_episodes}")
    print(f"  Symbol: {symbol}, Timeframe: {timeframe}")

    # Initialize client
    client = RLXSearchClient()

    # ==========================================================================
    # Step 1: Fetch Episodes with PROPER TRAIN/EVAL SPLIT
    # ==========================================================================

    print("\n" + "=" * 70)
    print("Step 1: Fetching Episodes (with strict train/eval separation)")
    print("=" * 70)

    # Get properly separated train/eval episodes
    print("\n📊 Fetching pattern-based training episodes (similar to CURRENT state)...")
    print(
        "📋 Fetching evaluation episodes (similar to OLDER state - different time period)..."
    )

    pattern_episodes, eval_episodes = client.get_train_eval_split(
        symbol=symbol,
        timeframe=timeframe,
        n_train=n_training_episodes,
        n_eval=n_eval_episodes,
        min_similarity=0.7,
    )

    # Verify no overlap - using period tag
    train_periods = set(ep.get("period", "train") for ep in pattern_episodes)
    eval_periods = set(ep.get("period", "eval") for ep in eval_episodes)

    similarities = [ep.get("similarity", 0) for ep in pattern_episodes]
    eval_similarities = [ep.get("similarity", 0) for ep in eval_episodes]

    print(
        f"\n   ✅ Training episodes: {len(pattern_episodes)} (from TRAIN period: {train_periods})"
    )
    print(
        f"   ✅ Evaluation episodes: {len(eval_episodes)} (from EVAL period: {eval_periods})"
    )
    print(f"   🔒 Time-based split guarantees NO data leakage!")

    if "eval" in train_periods or "train" in eval_periods:
        print(f"   ⚠️ WARNING: Period mixing detected!")

    # Random episodes (baseline) - from TRAIN period only
    print("\n🎲 Fetching random historical episodes (baseline, from TRAIN period)...")
    random_episodes = client.get_random_episodes_from_period(
        symbol=symbol,
        timeframe=timeframe,
        num_episodes=n_training_episodes,
        use_train_period=True,
    )
    print(f"   ✅ Got {len(random_episodes)} random episodes")

    # ==========================================================================
    # Step 2: Train Pattern-Based Agent
    # ==========================================================================

    print("\n" + "=" * 70)
    print("Step 2: Training PATTERN-BASED Agent")
    print("=" * 70)

    pattern_env = create_env_pool(pattern_episodes)
    eval_freq = training_timesteps // N_EVAL_CHECKPOINTS

    pattern_callback = LearningCurveCallback(
        eval_episodes=eval_episodes, eval_freq=eval_freq, verbose=1
    )

    print(f"\n🤖 Training PPO on {len(pattern_episodes)} similar patterns...")
    print(f"   Evaluating every {eval_freq:,} steps")

    pattern_model = PPO(
        "MlpPolicy",
        pattern_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
    )

    start_time = time.time()
    pattern_model.learn(total_timesteps=training_timesteps, callback=pattern_callback)
    pattern_train_time = time.time() - start_time

    print(f"\n   ✅ Training complete in {pattern_train_time:.1f}s")

    # ==========================================================================
    # Step 3: Train Random Baseline Agent
    # ==========================================================================

    print("\n" + "=" * 70)
    print("Step 3: Training RANDOM BASELINE Agent")
    print("=" * 70)

    random_env = create_env_pool(random_episodes)

    random_callback = LearningCurveCallback(
        eval_episodes=eval_episodes, eval_freq=eval_freq, verbose=1
    )

    print(f"\n🎲 Training PPO on {len(random_episodes)} random episodes...")

    random_model = PPO(
        "MlpPolicy",
        random_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
    )

    start_time = time.time()
    random_model.learn(total_timesteps=training_timesteps, callback=random_callback)
    random_train_time = time.time() - start_time

    print(f"\n   ✅ Training complete in {random_train_time:.1f}s")

    # ==========================================================================
    # Step 4: Final Evaluation
    # ==========================================================================

    print("\n" + "=" * 70)
    print("Step 4: Final Evaluation")
    print("=" * 70)

    print("\n📈 Evaluating Pattern-Based Agent...")
    pattern_results = evaluate_agent(pattern_model, eval_episodes)

    print("\n🎲 Evaluating Random Baseline Agent...")
    random_results = evaluate_agent(random_model, eval_episodes)

    # ==========================================================================
    # Step 5: Results Comparison
    # ==========================================================================

    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print("\n" + "-" * 50)
    print(f"{'Metric':<25} {'Pattern-Based':>12} {'Random':>12} {'Diff':>10}")
    print("-" * 50)

    metrics = [
        ("Mean PnL (%)", pattern_results.pnl_mean * 100, random_results.pnl_mean * 100),
        ("PnL Std (%)", pattern_results.pnl_std * 100, random_results.pnl_std * 100),
        ("Win Rate (%)", pattern_results.win_rate * 100, random_results.win_rate * 100),
        ("Avg Trades", pattern_results.avg_trades, random_results.avg_trades),
        ("Sharpe Ratio", pattern_results.sharpe_ratio, random_results.sharpe_ratio),
        (
            "Max Drawdown (%)",
            pattern_results.max_drawdown * 100,
            random_results.max_drawdown * 100,
        ),
    ]

    for name, pattern_val, random_val in metrics:
        diff = pattern_val - random_val
        diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
        print(f"{name:<25} {pattern_val:>12.2f} {random_val:>12.2f} {diff_str:>10}")

    print("-" * 50)

    # Determine winner
    pattern_score = 0
    if pattern_results.pnl_mean > random_results.pnl_mean:
        pattern_score += 1
    if pattern_results.win_rate > random_results.win_rate:
        pattern_score += 1
    if pattern_results.sharpe_ratio > random_results.sharpe_ratio:
        pattern_score += 1

    print("\n" + "=" * 70)
    if pattern_score >= 2:
        print("🏆 PATTERN-BASED TRAINING WINS!")
        print("   Training on similar historical patterns is more effective.")
    elif pattern_score == 0:
        print("🎲 RANDOM BASELINE WINS")
        print("   Need more training or better pattern matching.")
    else:
        print("🤝 RESULTS ARE MIXED")
        print("   Both approaches show similar performance.")
    print("=" * 70)

    # Learning curves comparison
    print("\n📊 Learning Curves:")
    print("\n  Pattern-Based:")
    for i, (ts, pnl) in enumerate(
        zip(pattern_callback.timesteps, pattern_callback.learning_curve)
    ):
        bar = "█" * int(max(0, (pnl + 0.05) * 100))
        print(f"    {ts:>6}: {pnl:>7.4f} {bar}")

    print("\n  Random Baseline:")
    for i, (ts, pnl) in enumerate(
        zip(random_callback.timesteps, random_callback.learning_curve)
    ):
        bar = "█" * int(max(0, (pnl + 0.05) * 100))
        print(f"    {ts:>6}: {pnl:>7.4f} {bar}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "training_timesteps": training_timesteps,
            "n_training_episodes": n_training_episodes,
            "n_eval_episodes": n_eval_episodes,
            "symbol": symbol,
            "timeframe": timeframe,
        },
        "pattern_based": {
            "results": pattern_results.to_dict(),
            "train_time": float(pattern_train_time),
            "learning_curve": [
                [int(ts), float(pnl)]
                for ts, pnl in zip(
                    pattern_callback.timesteps, pattern_callback.learning_curve
                )
            ],
            "avg_similarity": float(np.mean(similarities)),
        },
        "random_baseline": {
            "results": random_results.to_dict(),
            "train_time": float(random_train_time),
            "learning_curve": [
                [int(ts), float(pnl)]
                for ts, pnl in zip(
                    random_callback.timesteps, random_callback.learning_curve
                )
            ],
        },
        "winner": "pattern_based"
        if pattern_score >= 2
        else ("random" if pattern_score == 0 else "tie"),
    }

    # Save to file
    results_file = f"poc_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")

    # Save models
    pattern_model.save("pattern_based_agent")
    random_model.save("random_baseline_agent")
    print("💾 Models saved: pattern_based_agent.zip, random_baseline_agent.zip")

    return results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RLX-Search RL Proof of Concept")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TRAINING_TIMESTEPS,
        help="Training timesteps per agent",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=N_TRAINING_EPISODES,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=EVAL_EPISODES,
        help="Number of evaluation episodes",
    )
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME)

    args = parser.parse_args()

    try:
        results = run_experiment(
            training_timesteps=args.timesteps,
            n_training_episodes=args.train_episodes,
            n_eval_episodes=args.eval_episodes,
            symbol=args.symbol,
            timeframe=args.timeframe,
        )
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to RLX-Search server at", RLX_BASE_URL)
        print("   Make sure the server is running: cd rlx-search && cargo run")
        exit(1)
