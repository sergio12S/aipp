#!/usr/bin/env python3
"""
Simple RL Training using aipricepatterns SDK
One decision per episode: LONG, SHORT, or HOLD
"""

import os
import numpy as np
from collections import defaultdict
from aipricepatterns import Client

# Configuration
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
QUERY_LENGTH = 40
FORECAST_HORIZON = 24

# Training config
NUM_EPOCHS = 50
EPISODES_PER_EPOCH = 200
MIN_SIMILARITY = 0.80
LEARNING_RATE = 0.001

# Time-based split for train/eval (prevent data leakage)
TRAIN_END_TS = 1719792000000  # July 1, 2024


class SimpleQAgent:
    def __init__(self, state_dim: int, n_actions: int = 3, lr: float = 0.01):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = lr
        self.weights = np.random.randn(n_actions, state_dim) * 0.01
        self.biases = np.zeros(n_actions)
        self.epsilon = 0.5
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.action_counts = defaultdict(int)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        return self.weights @ state + self.biases

    def predict(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            q_values = self.get_q_values(state)
            action = np.argmax(q_values)
        self.action_counts[action] += 1
        return action

    def learn(self, state: np.ndarray, action: int, reward: float):
        q_values = self.get_q_values(state)
        error = reward - q_values[action]
        self.weights[action] += self.lr * error * state
        self.biases[action] += self.lr * error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def main():
    base_url = os.getenv("AIPP_BASE_URL", "https://aipricepatterns.com/api/rust")
    client = Client(base_url=base_url)

    print(f"Fetching training episodes for {SYMBOL} {INTERVAL}...")
    # Use the SDK to fetch episodes
    try:
        resp = client.get_rl_simple(
            symbol=SYMBOL,
            interval=INTERVAL,
            timestamp=TRAIN_END_TS,
            forecast_horizon=FORECAST_HORIZON,
            num_episodes=1000,
            min_similarity=MIN_SIMILARITY,
        )
        episodes = resp.get("episodes", [])
    except Exception as e:
        print(f"Error fetching episodes: {e}")
        return

    if not episodes:
        print("No episodes found. Check if dataset is loaded.")
        return

    print(f"Loaded {len(episodes)} episodes. Starting training...")

    agent = SimpleQAgent(state_dim=QUERY_LENGTH)

    for epoch in range(NUM_EPOCHS):
        epoch_rewards = []
        # Shuffle episodes
        np.random.shuffle(episodes)

        for ep in episodes[:EPISODES_PER_EPOCH]:
            state = np.array(ep["observation"])
            action = agent.predict(state)

            # Reward logic
            if action == 1:  # LONG
                reward = ep["rewardLong"]
            elif action == 2:  # SHORT
                reward = ep["rewardShort"]
            else:  # HOLD
                reward = 0

            agent.learn(state, action, reward)
            epoch_rewards.append(reward)

        agent.decay_epsilon()
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | Avg Reward: {np.mean(epoch_rewards):.4f} | Epsilon: {agent.epsilon:.2f}"
            )

    print("\nTraining complete.")
    print(f"Action distribution: {dict(agent.action_counts)}")


if __name__ == "__main__":
    main()
