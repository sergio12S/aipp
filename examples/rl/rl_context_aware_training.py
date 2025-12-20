import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# ==========================================
# 1. Define the Environment
# ==========================================
class ContextAwareTradingEnv(gym.Env):
    """
    A trading environment that resets to a specific historical episode
    structurally similar to our target market condition.
    """

    def __init__(self, episodes_data):
        super(ContextAwareTradingEnv, self).__init__()
        self.episodes = episodes_data
        self.current_episode = None
        self.current_transitions = []
        self.step_idx = 0
        self.max_steps = 24

        # Actions: 0=Hold, 1=Long, 2=Short
        self.action_space = spaces.Discrete(3)

        # Observation: [Price (normalized), Volatility, Time_Left]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly select one of the similar historical episodes
        self.current_episode = np.random.choice(self.episodes)
        self.current_transitions = self.current_episode.get("transitions", [])
        self.step_idx = 0

        return self._get_obs(), {}

    def step(self, action):
        if self.step_idx >= len(self.current_transitions):
            return self._get_obs(), 0.0, True, False, {}

        # Get data for the current step
        step_data = self.current_transitions[self.step_idx]

        # Calculate Reward
        step_return = step_data.get("ret", 0.0)

        reward = 0.0
        if action == 1:  # Long
            reward = step_return
        elif action == 2:  # Short
            reward = -step_return

        self.step_idx += 1
        terminated = self.step_idx >= len(self.current_transitions)

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        # If we are at the start, use initial state
        if self.step_idx == 0:
            initial_state = self.current_episode.get("initialState", {})
            features = initial_state.get("features", {})
            prices = initial_state.get("prices", [])

            price = prices[-1] if prices else 0.0
            volatility = features.get("volatility", 0.0)
        else:
            # Use data from the PREVIOUS transition (the state we just arrived at)
            prev_step_data = self.current_transitions[self.step_idx - 1]
            price = prev_step_data.get("price", 0.0)
            volatility = prev_step_data.get("volatility", 0.0)

        time_left = 1.0 - (self.step_idx / self.max_steps)

        return np.array([price, volatility, time_left], dtype=np.float32)


# ==========================================
# 2. Fetch Training Data
# ==========================================
def fetch_training_data(anchor_ts):
    url = "https://aipricepatterns.com/api/rust/api/rl/episodes"
    payload = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "anchorTs": anchor_ts,
        "forecastHorizon": 24,
        "numEpisodes": 50,
        "minSimilarity": 0.60,
    }

    print(f"Fetching episodes similar to {anchor_ts}...")
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"Successfully fetched {len(data['episodes'])} episodes.")
        return data["episodes"]
    except Exception as e:
        print(f"Error fetching data: {e}")
        # Return dummy data for demonstration if API fails
        print("Using DUMMY data for demonstration.")
        return [
            {
                "initialState": {
                    "prices": np.random.randn(40).tolist(),
                    "features": {"volatility": 0.02},
                },
                "transitions": [
                    {
                        "price": 100.0 + i,
                        "ret": 0.01 if i % 2 == 0 else -0.01,
                        "volatility": 0.02,
                    }
                    for i in range(24)
                ],
            }
            for _ in range(50)
        ]


# ==========================================
# 3. Main Training Loop
# ==========================================
def main():
    # SVB Crisis Timestamp (approx March 10, 2023)
    ANCHOR_TS = 1678406400000

    # 1. Get Data
    episodes = fetch_training_data(ANCHOR_TS)

    if not episodes:
        print("No episodes found. Exiting.")
        return

    # 2. Create Environment
    env = ContextAwareTradingEnv(episodes)
    env = DummyVecEnv([lambda: env])

    # 3. Initialize Agent (PPO)
    print("\nInitializing PPO Agent...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)

    # 4. Train
    print("Training on similar historical episodes...")
    model.learn(total_timesteps=5000)
    print("Training complete.")

    # 5. Evaluate
    print("\nEvaluating Agent Strategy:")
    obs = env.reset()
    total_rewards = []
    actions = []

    for _ in range(20):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        total_rewards.append(rewards[0])
        actions.append(action[0])

    avg_reward = np.mean(total_rewards)
    action_counts = {0: 0, 1: 0, 2: 0}
    for a in actions:
        action_counts[a] += 1

    print(f"Average Reward per Episode: {avg_reward:.4f}")
    print(
        f"Action Distribution: Hold={action_counts[0]}, Long={action_counts[1]}, Short={action_counts[2]}"
    )

    if avg_reward > 0:
        print("\nSUCCESS: Agent found a profitable strategy for this market regime!")
    else:
        print("\nRESULT: Market regime is difficult or neutral.")


if __name__ == "__main__":
    main()
