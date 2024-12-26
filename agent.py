import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class RLAgent:
    def __init__(self, env_name="CartPole-v1"):
        from stable_baselines3 import PPO
        from gymnasium import make

        self.env = make(env_name)
        self.model_path = "models/ppo_cartpole"  # Default path
        self.model = None

        # Load or initialize model
        if os.path.exists(self.model_path + ".zip"):
            print("Loading pre-trained model...")
            self.model = PPO.load(self.model_path, env=self.env)
        else:
            print("No pre-trained model found. Initializing a new PPO model...")
            self.model = PPO("MlpPolicy", self.env, verbose=1)

    def train(self, timesteps):
        self.model.learn(total_timesteps=timesteps)

    def save_model(self, path=None):
        if path is None:
            path = self.model_path  # Use default path if none provided
        print(f"Saving model at {path}")
        self.model.save(path)

    def evaluate(self, episodes=5):
        total_rewards = []
        for ep in range(episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
            total_rewards.append(episode_reward)
            print(f"Episode {ep + 1}: Total Reward = {episode_reward}")
        return total_rewards
