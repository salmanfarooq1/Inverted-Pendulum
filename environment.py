import gym
from gym.wrappers import RecordVideo

class InvertedPendulumEnv:
    def __init__(self, env_name="CartPole-v1", video_path="videos/"):
        """
        Initialize the environment.

        :param env_name: Name of the Gym environment.
        :param video_path: Path to save the video recordings of the simulation.
        """
        self.env_name = env_name
        self.video_path = video_path
        self.env = gym.make(self.env_name)

        # Wrap the environment for video recording
        self.env = RecordVideo(self.env, self.video_path, episode_trigger=lambda episode_id: episode_id % 10 == 0)

    def reset(self):
        """
        Reset the environment to its initial state.

        :return: Initial observation.
        """
        return self.env.reset()

    def step(self, action):
        """
        Take a step in the environment using the given action.

        :param action: The action to take.
        :return: A tuple (observation, reward, done, info).
        """
        return self.env.step(action)

    def render(self):
        """
        Render the environment.
        """
        self.env.render()

    def close(self):
        """
        Close the environment.
        """
        self.env.close()

# Test the environment setup
if __name__ == "__main__":
    env_wrapper = InvertedPendulumEnv()
    print(env_wrapper.env.observation_space, env_wrapper.env.action_space)
    env_wrapper.close()
