
import gymnasium as gym

from src.reward_shaping import RewardShapingWrapper


class LunarPilotEnvironment:

    def __init__(self, render_mode=None):

        base_env = gym.make(
            "LunarLander-v3",
            render_mode=render_mode
        )

        self.env = RewardShapingWrapper(base_env)

    def reset(self):
        observation, info = self.env.reset()
        return observation, info

    def step(self, action):

        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        done = terminated or truncated

        return observation, reward, done, info

    def close(self):
        self.env.close()


def make_env(render_mode=None):

    base_env = gym.make(
        "LunarLander-v3",
        render_mode=render_mode
    )

    env = RewardShapingWrapper(base_env)

    return env
