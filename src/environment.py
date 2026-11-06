import gymnasium as gym

from src.reward_shaping import RewardShapingWrapper


class LunarPilotEnvironment:
    def __init__(self, render_mode=None, continuous=True):
        env_id = "LunarLanderContinuous-v3" if continuous else "LunarLander-v3"
        base_env = gym.make(
            env_id,
            render_mode=render_mode
        )
        self.env = RewardShapingWrapper(base_env)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def close(self):
        self.env.close()


def make_env(render_mode=None, continuous=True):
    env_id = "LunarLanderContinuous-v3" if continuous else "LunarLander-v3"
    base_env = gym.make(
        env_id,
        render_mode=render_mode
    )
    return RewardShapingWrapper(base_env)
