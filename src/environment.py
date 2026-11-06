import gymnasium as gym

from src.reward_shaping import RewardShapingWrapper


class LunarPilotEnvironment:
    def __init__(self, render_mode=None, continuous=True, enable_avionics=False, **avionics_kwargs):
        env_id = "LunarLanderContinuous-v3" if continuous else "LunarLander-v3"
        base_env = gym.make(
            env_id,
            render_mode=render_mode
        )
        self.env = RewardShapingWrapper(base_env)
        if enable_avionics:
            from src.avionics import AvionicsSuite
            self.env = AvionicsSuite(self.env, **avionics_kwargs)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def close(self):
        self.env.close()


def make_env(render_mode=None, continuous=True, enable_avionics=False, **avionics_kwargs):
    """
    Creates the LunarPilot environment.
    If enable_avionics=True, wraps with the complete 5-subsystem Aerospace Avionics Suite.
    """
    env_id = "LunarLanderContinuous-v3" if continuous else "LunarLander-v3"
    base_env = gym.make(
        env_id,
        render_mode=render_mode
    )
    env = RewardShapingWrapper(base_env)
    if enable_avionics:
        from src.avionics import AvionicsSuite
        env = AvionicsSuite(env, **avionics_kwargs)
    return env
