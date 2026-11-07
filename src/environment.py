import gymnasium as gym

from src.reward_shaping import RewardShapingWrapper


class LunarPilotEnvironment:
    def __init__(
        self,
        render_mode=None,
        continuous=True,
        enable_avionics=False,
        window_scale=2.0,
        window_width=None,
        window_height=None,
        enable_hud=True,
        **avionics_kwargs,
    ):
        env_id = "LunarLanderContinuous-v3" if continuous else "LunarLander-v3"
        base_env = gym.make(env_id, render_mode=render_mode)
        self.env = RewardShapingWrapper(base_env)
        if enable_avionics:
            from src.avionics import AvionicsSuite

            self.env = AvionicsSuite(self.env, **avionics_kwargs)
        if render_mode == "human":
            from src.display import ScaledRenderWrapper

            self.env = ScaledRenderWrapper(
                self.env,
                scale=window_scale,
                width=window_width,
                height=window_height,
                enable_hud=enable_hud,
            )

    def reset(self):
        return self.env.reset()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def close(self):
        self.env.close()


def make_env(
    render_mode=None,
    continuous=True,
    enable_avionics=False,
    window_scale=2.0,
    window_width=None,
    window_height=None,
    enable_hud=True,
    **avionics_kwargs,
):
    """
    Creates the LunarPilot environment.
    If enable_avionics=True, wraps with the complete 5-subsystem Aerospace Avionics Suite.
    If render_mode='human', wraps with ScaledRenderWrapper for a 1200x800 resizable
    Pygame window and avionics telemetry HUD.
    """
    env_id = "LunarLanderContinuous-v3" if continuous else "LunarLander-v3"
    base_env = gym.make(env_id, render_mode=render_mode)
    env = RewardShapingWrapper(base_env)
    if enable_avionics:
        from src.avionics import AvionicsSuite

        env = AvionicsSuite(env, **avionics_kwargs)
    if render_mode == "human":
        from src.display import ScaledRenderWrapper

        env = ScaledRenderWrapper(
            env,
            scale=window_scale,
            width=window_width,
            height=window_height,
            enable_hud=enable_hud,
        )
    return env
