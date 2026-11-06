import numpy as np
import gymnasium as gym


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, gamma=0.99):
        super().__init__(env)
        self.gamma = gamma
        self.previous_potential = 0.0

    def calculate_potential(self, observation):
        x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = observation

        # Strong pull towards the landing pad center (x=0) between the flags (-0.2 <= x <= 0.2)
        horizontal_penalty = -25.0 * abs(x)
        vertical_penalty = -10.0 * abs(y)
        speed_penalty = -5.0 * np.sqrt(vx**2 + vy**2)
        tilt_penalty = -5.0 * abs(angle)
        leg_bonus = 10.0 * (left_leg + right_leg)

        return (
            horizontal_penalty
            + vertical_penalty
            + speed_penalty
            + tilt_penalty
            + leg_bonus
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_potential = self.calculate_potential(observation)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        current_potential = self.calculate_potential(observation)
        shaping_reward = self.gamma * current_potential - self.previous_potential

        x = observation[0]
        vx = observation[2]
        vy = observation[3]
        left_leg = observation[6]
        right_leg = observation[7]

        # Contact light / Engine cutoff: If both legs make contact and velocity is zero,
        # conclude touchdown immediately instead of letting the agent stay awake puffing thrusters.
        if left_leg == 1.0 and right_leg == 1.0 and abs(vx) < 0.1 and abs(vy) < 0.1:
            terminated = True

        # Terminal bonus/penalty to ensure landing happens strictly between the flags
        if terminated:
            if left_leg == 1.0 or right_leg == 1.0:
                # Flags are at x = +-0.20; with the lander's physical width, <= 0.22 is on the pad
                if abs(x) <= 0.22:
                    reward += 100.0
                else:
                    reward -= 100.0

        total_reward = reward + shaping_reward
        self.previous_potential = current_potential

        return observation, total_reward, terminated, truncated, info
