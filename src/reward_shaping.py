import numpy as np
import gymnasium as gym


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_potential = 0.0

    def calculate_potential(self, observation):
        x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = observation

        # 1. Parabolic centering potential (L1 + L2 well): forceful pull toward dead center (x=0)
        horizontal_potential = -35.0 * abs(x) - 75.0 * (x ** 2)

        # 2. Gated vertical descent potential:
        # Full descent reward (+12 per unit) is ONLY unlocked when aligned over the pad (|x| < 0.20).
        # When far off-center (|x| >= 0.25), descent potential is damped by 80%, compelling high-altitude centering.
        center_alignment_factor = max(0.0, 1.0 - 4.0 * abs(x))
        vertical_potential = -12.0 * abs(y) * (0.2 + 0.8 * center_alignment_factor)

        # 3. Kinetic energy damping: penalize speed with strong focus on lateral shear (vx)
        speed_penalty = -10.0 * np.sqrt(vx**2 + vy**2) - 12.0 * abs(vx)

        # 4. Attitude alignment: heavily penalize tilt to enforce level vertical touchdowns
        tilt_penalty = -16.0 * abs(angle) - 4.0 * abs(angular_velocity)

        # 5. Dual leg surface contact bonus
        leg_bonus = 15.0 * (left_leg + right_leg)

        return (
            horizontal_potential
            + vertical_potential
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

        # Telescoping potential difference: ΔΦ = Φ(s') - Φ(s)
        # Prevents positive reward leakage during hovering (where ΔΦ = 0)
        shaping_reward = current_potential - self.previous_potential

        x = observation[0]
        y = observation[1]
        vx = observation[2]
        vy = observation[3]
        angle = observation[4]
        left_leg = observation[6]
        right_leg = observation[7]

        # Contact light / Engine cutoff:
        # Protect landing gear and revolute joints from extreme lateral shear and bounce.
        # Require confirmed ground contact with calm residual velocities and level attitude.
        dual_leg_contact = (left_leg == 1.0 and right_leg == 1.0)
        is_soft_and_level = abs(vx) < 0.08 and abs(vy) < 0.10 and abs(angle) < 0.10

        # Single-leg settle fallback: prevents lingering if one leg has 1mm gap on terrain
        single_leg_settle = (
            (left_leg == 1.0 or right_leg == 1.0)
            and y <= 0.03
            and abs(vx) < 0.04
            and abs(vy) < 0.04
            and abs(angle) < 0.06
        )

        is_grounded = dual_leg_contact or (
            (left_leg == 1.0 or right_leg == 1.0) and y <= 0.04
        )

        if (dual_leg_contact and is_soft_and_level) or single_leg_settle:
            terminated = True

        # Terminal bonus/penalty: enforce structural health, zero shear, and dead-center precision
        if terminated or truncated:
            if truncated:
                # Penalize hovering until timeout without landing
                reward -= 100.0
            elif is_grounded:
                if abs(x) <= 0.22:
                    impact_speed = np.sqrt(vx**2 + vy**2)

                    # 1. Structural Softness & Low Impact Energy Score (up to +70 pts)
                    softness_score = max(0.0, 70.0 - 250.0 * impact_speed - 400.0 * (impact_speed ** 2))

                    # 2. Level Attitude Touchdown Score (up to +40 pts)
                    attitude_score = max(0.0, 40.0 * (1.0 - abs(angle) / 0.12))

                    # 3. Lateral Ground Skid Cancellation Score (up to +35 pts)
                    shear_score = max(0.0, 35.0 * (1.0 - abs(vx) / 0.08))

                    # 4. Bullseye Centering Score (up to +45 pts)
                    centering_score = max(0.0, 45.0 * (1.0 - (abs(x) / 0.22)))

                    # 5. Base safe landing bonus
                    base_landing_bonus = 40.0

                    reward += (
                        base_landing_bonus
                        + softness_score
                        + attitude_score
                        + shear_score
                        + centering_score
                    )
                else:
                    # Penalty for touching down on rough terrain outside flags
                    reward -= 100.0

        total_reward = reward + shaping_reward
        self.previous_potential = current_potential

        return observation, total_reward, terminated, truncated, info
