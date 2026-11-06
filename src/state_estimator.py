"""
Sensors & State Estimation Suite (Extended Kalman Filter) for LunarPilot.
Adds realistic sensor noise (radar altimeter, IMU, Doppler) and applies an
Extended Kalman Filter (EKF) to reconstruct the true spacecraft flight state.
"""

import numpy as np


class SensorSuite:
    def __init__(
        self,
        pos_noise_std: float = 0.02,
        vel_noise_std: float = 0.03,
        angle_noise_std: float = 0.02,
    ):
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.angle_noise_std = angle_noise_std

    def add_noise(self, true_state: np.ndarray) -> np.ndarray:
        """
        Simulates noisy measurements:
        true_state: [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]
        """
        noisy_state = true_state.copy()

        # Add Gaussian measurement noise to continuous state channels
        noisy_state[0] += np.random.normal(0, self.pos_noise_std)   # Radar x
        noisy_state[1] += np.random.normal(0, self.pos_noise_std)   # Radar altimeter y
        noisy_state[2] += np.random.normal(0, self.vel_noise_std)   # Doppler vx
        noisy_state[3] += np.random.normal(0, self.vel_noise_std)   # Doppler vy
        noisy_state[4] += np.random.normal(0, self.angle_noise_std) # Laser gyro pitch
        noisy_state[5] += np.random.normal(0, self.angle_noise_std) # Gyro rate

        # Leg contacts are direct physical switches, preserved without noise
        return noisy_state


class ExtendedKalmanFilter:
    def __init__(self, dt: float = 0.02):
        self.dt = dt
        self.dim = 6  # [x, y, vx, vy, angle, angular_velocity]

        # State transition matrix F
        self.F = np.eye(self.dim)
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.F[4, 5] = dt

        # Measurement matrix H
        self.H = np.eye(self.dim)

        # Process noise covariance Q
        self.Q = np.eye(self.dim) * 0.001
        self.Q[2, 2] = 0.005
        self.Q[3, 3] = 0.005

        # Measurement noise covariance R
        self.R = np.eye(self.dim) * 0.005
        self.R[0, 0] = 0.01
        self.R[1, 1] = 0.01

        # State and covariance
        self.x = np.zeros(self.dim)
        self.P = np.eye(self.dim) * 0.1

    def reset(self, initial_state: np.ndarray):
        self.x = initial_state[: self.dim].copy()
        self.P = np.eye(self.dim) * 0.1

    def update(self, noisy_observation: np.ndarray) -> np.ndarray:
        """
        Runs EKF Predict and Measurement Update cycles.
        Returns filtered state vector [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg].
        """
        z = noisy_observation[: self.dim]

        # 1. Prediction step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # 2. Measurement Update (Kalman Gain)
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ y
        self.P = (np.eye(self.dim) - K @ self.H) @ P_pred

        # Reconstruct full 8-dimensional observation vector
        filtered_state = np.zeros(8)
        filtered_state[: self.dim] = self.x
        filtered_state[6:] = noisy_observation[6:]  # Legs

        return filtered_state
