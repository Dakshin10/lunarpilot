"""
Unified Aerospace Avionics Suite for LunarPilot.
Integrates all 5 advanced GNC subsystems:
1. Propellant & Fuel Flameout System
2. Sensor Suite & Extended Kalman Filter (EKF) State Estimator
3. Fault Injection & Thruster Anomaly Simulator
4. Hazard Detection & Dynamic Landing Site Retargeting
5. Black Box Flight Data Recorder (FDR)
"""

import gymnasium as gym
import numpy as np

from src.fault_injector import AnomalyType, FaultInjector
from src.flight_recorder import BlackBoxRecorder
from src.fuel_system import PropellantSystem
from src.hazard_detection import HazardDetectionSystem
from src.state_estimator import ExtendedKalmanFilter, SensorSuite


class AvionicsSuite(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        enable_fuel_system: bool = True,
        initial_fuel: float = 100.0,
        enable_ekf_sensors: bool = True,
        anomaly_type: AnomalyType = AnomalyType.NONE,
        enable_hazard_detection: bool = False,
        hazard_divert_x: float = -0.45,
        enable_black_box: bool = True,
    ):
        super().__init__(env)

        self.enable_fuel = enable_fuel_system
        self.enable_ekf = enable_ekf_sensors
        self.enable_hazard = enable_hazard_detection
        self.enable_recorder = enable_black_box

        # Subsystems
        self.fuel_system = PropellantSystem(initial_fuel=initial_fuel)
        self.sensors = SensorSuite()
        self.ekf = ExtendedKalmanFilter()
        self.fault_injector = FaultInjector(anomaly_type=anomaly_type)
        self.hazard_system = HazardDetectionSystem(
            hazard_on_primary_pad=enable_hazard_detection,
            divert_target_x=hazard_divert_x,
        )
        self.recorder = BlackBoxRecorder()

        self.current_step = 0
        self.last_true_state = np.zeros(8)
        self.last_est_state = np.zeros(8)

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        self.current_step = 0

        self.fuel_system.reset()
        self.fault_injector.reset()
        self.hazard_system.reset()
        self.recorder.reset()

        if self.enable_ekf:
            noisy_obs = self.sensors.add_noise(raw_obs)
            self.ekf.reset(noisy_obs)
            est_obs = self.ekf.update(noisy_obs)
        else:
            est_obs = raw_obs.copy()

        self.last_true_state = raw_obs
        self.last_est_state = est_obs

        # Include avionics telemetry in info dict
        info["fuel_pct"] = self.fuel_system.percentage
        info["target_x"] = self.hazard_system.active_target_x
        info["flameout"] = self.fuel_system.flameout

        return est_obs, info

    def step(self, action):
        self.current_step += 1
        raw_action = np.array(action, dtype=np.float32)

        # 1. Fuel consumption & flameout check
        if self.enable_fuel:
            actuation, fuel_used = self.fuel_system.consume(raw_action)
        else:
            actuation = raw_action
            fuel_used = 0.0

        # 2. Hardware fault injection (if triggered by altitude)
        altitude = self.last_true_state[1]
        commanded_action = self.fault_injector.process_actuation(actuation, altitude=altitude)

        # 3. Step the physical simulation (Box2D)
        true_obs, reward, terminated, truncated, info = self.env.step(commanded_action)
        self.last_true_state = true_obs

        # 4. Hazard scan & dynamic retargeting
        target_x = self.hazard_system.scan_terrain(current_altitude=true_obs[1])

        # If retargeting is active, reward guidance toward alternate divert pad
        if self.hazard_system.divert_commanded:
            dist_divert = abs(true_obs[0] - target_x)
            reward -= 5.0 * dist_divert
            if terminated and (true_obs[6] == 1.0 or true_obs[7] == 1.0):
                if abs(true_obs[0] - target_x) <= 0.22:
                    reward += 100.0  # Safe touchdown on alternate pad!

        # 5. Sensor noise & EKF estimation
        if self.enable_ekf:
            noisy_obs = self.sensors.add_noise(true_obs)
            est_obs = self.ekf.update(noisy_obs)
        else:
            est_obs = true_obs.copy()

        self.last_est_state = est_obs

        # 6. Black Box flight recording
        if self.enable_recorder:
            self.recorder.log_step(
                step=self.current_step,
                true_state=true_obs,
                estimated_state=est_obs,
                action=commanded_action,
                reward=reward,
                fuel_pct=self.fuel_system.percentage,
                fault_active=self.fault_injector.fault_active,
                target_x=target_x,
            )

        # 7. Fuel flameout consequences
        if self.fuel_system.flameout:
            info["flameout"] = True
            # Extra penalty if craft runs out of fuel while high above ground
            if true_obs[1] > 0.3:
                reward -= 1.0

        info["fuel_pct"] = self.fuel_system.percentage
        info["fault_active"] = self.fault_injector.fault_active
        info["divert_active"] = self.hazard_system.divert_commanded
        info["target_x"] = target_x

        return est_obs, reward, terminated, truncated, info

