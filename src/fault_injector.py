"""
Hardware Fault Injection & Anomaly Simulator for LunarPilot.
Simulates thruster failures, actuator jamming, and engine thrust degradation.
"""

from enum import Enum
import numpy as np


class AnomalyType(Enum):
    NONE = "NOMINAL"
    LEFT_RCS_JAM = "LEFT_RCS_OFFLINE"
    RIGHT_RCS_JAM = "RIGHT_RCS_OFFLINE"
    MAIN_ENGINE_DEGRADE = "MAIN_ENGINE_DEGRADE_50%"


class FaultInjector:
    def __init__(
        self,
        anomaly_type: AnomalyType = AnomalyType.NONE,
        trigger_altitude: float = 0.6,
    ):
        self.anomaly_type = anomaly_type
        self.trigger_altitude = trigger_altitude
        self.fault_active = False

    def reset(self):
        self.fault_active = False

    def process_actuation(self, action: np.ndarray, altitude: float) -> np.ndarray:
        """
        Modifies physical thrust delivered to Box2D if an in-flight anomaly is active.
        """
        if self.anomaly_type == AnomalyType.NONE:
            return action

        # Trigger failure once altitude descends below trigger threshold
        if altitude <= self.trigger_altitude:
            self.fault_active = True

        if not self.fault_active:
            return action

        corrupted_action = action.copy()

        if self.anomaly_type == AnomalyType.LEFT_RCS_JAM:
            # Left engine cannot fire (prevent positive side thrust > 0.5)
            if corrupted_action[1] > 0.5:
                corrupted_action[1] = 0.0

        elif self.anomaly_type == AnomalyType.RIGHT_RCS_JAM:
            # Right engine cannot fire (prevent negative side thrust < -0.5)
            if corrupted_action[1] < -0.5:
                corrupted_action[1] = 0.0

        elif self.anomaly_type == AnomalyType.MAIN_ENGINE_DEGRADE:
            # Main engine limited to 50% power
            if corrupted_action[0] > 0.0:
                corrupted_action[0] = corrupted_action[0] * 0.5

        return corrupted_action
