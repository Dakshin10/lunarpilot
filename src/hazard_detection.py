"""
Hazard Detection & Dynamic Helipad Retargeting System for LunarPilot.
Scans primary landing pad for terrain obstacles (boulders, craters) and dynamically
retargets guidance to secondary divert landing zones.
"""


class HazardDetectionSystem:
    def __init__(
        self,
        hazard_on_primary_pad: bool = False,
        scan_altitude: float = 0.85,
        divert_target_x: float = -0.45,
    ):
        self.hazard_present = hazard_on_primary_pad
        self.scan_altitude = scan_altitude
        self.divert_target_x = divert_target_x
        self.divert_commanded = False
        self.active_target_x = 0.0

    def reset(self):
        self.divert_commanded = False
        self.active_target_x = 0.0

    def scan_terrain(self, current_altitude: float) -> float:
        """
        Scans terrain. If obstacle is present at primary pad (x=0) and vehicle descends below
        scan altitude, dynamically issues an abort/divert command to the alternate pad.
        """
        if self.hazard_present and current_altitude <= self.scan_altitude:
            self.divert_commanded = True
            self.active_target_x = self.divert_target_x
        else:
            self.active_target_x = 0.0

        return self.active_target_x
