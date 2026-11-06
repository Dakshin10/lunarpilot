"""
Propellant & Fuel Management System for LunarPilot.
Tracks fuel mass, burn rate, and handles engine flameout when propellant is exhausted.
"""


class PropellantSystem:
    def __init__(
        self,
        initial_fuel: float = 100.0,
        main_burn_rate: float = 0.15,
        rcs_burn_rate: float = 0.03,
    ):
        self.initial_fuel = initial_fuel
        self.fuel = initial_fuel
        self.main_burn_rate = main_burn_rate
        self.rcs_burn_rate = rcs_burn_rate
        self.flameout = False

    def reset(self):
        self.fuel = self.initial_fuel
        self.flameout = False
        return self.fuel

    def consume(self, action):
        """
        Processes propellant consumption for continuous control action: [main_throttle, side_steering].
        If propellant is exhausted, triggers flameout and forces engines to shut down.
        """
        if self.fuel <= 0.0:
            self.fuel = 0.0
            self.flameout = True
            # Cut all engines due to fuel starvation
            return [-1.0, 0.0], 0.0

        main_throttle = float(action[0])
        side_throttle = float(action[1])

        fuel_used = 0.0

        # Main engine consumption
        if main_throttle > 0.0:
            power = (min(max(main_throttle, 0.0), 1.0) + 1.0) * 0.5
            fuel_used += power * self.main_burn_rate

        # RCS attitude thruster consumption
        if abs(side_throttle) > 0.5:
            fuel_used += min(max(abs(side_throttle), 0.5), 1.0) * self.rcs_burn_rate

        self.fuel = max(0.0, self.fuel - fuel_used)

        if self.fuel <= 0.0:
            self.flameout = True
            return [-1.0, 0.0], fuel_used

        return action, fuel_used

    @property
    def percentage(self) -> float:
        return (self.fuel / self.initial_fuel) * 100.0

