"""
Black Box Flight Data Recorder (FDR) & Telemetry Analytics Suite for LunarPilot.
Logs step-by-step flight telemetry, actuator commands, fuel consumption, and
generates aerospace-grade mission diagnostic trajectory plots.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BlackBoxRecorder:
    def __init__(self):
        self.records = []

    def reset(self):
        self.records = []

    def log_step(
        self,
        step: int,
        true_state: np.ndarray,
        estimated_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        fuel_pct: float,
        fault_active: bool,
        target_x: float,
    ):
        """
        Logs a single simulation time step.
        """
        record = {
            "step": step,
            "x_true": float(true_state[0]),
            "y_true": float(true_state[1]),
            "vx_true": float(true_state[2]),
            "vy_true": float(true_state[3]),
            "theta_true": float(true_state[4]),
            "omega_true": float(true_state[5]),
            "leg_L": int(true_state[6]),
            "leg_R": int(true_state[7]),
            "x_est": float(estimated_state[0]),
            "y_est": float(estimated_state[1]),
            "vx_est": float(estimated_state[2]),
            "vy_est": float(estimated_state[3]),
            "main_throttle": float(action[0]),
            "rcs_command": float(action[1]),
            "reward": float(reward),
            "fuel_pct": float(fuel_pct),
            "fault_active": int(fault_active),
            "target_x": float(target_x),
        }
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def generate_mission_report(self, output_path: str = "assets/mission_report_latest.png"):
        """
        Generates a 6-panel aerospace telemetry chart with dark aerospace styling.
        """
        if not self.records:
            return

        df = self.to_dataframe()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Style configuration
        plt.style.use("dark_background")
        fig, axs = plt.subplots(3, 2, figsize=(14, 11), facecolor="#0B0F19")
        fig.suptitle(
            "LUNARPILOT // FLIGHT DATA RECORDER TELEMETRY REPORT",
            fontsize=16,
            fontweight="bold",
            color="#FFFFFF",
            y=0.98,
        )

        grid_color = "#1E293B"

        # 1. Trajectory Profile (Altitude & Lateral Position)
        ax = axs[0, 0]
        ax.set_facecolor("#131A29")
        ax.plot(df["step"], df["y_true"], color="#38BDF8", lw=2, label="Altitude (y)")
        ax.plot(df["step"], df["x_true"], color="#A855F7", lw=2, label="Lateral (x)")
        ax.plot(df["step"], df["target_x"], color="#F1C40F", ls="--", lw=1.5, label="Target (x)")
        ax.axhline(0, color="#64748B", ls=":", alpha=0.6)
        ax.set_title("Trajectory Position (m)", color="#FFFFFF", fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalized Units", color="#94A3B8")
        ax.grid(True, color=grid_color, alpha=0.5)
        ax.legend(loc="upper right", facecolor="#1E293B", edgecolor="#334155")

        # 2. Velocity Vector Profile
        ax = axs[0, 1]
        ax.set_facecolor("#131A29")
        ax.plot(df["step"], df["vy_true"], color="#10B981", lw=2, label="Vertical (vy)")
        ax.plot(df["step"], df["vx_true"], color="#F59E0B", lw=2, label="Horizontal (vx)")
        ax.axhline(0, color="#64748B", ls=":", alpha=0.6)
        ax.set_title("Velocity Components (m/s)", color="#FFFFFF", fontsize=11, fontweight="bold")
        ax.grid(True, color=grid_color, alpha=0.5)
        ax.legend(loc="lower right", facecolor="#1E293B", edgecolor="#334155")

        # 3. Continuous Thruster Actuation
        ax = axs[1, 0]
        ax.set_facecolor("#131A29")
        ax.plot(df["step"], df["main_throttle"], color="#EF4444", lw=1.8, label="Main Throttle ([-1, 1])")
        ax.plot(df["step"], df["rcs_command"], color="#C084FC", lw=1.5, label="RCS Trim ([-1, 1])")
        ax.axhline(0, color="#64748B", ls=":", alpha=0.6)
        ax.set_title("Continuous Thruster Actuation", color="#FFFFFF", fontsize=11, fontweight="bold")
        ax.set_ylabel("Command Ratio", color="#94A3B8")
        ax.grid(True, color=grid_color, alpha=0.5)
        ax.legend(loc="lower left", facecolor="#1E293B", edgecolor="#334155")

        # 4. Propellant Consumption Curve
        ax = axs[1, 1]
        ax.set_facecolor("#131A29")
        ax.plot(df["step"], df["fuel_pct"], color="#F97316", lw=2, label="Fuel Tank %")
        ax.fill_between(df["step"], df["fuel_pct"], color="#F97316", alpha=0.2)
        ax.set_ylim(0, 105)
        ax.set_title("Propellant Remaining (%)", color="#FFFFFF", fontsize=11, fontweight="bold")
        ax.set_ylabel("Tank Level (%)", color="#94A3B8")
        ax.grid(True, color=grid_color, alpha=0.5)
        ax.legend(loc="lower left", facecolor="#1E293B", edgecolor="#334155")

        # 5. Extended Kalman Filter (EKF) Tracking Error
        ax = axs[2, 0]
        ax.set_facecolor("#131A29")
        error_y = df["y_true"] - df["y_est"]
        error_x = df["x_true"] - df["x_est"]
        ax.plot(df["step"], error_y, color="#38BDF8", lw=1.5, label="EKF Residual Δy")
        ax.plot(df["step"], error_x, color="#A855F7", lw=1.5, label="EKF Residual Δx")
        ax.axhline(0, color="#64748B", ls=":", alpha=0.6)
        ax.set_title("EKF Estimation Residual Error", color="#FFFFFF", fontsize=11, fontweight="bold")
        ax.set_xlabel("Flight Steps", color="#94A3B8")
        ax.set_ylabel("Estimation Error", color="#94A3B8")
        ax.grid(True, color=grid_color, alpha=0.5)
        ax.legend(loc="upper right", facecolor="#1E293B", edgecolor="#334155")

        # 6. Phase Portrait (Descent Flare Profile)
        ax = axs[2, 1]
        ax.set_facecolor("#131A29")
        ax.plot(df["y_true"], df["vy_true"], color="#10B981", lw=2, label="Phase Curve (y vs vy)")
        ax.scatter(df["y_true"].iloc[0], df["vy_true"].iloc[0], color="#38BDF8", s=60, label="Ignition Point", zorder=5)
        ax.scatter(df["y_true"].iloc[-1], df["vy_true"].iloc[-1], color="#E74C3C", s=60, label="Touchdown Point", zorder=5)
        ax.set_title("Flare Maneuver Phase Portrait (Altitude vs Vertical Velocity)", color="#FFFFFF", fontsize=11, fontweight="bold")
        ax.set_xlabel("Altitude y (m)", color="#94A3B8")
        ax.set_ylabel("Vertical Velocity vy (m/s)", color="#94A3B8")
        ax.grid(True, color=grid_color, alpha=0.5)
        ax.legend(loc="lower right", facecolor="#1E293B", edgecolor="#334155")

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()
        print(f"📊 Telemetry Mission Report exported to: {output_path}")
