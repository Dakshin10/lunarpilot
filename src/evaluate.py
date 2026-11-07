import argparse
import sys
from pathlib import Path

# Add project root to sys.path so 'from src...' works when running directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO
from src.environment import make_env

parser = argparse.ArgumentParser(description="Evaluate LunarPilot autonomous agent.")
parser.add_argument(
    "--scale",
    type=float,
    default=2.0,
    help="Display window scale factor (default: 2.0 = 1200x800)",
)
parser.add_argument(
    "--width",
    type=int,
    default=None,
    help="Custom window width in pixels (e.g. 1200, 1600)",
)
parser.add_argument(
    "--height",
    type=int,
    default=None,
    help="Custom window height in pixels (e.g. 800, 1000)",
)
parser.add_argument(
    "--no-hud",
    action="store_true",
    help="Disable the real-time avionics telemetry HUD banner",
)
parser.add_argument(
    "--episodes",
    type=int,
    default=10,
    help="Number of flight test episodes (default: 10)",
)
args, _ = parser.parse_known_args()

# Determine model to load (prefer continuous model if trained)
continuous_model_path = Path("models/lunarpilot_ppo_continuous.zip")
discrete_model_path = Path("models/lunarpilot_ppo.zip")

if continuous_model_path.exists():
    print("🛸 Loading Continuous Control Pilot (LunarLanderContinuous-v3)...")
    model_path = "models/lunarpilot_ppo_continuous"
    continuous = True
elif discrete_model_path.exists():
    print("🕹️ Loading Discrete Control Pilot (LunarLander-v3)...")
    model_path = "models/lunarpilot_ppo"
    continuous = False
else:
    raise FileNotFoundError("No trained model found in models/ directory!")

# Initialize with the Full 5-Subsystem Avionics Suite and Resizable High-Res Display
env = make_env(
    render_mode="human",
    continuous=continuous,
    enable_avionics=True,
    enable_fuel_system=True,
    initial_fuel=100.0,
    enable_ekf_sensors=True,
    enable_black_box=True,
    window_scale=args.scale,
    window_width=args.width,
    window_height=args.height,
    enable_hud=not args.no_hud,
)
model = PPO.load(model_path)

num_episodes = args.episodes
flight_history = []

print(f"\n🚀 Commencing Avionics Flight Test ({num_episodes} Episodes)...")
print("Subsystems Active: Propellant Tracking • EKF State Filter • Black Box FDR • Gear Stress Telemetry • Contact Cutoff\n")

for episode in range(num_episodes):
    observation, info = env.reset()
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _states = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        if getattr(env, "should_exit", False):
            break

    if getattr(env, "should_exit", False):
        print("\n🛑 Flight test session terminated by user.")
        break

    # Retrieve true state from avionics suite for accurate post-flight telemetry
    if hasattr(env, "last_true_state"):
        x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = env.last_true_state
    elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "last_true_state"):
        x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = env.unwrapped.last_true_state
    else:
        x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = observation

    # EKF state estimation accuracy telemetry
    est_state = getattr(env, "last_est_state", None)
    if est_state is not None:
        pos_err = float(np.sqrt((x - est_state[0])**2 + (y - est_state[1])**2))
        vel_err = float(np.sqrt((vx - est_state[2])**2 + (vy - est_state[3])**2))
    else:
        pos_err, vel_err = 0.0, 0.0

    target_x = float(info.get("target_x", 0.0))
    target_deviation = abs(x - target_x)
    impact_speed = float(np.sqrt(vx**2 + vy**2))
    angle_deg = float(np.degrees(angle))
    omega_deg = float(np.degrees(angular_velocity))

    # Flags are at x = +-0.20; accounting for the lander's physical width, <= 0.22 touches the pad
    between_flags = abs(x - target_x) <= 0.22
    touchdown = (
        (left_leg == 1.0 and right_leg == 1.0)
        or ((left_leg == 1.0 or right_leg == 1.0) and y <= 0.05)
        or (y <= 0.05 and abs(vx) < 0.05 and abs(vy) < 0.05)
    )
    landed = (
        between_flags
        and touchdown
        and abs(vx) < 0.18
        and abs(vy) < 0.18
        and abs(angle) < 0.2
    )

    # Spacecraft Structural Health & Damage Assessment
    kinetic_damage = 40.0 * (vx ** 2) + 35.0 * (vy ** 2)
    shear_damage = 25.0 * abs(vx)
    tilt_damage = 30.0 * abs(angle) + 5.0 * abs(angular_velocity)
    total_damage_pct = min(100.0, float(kinetic_damage + shear_damage + tilt_damage))
    health_pct = max(0.0, 100.0 - total_damage_pct)

    if health_pct >= 95.0 and abs(vx) < 0.06 and abs(angle) < 0.08:
        gear_stress = "NOMINAL (Pristine / Soft Touchdown)"
        gear_code = "NOMINAL"
    elif health_pct >= 85.0:
        gear_stress = "MODERATE (Acceptable Load / Minor Strain)"
        gear_code = "MODERATE"
    else:
        gear_stress = "HIGH STRESS (Shear / Hard Rebound)"
        gear_code = "HIGH_STRESS"

    fuel_remaining = float(info.get("fuel_pct", 100.0))
    fuel_used = 100.0 - fuel_remaining
    flameout = bool(info.get("flameout", False))
    fault_active = bool(info.get("fault_active", False))
    divert_active = bool(info.get("divert_active", False))

    status_str = (
        "SUCCESS (Pinpoint Touchdown)" if landed and target_deviation <= 0.05
        else "SUCCESS (Between Flags)" if landed
        else "UNSUCCESSFUL"
    )

    # Record telemetry for fleet statistics
    flight_history.append({
        "reward": total_reward,
        "landed": landed,
        "impact_speed": impact_speed,
        "vx": vx,
        "vy": vy,
        "angle_deg": angle_deg,
        "health_pct": health_pct,
        "damage_pct": total_damage_pct,
        "gear_code": gear_code,
        "fuel_used": fuel_used,
        "fuel_remaining": fuel_remaining,
        "steps": step_count,
        "target_deviation": target_deviation,
        "pos_err": pos_err,
        "vel_err": vel_err,
    })

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Episode: {episode + 1} / {num_episodes} | Duration: {step_count} steps")
    print("─────────────────────────────────────────────────────────")
    print("📊 Mission Score:")
    print(f"  • Total Reward: {total_reward:+.2f}")
    print(f"  • Landing Status: {status_str}")
    print("\n📍 Kinematics & Touchdown:")
    print(f"  • Coordinates (x, y): x = {x:+.3f} m, y = {y:+.3f} m")
    print(f"  • Target Alignment: Pad x = {target_x:+.2f} m (Deviation Δx = {target_deviation:.3f} m | Inside Flags: {'YES' if between_flags else 'NO'})")
    print(f"  • Touchdown Velocity: vx = {vx:+.3f} m/s, vy = {vy:+.3f} m/s")
    print(f"  • Total Impact Speed: {impact_speed:.3f} m/s")
    print(f"  • Attitude (Tilt / Rate): Pitch θ = {angle_deg:+.2f}° ({angle:+.3f} rad) | Angular Rate ω = {omega_deg:+.2f}°/s")
    print("\n⚙️ Mechanical & Landing Gear:")
    print(f"  • Structural Health Index: {health_pct:.1f}% (Wear / Damage: {total_damage_pct:.1f}%)")
    print(f"  • Leg Contacts: Left = {'GROUNDED' if left_leg == 1.0 else 'FREE'}, Right = {'GROUNDED' if right_leg == 1.0 else 'FREE'}")
    print(f"  • Landing Gear Load: {gear_stress}")
    print("\n🛸 Propulsion & Avionics Subsystems:")
    print(f"  • Propellant Tank: {fuel_remaining:.1f}% remaining ({fuel_used:.1f}% burned) [{'FLAMEOUT' if flameout else 'NOMINAL'}]")
    print(f"  • EKF State Estimation Error: Δpos = {pos_err:.4f} m | Δvel = {vel_err:.4f} m/s")
    print(f"  • Actuator Fault Status: {'ANOMALY ACTIVE' if fault_active else 'NOMINAL (Zero Faults)'}")
    print(f"  • Hazard Retargeting: {'DIVERT COMMANDED' if divert_active else 'NOMINAL (Primary Helipad)'}")

# Generate and export the Black Box Mission Report from the last flight
if hasattr(env, "recorder"):
    env.recorder.generate_mission_report("assets/mission_report_latest.png")
elif hasattr(env, "env") and hasattr(env.env, "recorder"):
    env.env.recorder.generate_mission_report("assets/mission_report_latest.png")

env.close()

# Compute aggregate fleet metrics
rewards = [f["reward"] for f in flight_history]
successes = [f["landed"] for f in flight_history]
speeds = [f["impact_speed"] for f in flight_history]
vx_vals = [abs(f["vx"]) for f in flight_history]
angles = [abs(f["angle_deg"]) for f in flight_history]
healths = [f["health_pct"] for f in flight_history]
damages = [f["damage_pct"] for f in flight_history]
fuels_used = [f["fuel_used"] for f in flight_history]
durations = [f["steps"] for f in flight_history]
gear_codes = [f["gear_code"] for f in flight_history]
deviations = [f["target_deviation"] for f in flight_history]

success_rate = (sum(successes) / len(successes)) * 100.0
pinpoint_count = sum(1 for d in deviations if d <= 0.05)
pct_nominal_gear = (gear_codes.count("NOMINAL") / len(gear_codes)) * 100.0
pct_moderate_gear = (gear_codes.count("MODERATE") / len(gear_codes)) * 100.0
pct_high_stress_gear = (gear_codes.count("HIGH_STRESS") / len(gear_codes)) * 100.0

print("\n" + "=" * 65)
print("              AVIONICS FLIGHT TEST SUMMARY REPORT")
print("=" * 65)
print(f"  • Total Flights Tested:          {len(flight_history)}")
print(f"  • Overall Success Rate:          {sum(successes)} / {len(successes)} ({success_rate:.1f}%)")
print(f"  • Centering Precision (Mean Δx): {np.mean(deviations):.3f} m (Pinpoint ≤5cm: {pinpoint_count}/{len(deviations)})")
print(f"  • Mean Vehicle Structural Health:{np.mean(healths):.1f}% (Mean Wear/Damage: {np.mean(damages):.1f}%)")
print(f"  • Mean Episode Return:           {np.mean(rewards):+.2f} ± {np.std(rewards):.2f} pts (Min: {np.min(rewards):+.2f}, Max: {np.max(rewards):+.2f})")
print(f"  • Mean Touchdown Impact Speed:   {np.mean(speeds):.3f} m/s (Max: {np.max(speeds):.3f} m/s)")
print(f"  • Mean Lateral Drift (|vx|):     {np.mean(vx_vals):.3f} m/s")
print(f"  • Mean Attitude Tilt (|θ|):      {np.mean(angles):.2f}°")
print(f"  • Landing Gear Load:             {pct_nominal_gear:.1f}% Nominal | {pct_moderate_gear:.1f}% Moderate | {pct_high_stress_gear:.1f}% High Stress")
print(f"  • Mean Propellant Consumed:      {np.mean(fuels_used):.1f}% (Mean Remaining: {100.0 - np.mean(fuels_used):.1f}%)")
print(f"  • Mean Flight Duration:          {np.mean(durations):.1f} steps")
print("=" * 65)
print("🎯 Avionics evaluation run complete!\n")
