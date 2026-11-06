import sys
from pathlib import Path

# Add project root to sys.path so 'from src...' works when running directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from src.environment import make_env

env = make_env(render_mode="human")
model = PPO.load("models/lunarpilot_ppo")

for episode in range(5):
    observation, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _states = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = observation

    # Flags are at x = +-0.20; accounting for the lander's physical width, <= 0.22 touches the pad
    between_flags = abs(x) <= 0.22
    landed = (
        between_flags
        and left_leg == 1.0
        and right_leg == 1.0
        and abs(vx) < 0.2
        and abs(vy) < 0.2
        and abs(angle) < 0.2
    )

    print(f"\nEpisode: {episode + 1}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Final Position: x={x:.2f}, y={y:.2f} (Between Flags: {'YES' if between_flags else 'NO'})")
    print(f"Final Velocity: vx={vx:.2f}, vy={vy:.2f}")
    print(f"Leg Contacts: Left={left_leg}, Right={right_leg}")
    print(f"Landing Status: {'SUCCESS' if landed else 'UNSUCCESSFUL'}")

env.close()
