
import sys
from pathlib import Path

# Add project root to sys.path so 'from src...' works when running directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.environment import LunarPilotEnvironment

env = LunarPilotEnvironment(render_mode="human")

observation, info = env.reset()

print("Initial Observation:", observation)
print("Observation Shape:", observation.shape)

for step in range(500):

    action = env.env.action_space.sample()

    observation, reward, done, info = env.step(action)

    print(f"Step: {step} | Reward: {reward:.2f}")

    if done:
        print("Episode finished!")
        observation, info = env.reset()

env.close()
