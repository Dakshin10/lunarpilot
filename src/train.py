import sys
from pathlib import Path

# Add project root to sys.path so 'from src...' imports work when running directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.environment import make_env


def main():
    # Create the continuous environment
    env = Monitor(make_env(continuous=True))

    # Create the Continuous PPO model (Gaussian Actor-Critic Policy)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/"
    )

    # Train the continuous AI
    model.learn(
        total_timesteps=350_000
    )

    # Create models directory
    Path("models").mkdir(
        parents=True,
        exist_ok=True
    )

    # Save the continuous model weights
    model.save("models/lunarpilot_ppo_continuous")

    # Close the environment
    env.close()

    print("Continuous training completed! Model saved to models/lunarpilot_ppo_continuous.zip")


if __name__ == "__main__":
    main()
