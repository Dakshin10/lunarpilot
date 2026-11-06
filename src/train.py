import argparse
import sys
from pathlib import Path

# Add project root to sys.path so 'from src...' imports work when running directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.environment import make_env


def main():
    parser = argparse.ArgumentParser(description="Train or fine-tune the LunarPilot Continuous PPO Agent")
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="Train completely from scratch instead of fine-tuning existing weights",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=150_000,
        help="Training timesteps budget (default: 150,000)",
    )
    args = parser.parse_args()

    # Create the continuous environment
    env = Monitor(make_env(continuous=True))

    model_path = Path("models/lunarpilot_ppo_continuous.zip")

    if model_path.exists() and not args.scratch:
        print(f"🛸 Loading existing policy weights from {model_path} to fine-tune...")
        model = PPO.load(str(model_path).replace(".zip", ""), env=env)
    else:
        print("🌱 Initializing fresh Continuous PPO model (Gaussian Actor-Critic Policy)...")
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
            tensorboard_log="./logs/",
        )

    # Train / fine-tune the continuous AI
    model.learn(total_timesteps=args.timesteps)

    # Ensure models directory exists
    Path("models").mkdir(parents=True, exist_ok=True)

    # Save the continuous model weights
    model.save("models/lunarpilot_ppo_continuous")

    # Close the environment
    env.close()

    print("✅ Continuous training completed! Model saved to models/lunarpilot_ppo_continuous.zip")


if __name__ == "__main__":
    main()
