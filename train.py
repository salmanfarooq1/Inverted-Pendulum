import os
from agent import RLAgent

def main(train_timesteps=10000, model_save_path="models/ppo_cartpole"):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Initialize the RLAgent
    agent = RLAgent()

    # Train the agent
    print(f"Training the agent for {train_timesteps} timesteps...")
    agent.train(train_timesteps)

    # Save the trained model
    agent.save_model(model_save_path)
    print(f"Model successfully trained and saved at: {model_save_path}")


if __name__ == "__main__":
    main(train_timesteps=100000, model_save_path="models/ppo_cartpole")
