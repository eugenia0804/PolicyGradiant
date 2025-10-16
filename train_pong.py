import os
import sys
import time
import torch
import gymnasium as gym
import ale_py
from utils import preprocess_frame
from train import train_policy
from policy import PolicyNetwork
from utils import rollout_policy

# Hyperparameters
policy_lr = 1e-3 
gamma = 0.99 # Discount factor
training_episodes = 3000
hidden_dim = 128 # Number of hidden units in policy network
cuda = 1
basline = True # Use baseline for variance reduction
baseline_window = 200 # Moving average window for baseline

# Create output directory
run_name = f"pong_lr{policy_lr}_g{gamma}_ep{training_episodes}_h{hidden_dim}_baseline{basline}:{baseline_window}"
output_dir = os.path.join("runs", run_name)
os.makedirs(output_dir, exist_ok=True)

log_path = os.path.join(output_dir, "training.log")
model_path = os.path.join(output_dir, "policy.pt")
rewards_path = os.path.join(output_dir, "rewards.pt")
rollout_rewards_path = os.path.join(output_dir, "rollout_rewards.pt")

# Make output unbuffered and constantly updated
log_file = open(log_path, "w", buffering=1) 
sys.stdout = log_file
sys.stderr = log_file

# Ensure print always flushes
print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

# Create environment and set device
env = gym.make("ALE/Pong-v5")
device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

obs_shape = (80, 80)
obs_dim = obs_shape[0] * obs_shape[1]
act_dim = 2

# Preprocess observation function
def preprocess_obs(obs):
    frame = preprocess_frame(obs)
    return torch.as_tensor(frame.flatten(), dtype=torch.float32, device=device)

# Map envrionment index to action index in training
def action_map(idx):
    return [2, 3][idx] 

# # Train policy
# policy, rewards = train_policy(
#     env=env,
#     device=device,
#     obs_dim=obs_dim,
#     act_dim=act_dim,
#     preprocess_obs=preprocess_obs,
#     action_map=action_map,
#     policy_lr=policy_lr,
#     gamma=gamma,
#     training_episodes=training_episodes,
#     hidden_dim=hidden_dim,
#     baseline=basline,
#     baseline_window=baseline_window
# )

# # Save model and rewards
# torch.save(policy.state_dict(), model_path)
# torch.save(rewards, rewards_path)

# print("\nTraining completed successfully!")
# print(f"Model saved to: {model_path}")
# print(f"Rewards saved to: {rewards_path}")
# print(f"Full logs at: {log_path}")


# Rollout trained policy
trained_policy = PolicyNetwork(obs_dim, hidden_dim=hidden_dim, act_dim=2).to(device)
trained_policy.load_state_dict(torch.load(model_path, map_location=device))
rollout_rewards = rollout_policy(env, policy=trained_policy, device=device, episodes=500, preprocess_obs=preprocess_obs, action_map=action_map)

# Save rollout rewards
torch.save(rollout_rewards, rollout_rewards_path)

print("\Rolling out trained policy network completed successfully!")
print(f"Resulted rewards saved to: {rollout_rewards_path}")