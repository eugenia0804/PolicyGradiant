import torch
import torch.optim as optim
import numpy as np
from collections import deque
from policy import PolicyNetwork

def train_policy(
    env,
    device,
    obs_dim,
    act_dim,
    preprocess_obs=None,           
    action_map=None,              
    policy_lr=1e-3,
    gamma=0.99,
    training_episodes=1000,
    hidden_dim=64,
    baseline=False,
    baseline_window=100,
):
    """
    Train a policy using Policy Gradiant algorithm
    """

    policy = PolicyNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim, act_dim=act_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=policy_lr)
    all_rewards = []
    # Create a buffer to store baseline values
    baseline_buffer = deque(maxlen=baseline_window)

    for episode in range(training_episodes):
        obs, _ = env.reset()
        log_probs, rewards = [], []

        done = False
        while not done:
            # Preprocess observation
            if preprocess_obs:
                obs_t = preprocess_obs(obs)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

            # Get action probabilities
            probs = policy(obs_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            # Map action index to environment action
            if action_map:
                env_action = action_map(action.item())
            else:
                env_action = action.item()

            log_probs.append(dist.log_prob(action))
            obs, reward, terminated, truncated, _ = env.step(env_action)
            rewards.append(reward)
            done = terminated or truncated

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # Baseline subtraction
        if baseline:
            # Update baseline buffer
            avg_baseline = np.mean(baseline_buffer) if len(baseline_buffer) > 0 else 0.0
            returns = returns - avg_baseline

        # Normalize returns
        final_value = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient update
        loss = 0
        for log_prob, Gt in zip(log_probs, final_value):
            loss += -log_prob * Gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        all_rewards.append(total_reward)

        if episode % 50 == 0:
            print(f"Episode {episode:4d} | Return: {total_reward:.1f}")

    env.close()
    return policy, all_rewards