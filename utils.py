import numpy as np
import torch

def preprocess_frame(image):
    """
    Preprocess a 210x160x3 uint8 frame into a 80x80 float32 array
    """
    image = image[35:195]
    image = image[::2, ::2, 0]
    image[image == 144] = 0
    image[image == 109] = 0
    image[image != 0] = 1
    return np.reshape(image.astype(np.float32).ravel(), [80,80])

    
def rollout_policy(env, policy, device, episodes=500, preprocess_obs=None, action_map=None):
    """
    Evaluate a policy by running it in the environment for a number of episodes
    """
    rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Preprocess observation if a preprocessing function is provided
            if preprocess_obs is not None:
                obs_t = preprocess_obs(obs)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

            with torch.no_grad():
                probs = policy(obs_t)

            # Select action with highest probability
            action = torch.argmax(probs).item()

            # Map action if an action mapping function is provided
            if action_map is not None:
                env_action = action_map(action)
            else:
                env_action = action

            # Step the environment
            obs, reward, terminated, truncated, _ = env.step(env_action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

        if episode % 50 == 0:
            print(f"Episode {episode:4d} | Return: {total_reward:.1f}")

    env.close()
    return np.array(rewards)
