# CartPole & Pong Reinforcement Learning with Policy Gradient

This project implements a Policy Gradient algorithm to train an agent to play CartPole and Pong.

## Project Structure

- `train.py`: Main training script
- `train_pong.py`: Training script for Pong
- `utils.py`: Data processing and evaluation utilities
- `vis.py`: Visualization utilities
- `policy.py`: Architecture for the Policy Network

## Requirements

```bash
pip install torch gymnasium ale-py matplotlib numpy
```

## Components

### Key hyperparameters:
```python
policy_lr = 1e-3           # Learning rate
gamma = 0.99               # Discount factor
training_episodes = 3000   # Number of training episodes
hidden_dim = 128          # Hidden layer dimensions
baseline = True           # Use baseline for variance reduction
```

### Policy Network Architecture (`policy.py`)

A simple Multi-Layer Perceptron (MLP) that serves as our policy network:
- Input: Flattened game state (eg. 80x80 = 6400 dimensions for Pong)
- Hidden layer: Configurable dimensions with ReLU activation
- Output: Action probabilities using Softmax

### Policy Gradiant Algorithm (`train.py`)

The implementation involves three main stages. First, episodes are collected by running the current policy to generate trajectories while recording states, actions, rewards, and the log-probabilities of the chosen actions. Next, discounted rewards are computed using the discount factor (gamma) and normalized to improve training stability. The policy gradient is then calculated from the collected data and used to update the network parameters through gradient ascent with the Adam optimizer. Finally, training progress is monitored by tracking episode rewards, saving model checkpoints, and logging relevant training metrics.


### Moving Average Baseline Implementation (`train.py`)

The baseline implementation is designed to reduce variance in policy gradient estimates. It maintains a moving average of recent episode returns, typically using a window of 100 episodes and a deque for efficient updates. This baseline is subtracted from the returns before policy updates, which helps stabilize training and allows the algorithm to adapt to changing performance levels. The baseline is updated after each episode and is applied only when the baseline option is enabled.

### Result Visualization

Detailed visualization and analysis are available in `res_cartpole.ipynb` and `res_pong.ipynb`.


### Pong Training Output Directory Structure (`train_pong.py`)

The Pong training script creates a directory under `runs/` with the following structure:
```
runs/
└── pong_lr{lr}_g{gamma}_ep{episodes}_h{hidden_dim}_baseline{baseline}:{window}/
    ├── policy.pt           # Saved model weights
    ├── rewards.pt          # Training rewards
    ├── rollout_rewards.pt  # Evaluation rewards
    └── training.log        # Training logs
```
