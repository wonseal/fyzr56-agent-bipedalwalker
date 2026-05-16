# SAC Agent for BipedalWalker

Soft Actor-Critic (SAC) implementation trained on the `rldurham/Walker` environment (BipedalWalker). Includes two variants: **normal** and **hardcore** mode.

## Results

| Mode | Best Score | Episodes |
|------|-----------|----------|
| Normal | 241.23 | 999 |
| Hardcore | 214.50 | 1899 |

## Algorithm

**SAC** (Soft Actor-Critic) is an off-policy deep RL algorithm that maximizes both cumulative reward and policy entropy, encouraging exploration and robustness.

### Architecture

- **Actor (`PolicyNet`)**: 2-layer MLP (256 hidden units) outputting a Gaussian distribution over actions. Uses tanh squashing to bound actions and reparameterized sampling for gradient estimation.
- **Critics (`QNet`)**: Twin Q-networks to mitigate overestimation bias (clipped double Q-learning).
- **Automatic entropy tuning**: The temperature parameter α is learned automatically to match a target entropy of `-act_dim`.
- **Replay buffer**: Uniform random experience replay.
- **Observation normalization**: Online running mean/variance normalization applied to observations.

### Key Hyperparameters

| Hyperparameter | Normal | Hardcore |
|---------------|--------|----------|
| Learning rate (π, Q, α) | 3e-4 | 3e-4 |
| Buffer size | 250,000 | 1,000,000 |
| Batch size | 256 | 128 |
| Discount γ | 0.99 | 0.99 |
| Soft update τ | 0.005 | 0.005 |
| Random exploration steps | 1,000 | 10,000 |
| Update frequency | every step | every 4 steps |
| Policy delay | 1 | 2 |
| LayerNorm | Yes | No |

## Files

```
fyzr56-agent-code.ipynb           # SAC agent — normal mode
fyzr56-agent-code-hardcore.ipynb  # SAC agent — hardcore mode
fyzr56-agent-log.txt              # Training log — normal
fyzr56-agent-log-hardcore.txt     # Training log — hardcore
fyzr56-agent-video,...mp4         # Recorded episode — normal (ep 999, score 241.23)
fyzr56-agent-video-hardcore,...mp4# Recorded episode — hardcore (ep 1899, score 214.50)
```

## Setup

The notebooks are designed to run on **Google Colab**. Checkpoints are saved to Google Drive automatically.

```bash
pip install swig
pip install rldurham
```

## Training

Open the notebook in Colab and run all cells. The agent will:

1. Initialize the Walker environment with reward clipping (`min_reward=-10`)
2. Collect random transitions for the first N steps (exploration warm-up)
3. Train SAC with experience replay
4. Save checkpoints to Google Drive every 10 episodes
5. Record video episodes periodically

Checkpoints store the full agent state (network weights, optimizer states, obs normalization statistics) and training can be resumed automatically if a checkpoint is found at the save path.