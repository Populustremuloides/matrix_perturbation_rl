# Matrix Perturbation Reinforcement Learning Project

This project implements a reinforcement learning agent to solve a matrix perturbation optimization problem. The goal is to find the minimal perturbation `B` such that the maximum real part of the eigenvalues of the perturbed matrix (after removing a specific row and column) is zero.

## Files

- `main.py`: Main script to run the training loop.
- `environment.py`: Custom Gymnasium environment `MatrixEnv`.
- `agent.py`: DDPG agent implementation.
- `models.py`: Neural network models for the Actor and Critic.
- `replay_buffer.py`: Experience replay buffer.
- `utils.py`: Utility functions (optional).

## Requirements

- Python 3.x
- Gymnasium
- PyTorch
- NumPy
- Matplotlib

## Usage

```bash
pip install -r requirements.txt
python main.py

