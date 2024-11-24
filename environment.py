# environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import scipy.linalg

class MatrixEnv(gym.Env):
    """
    Custom Environment for the matrix perturbation problem.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, A, index_i, alpha=1.0, beta=1.0):
        super(MatrixEnv, self).__init__()
        self.A = A
        self.n = A.shape[0]
        self.index_i = index_i
        self.alpha = alpha
        self.beta = beta

        # Action is the value of B_{i,i}
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        # Observation space can be empty or minimal since A is fixed
        # For simplicity, we'll have a dummy observation
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.state = None  # Placeholder state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.0], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        B_ii = action[0]
        B = np.zeros_like(self.A)
        B[self.index_i, self.index_i] = B_ii

        # Compute the perturbed matrix C
        C = self.A + B

        # Remove the i-th row and column to get C_reduced
        C_reduced = np.delete(np.delete(C, self.index_i, axis=0), self.index_i, axis=1)

        # Compute the eigenvalues of C_reduced
        eigenvalues = np.linalg.eigvals(C_reduced)
        max_real_part = np.max(np.real(eigenvalues))

        # Compute the reward
        reward = -self.alpha * (B_ii ** 2) - self.beta * abs(max_real_part)

        # For this environment, the episode ends after one step
        done = True

        # No additional info
        info = {'max_real_eigenvalue': max_real_part}

        return self.state, reward, done, False, info

    def render(self):
        pass  # Not implemented

    def close(self):
        pass  # Not implemented

