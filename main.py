# main.py
import numpy as np
import torch
import gymnasium as gym
from environment import MatrixEnv
from agent import DDPGAgent
import matplotlib.pyplot as plt

def main():
    # Define the matrix A and index i
    n = 5  # Matrix size
    A = np.random.randn(n, n)
    index_i = np.random.randint(0, n)  # Random index for i

    # Create the environment
    env = MatrixEnv(A=A, index_i=index_i, alpha=1.0, beta=1.0)

    # Get action and state space information
    action_space = env.action_space
    state_space = env.observation_space
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    # Create the agent
    agent = DDPGAgent(action_space, state_space, max_action, min_action)

    num_episodes = 1000
    batch_size = 64
    noise_scale = 0.1

    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state, noise_scale=noise_scale)
            next_state, reward, done, _, info = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            agent.train(batch_size)

        rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")

    # Plot the rewards
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.show()

    # Test the agent after training
    state, _ = env.reset()
    action = agent.select_action(state, noise_scale=0.0)  # No exploration noise
    print(f"Optimal B_ii: {action[0]}")

    # Compute the final matrices and eigenvalues
    B = np.zeros_like(A)
    B[index_i, index_i] = action[0]
    C = A + B
    C_reduced = np.delete(np.delete(C, index_i, axis=0), index_i, axis=1)
    eigenvalues = np.linalg.eigvals(C_reduced)
    max_real_part = np.max(np.real(eigenvalues))
    print(f"Max real part of eigenvalues: {max_real_part}")

if __name__ == '__main__':
    main()

