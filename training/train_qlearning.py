import numpy as np
from visualisation.plot_training import plot_training_results
from environment.traffic_env import TrafficSignalEnv
from agents.q_learning import QLearningAgent
import pickle


# Create environment
env = TrafficSignalEnv()

# Create Q-learning agent
agent = QLearningAgent(
    action_size=env.action_space.n,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
)

# Training configuration
episodes = 1000

episode_rewards = []
episode_queues = []


for episode in range(episodes):

    observation, info = env.reset()

    total_reward = 0
    total_queue = 0

    done = False

    while not done:

        # Agent chooses action
        action = agent.choose_action(observation)

        # Environment responds
        next_observation, reward, terminated, truncated, info = env.step(
            action
        )

        done = terminated or truncated

        # Agent learns
        agent.update(
            observation,
            action,
            reward,
            next_observation,
            done,
        )

        observation = next_observation

        total_reward += reward
        total_queue += info["total_queue"]

    # Reduce exploration
    agent.decay_epsilon()

    episode_rewards.append(total_reward)
    episode_queues.append(total_queue)

    # Print progress
    if (episode + 1) % 100 == 0:

        avg_reward = np.mean(
            episode_rewards[-100:]
        )

        avg_queue = np.mean(
            episode_queues[-100:]
        )

        print(
            f"Episode {episode + 1:4d} | "
            f"Avg Reward: {avg_reward:8.2f} | "
            f"Avg Queue: {avg_queue:8.2f} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"States: {len(agent.q_table)}"
        )

env.close()



plot_training_results(
    episode_rewards,
    episode_queues
)

env.close()

plot_training_results(
    episode_rewards,
    episode_queues
)

# Save the trained Q-learning agent
with open("results/q_learning_agent.pkl", "wb") as file:
    pickle.dump(agent, file)

print("Q-learning agent saved successfully.")