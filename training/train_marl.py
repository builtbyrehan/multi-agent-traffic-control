import pickle
import numpy as np

from environment.multi_agent_env import MultiAgentTrafficEnv
from agents.q_learning import QLearningAgent


EPISODES = 1000


env = MultiAgentTrafficEnv()


# --------------------------------------------------
# Create one Q-learning agent for each intersection
# --------------------------------------------------

agents = [
    QLearningAgent(
        action_size=2,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ),

    QLearningAgent(
        action_size=2,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )
]


episode_rewards = []


# --------------------------------------------------
# Training
# --------------------------------------------------

for episode in range(EPISODES):

    observations = env.reset()

    total_rewards = np.zeros(2)

    done = False

    while not done:

        actions = []

        # Each agent chooses its own action
        for agent_id in range(2):

            action = agents[agent_id].choose_action(
                observations[agent_id]
            )

            actions.append(action)

        # Environment executes both actions
        next_observations, rewards, done, info = env.step(
            actions
        )

        # Each agent learns independently
        for agent_id in range(2):

            agents[agent_id].update(
                observations[agent_id],
                actions[agent_id],
                rewards[agent_id],
                next_observations[agent_id],
                done,
            )

            total_rewards[agent_id] += rewards[agent_id]

        observations = next_observations

    # Reduce exploration
    for agent in agents:
        agent.decay_epsilon()

    total_episode_reward = np.sum(total_rewards)

    episode_rewards.append(total_episode_reward)

    if (episode + 1) % 100 == 0:

        print(
            f"Episode {episode + 1:4d} | "
            f"Total Reward: {total_episode_reward:8.2f} | "
            f"Agent 0 ε: {agents[0].epsilon:.3f} | "
            f"Agent 1 ε: {agents[1].epsilon:.3f} | "
            f"States A0: {len(agents[0].q_table):5d} | "
            f"States A1: {len(agents[1].q_table):5d}"
        )


env.close()


# --------------------------------------------------
# Save both trained agents
# --------------------------------------------------

with open("results/marl_agents.pkl", "wb") as file:

    pickle.dump(agents, file)


print("\nMARL training complete.")
print("Agents saved to results/marl_agents.pkl")