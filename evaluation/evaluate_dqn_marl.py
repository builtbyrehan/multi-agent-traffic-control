import pickle
import numpy as np

from environment.multi_agent_env import MultiAgentTrafficEnv


EPISODES = 100
MAX_STEPS = 100


def main():

    env = MultiAgentTrafficEnv()

    # Load trained DQN agents
    with open("results/dqn_marl_agents.pkl", "rb") as file:
        agents = pickle.load(file)

    # Disable exploration during evaluation
    for agent in agents:
        agent.epsilon = 0.0

    episode_queues = []
    episode_rewards = []

    print("\nRunning DQN-MARL evaluation...\n")

    for episode in range(EPISODES):

        # Same seeds used by our previous evaluation
        seed = 1000 + episode

        observations = env.reset(seed=seed)

        total_queue = 0
        total_reward = 0

        done = False
        step = 0

        while not done and step < MAX_STEPS:

            actions = []

            # Each DQN agent selects its action
            # using the trained neural network
            for agent_id in range(2):

                action = agents[agent_id].choose_action(
                    observations[agent_id]
                )

                actions.append(action)

            # Execute both actions
            next_observations, rewards, done, info = env.step(
                actions
            )

            total_queue += info["total_queue"]
            total_reward += sum(rewards)

            observations = next_observations

            step += 1

        episode_queues.append(total_queue)
        episode_rewards.append(total_reward)

    env.close()

    avg_queue = np.mean(episode_queues)
    avg_reward = np.mean(episode_rewards)

    avg_queue_per_step = avg_queue / MAX_STEPS

    print("=" * 60)
    print("           DQN-MARL EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nEpisodes: {EPISODES}")

    print(
        f"\nAverage Total Reward: "
        f"{avg_reward:.2f}"
    )

    print(
        f"Average Cumulative Network Queue: "
        f"{avg_queue:.2f}"
    )

    print(
        f"Average Network Queue Per Step: "
        f"{avg_queue_per_step:.2f}"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()