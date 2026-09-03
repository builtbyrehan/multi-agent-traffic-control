import pickle
import numpy as np

from environment.multi_agent_env import MultiAgentTrafficEnv


EPISODES = 100
MAX_STEPS = 100


def evaluate_marl():

    env = MultiAgentTrafficEnv()

    # Load trained MARL agents
    with open("results/marl_agents.pkl", "rb") as file:
        agents = pickle.load(file)

    episode_rewards = []
    episode_queues = []

    print("\nRunning MARL evaluation...\n")

    for episode in range(EPISODES):

        # Fixed seed for reproducible traffic conditions
        seed = 1000 + episode

        observations = env.reset(seed=seed)

        total_reward = 0
        total_queue = 0

        done = False
        step = 0

        while not done and step < MAX_STEPS:

            actions = []

            # Each trained agent chooses independently
            for agent_id in range(2):

                q_values = agents[agent_id].get_q_values(
                    observations[agent_id]
                )

                action = int(np.argmax(q_values))

                actions.append(action)

            # Execute both actions simultaneously
            observations, rewards, done, info = env.step(actions)

            total_reward += sum(rewards)
            total_queue += info["total_queue"]

            step += 1

        episode_rewards.append(total_reward)
        episode_queues.append(total_queue)

    env.close()

    avg_reward = np.mean(episode_rewards)
    avg_queue = np.mean(episode_queues)
    avg_queue_per_step = avg_queue / MAX_STEPS

    print("=" * 60)
    print("              MARL EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nEpisodes: {EPISODES}")

    print(f"\nAverage Total Reward: {avg_reward:.2f}")

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
    evaluate_marl()