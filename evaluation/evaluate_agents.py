import pickle
import numpy as np

from environment.traffic_env import TrafficSignalEnv


EPISODES = 100
MAX_STEPS = 100


def run_random_controller(env, seed):
    observation, _ = env.reset(seed=seed)

    total_reward = 0
    total_queue = 0

    for _ in range(MAX_STEPS):

        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_queue += info["total_queue"]

        if terminated or truncated:
            break

    return total_reward, total_queue


def run_fixed_controller(env, seed):
    observation, _ = env.reset(seed=seed)

    total_reward = 0
    total_queue = 0

    for step in range(MAX_STEPS):

        # Switch every 10 steps
        if step > 0 and step % 10 == 0:
            action = 1
        else:
            action = 0

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_queue += info["total_queue"]

        if terminated or truncated:
            break

    return total_reward, total_queue


def run_q_learning_controller(env, agent, seed):
    observation, _ = env.reset(seed=seed)

    total_reward = 0
    total_queue = 0

    for _ in range(MAX_STEPS):

        # No exploration during evaluation
        q_values = agent.get_q_values(observation)
        action = int(np.argmax(q_values))

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_queue += info["total_queue"]

        if terminated or truncated:
            break

    return total_reward, total_queue


def evaluate():

    env = TrafficSignalEnv()

    # Load trained Q-learning agent
    with open("results/q_learning_agent.pkl", "rb") as file:
        agent = pickle.load(file)

    random_rewards = []
    random_queues = []

    fixed_rewards = []
    fixed_queues = []

    q_rewards = []
    q_queues = []

    print("\nRunning evaluation...\n")

    for episode in range(EPISODES):

        # Same seed = same traffic conditions
        seed = 1000 + episode

        reward, queue = run_random_controller(
            env,
            seed
        )

        random_rewards.append(reward)
        random_queues.append(queue)

        reward, queue = run_fixed_controller(
            env,
            seed
        )

        fixed_rewards.append(reward)
        fixed_queues.append(queue)

        reward, queue = run_q_learning_controller(
            env,
            agent,
            seed
        )

        q_rewards.append(reward)
        q_queues.append(queue)

    env.close()

    print("=" * 60)
    print("                 EVALUATION RESULTS")
    print("=" * 60)

    print("\nRandom Controller")
    print(f"Average Reward: {np.mean(random_rewards):.2f}")
    print(f"Average Cumulative Queue: {np.mean(random_queues):.2f}")
    print(f"Average Queue Per Step: {np.mean(random_queues) / MAX_STEPS:.2f}")

    print("\nFixed-Time Controller")
    print(f"Average Reward: {np.mean(fixed_rewards):.2f}")
    print(f"Average Cumulative Queue: {np.mean(fixed_queues):.2f}")
    print(f"Average Queue Per Step: {np.mean(fixed_queues) / MAX_STEPS:.2f}")

    print("\nQ-Learning Controller")
    print(f"Average Reward: {np.mean(q_rewards):.2f}")
    print(f"Average Cumulative Queue: {np.mean(q_queues):.2f}")
    print(f"Average Queue Per Step: {np.mean(q_queues) / MAX_STEPS:.2f}")

    print("\n" + "=" * 60)

    # Improvement compared with random
    random_avg = np.mean(random_queues)
    q_avg = np.mean(q_queues)

    improvement = (
        (random_avg - q_avg)
        / random_avg
    ) * 100

    print(
        f"\nQ-Learning improvement over Random: "
        f"{improvement:.2f}%"
    )

    # Improvement compared with fixed
    fixed_avg = np.mean(fixed_queues)

    improvement_fixed = (
        (fixed_avg - q_avg)
        / fixed_avg
    ) * 100

    print(
        f"Q-Learning improvement over Fixed-Time: "
        f"{improvement_fixed:.2f}%"
    )

    print("=" * 60)


if __name__ == "__main__":
    evaluate()