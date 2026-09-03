import pickle
import numpy as np

from environment.multi_agent_env import MultiAgentTrafficEnv


EPISODES = 100
MAX_STEPS = 100
NUM_AGENTS = 2


def evaluate_controller(env, controller, agents=None):
    """
    Evaluate a controller over the same 100 traffic scenarios.

    Metric:
        Average Network Queue Per Step

    The Tabular MARL and DQN-MARL controllers use their
    trained policies with exploration disabled.
    """

    episode_queues = []

    for episode in range(EPISODES):

        # Same traffic seed for every controller
        seed = 1000 + episode

        observations = env.reset(seed=seed)

        total_queue = 0.0
        step = 0
        done = False

        while not done and step < MAX_STEPS:

            # ==================================================
            # RANDOM CONTROLLER
            # ==================================================

            if controller == "random":

                actions = []

                for agent_id in range(NUM_AGENTS):
                    action = np.random.randint(2)
                    actions.append(action)

            # ==================================================
            # FIXED-TIME CONTROLLER
            # ==================================================

            elif controller == "fixed":

                actions = []

                for agent_id in range(NUM_AGENTS):

                    # Switch every 5 steps
                    if step % 5 == 0:
                        action = 1
                    else:
                        action = 0

                    actions.append(action)

            # ==================================================
            # TABULAR MARL
            # ==================================================

            elif controller == "marl":

                actions = []

                for agent_id in range(NUM_AGENTS):

                    q_values = agents[agent_id].get_q_values(
                        observations[agent_id]
                    )

                    action = int(np.argmax(q_values))

                    actions.append(action)

            # ==================================================
            # DQN-MARL
            # ==================================================

            elif controller == "dqn":

                actions = []

                for agent_id in range(NUM_AGENTS):

                    # Disable exploration during evaluation
                    agents[agent_id].epsilon = 0.0

                    action = agents[agent_id].choose_action(
                        observations[agent_id]
                    )

                    actions.append(action)

            else:
                raise ValueError(
                    f"Unknown controller: {controller}"
                )

            # ==================================================
            # ENVIRONMENT STEP
            # ==================================================

            observations, rewards, done, info = env.step(actions)

            total_queue += info["total_queue"]

            step += 1

        # ======================================================
        # SAME METRIC USED BY evaluate_marl.py
        # ======================================================

        avg_queue_this_episode = total_queue / MAX_STEPS

        episode_queues.append(avg_queue_this_episode)

    return np.mean(episode_queues)


def load_tabular_agents():

    print("Loading Tabular MARL agents...")

    with open("results/marl_agents.pkl", "rb") as file:
        agents = pickle.load(file)

    print("Tabular MARL agents loaded successfully.")

    return agents


def load_dqn_agents():

    print("Loading DQN-MARL agents...")

    with open("results/dqn_marl_agents.pkl", "rb") as file:
        agents = pickle.load(file)

    print("DQN-MARL agents loaded successfully.")

    return agents


def main():

    print("\nRunning controller comparison...\n")

    # ==========================================================
    # LOAD TRAINED AGENTS
    # ==========================================================

    tabular_agents = load_tabular_agents()
    dqn_agents = load_dqn_agents()

    # ==========================================================
    # RANDOM CONTROLLER
    # ==========================================================

    env = MultiAgentTrafficEnv()

    random_avg = evaluate_controller(
        env,
        "random"
    )

    # ==========================================================
    # FIXED-TIME CONTROLLER
    # ==========================================================

    env = MultiAgentTrafficEnv()

    fixed_avg = evaluate_controller(
        env,
        "fixed"
    )

    # ==========================================================
    # TABULAR MARL
    # ==========================================================

    env = MultiAgentTrafficEnv()

    marl_avg = evaluate_controller(
        env,
        "marl",
        tabular_agents
    )

    # ==========================================================
    # DQN-MARL
    # ==========================================================

    env = MultiAgentTrafficEnv()

    dqn_avg = evaluate_controller(
        env,
        "dqn",
        dqn_agents
    )

    # ==========================================================
    # RESULTS
    # ==========================================================

    print("\n")
    print(f"Random Controller:       {random_avg:.2f}")
    print(f"Fixed-Time Controller:   {fixed_avg:.2f}")
    print(f"Tabular MARL Controller: {marl_avg:.2f}")
    print(f"DQN-MARL Controller:     {dqn_avg:.2f}")

    print("\n" + "=" * 60)
    print("           FINAL CONTROLLER COMPARISON")
    print("=" * 60)

    print()
    print(f"Random:       {random_avg:.2f}")
    print(f"Fixed-Time:   {fixed_avg:.2f}")
    print(f"Tabular MARL: {marl_avg:.2f}")
    print(f"DQN-MARL:     {dqn_avg:.2f}")

    # ==========================================================
    # IMPROVEMENT CALCULATIONS
    # ==========================================================

    random_improvement = (
        (random_avg - dqn_avg)
        / random_avg
    ) * 100

    fixed_improvement = (
        (fixed_avg - dqn_avg)
        / fixed_avg
    ) * 100

    marl_improvement = (
        (marl_avg - dqn_avg)
        / marl_avg
    ) * 100

    print("\n" + "=" * 60)
    print("             DQN-MARL IMPROVEMENT")
    print("=" * 60)

    print()
    print(
        f"DQN-MARL improvement over Random: "
        f"{random_improvement:.2f}%"
    )

    print(
        f"DQN-MARL improvement over Fixed:  "
        f"{fixed_improvement:.2f}%"
    )

    print(
        f"DQN-MARL improvement over Tabular MARL: "
        f"{marl_improvement:.2f}%"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()