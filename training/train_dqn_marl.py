import pickle
import numpy as np

from environment.multi_agent_env import MultiAgentTrafficEnv
from agents.dqn_agent import DQNAgent


# ============================================================
# Configuration
# ============================================================

EPISODES = 1000

STATE_SIZE = 7
ACTION_SIZE = 2


# ============================================================
# Create environment
# ============================================================

env = MultiAgentTrafficEnv()


# ============================================================
# Training history
# ============================================================

reward_history = []
loss_history = []


# ============================================================
# Create two DQN agents
# ============================================================

agents = [
    DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ),

    DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )
]


# ============================================================
# Training loop
# ============================================================

for episode in range(EPISODES):

    # Reset environment
    observations = env.reset()

    # Track rewards for both agents
    total_rewards = np.zeros(2)

    # Track losses during this episode
    losses = []

    done = False


    # ========================================================
    # Episode
    # ========================================================

    while not done:

        actions = []


        # ----------------------------------------------------
        # Each agent selects an action
        # ----------------------------------------------------

        for agent_id in range(2):

            action = agents[agent_id].choose_action(
                observations[agent_id]
            )

            actions.append(action)


        # ----------------------------------------------------
        # Environment executes both actions
        # ----------------------------------------------------

        next_observations, rewards, done, info = env.step(
            actions
        )


        # ----------------------------------------------------
        # Store experiences and train agents
        # ----------------------------------------------------

        for agent_id in range(2):

            agents[agent_id].remember(
                observations[agent_id],
                actions[agent_id],
                rewards[agent_id],
                next_observations[agent_id],
                done
            )


            # Train DQN using replay memory
            loss = agents[agent_id].replay()


            if loss is not None:
                losses.append(loss)


            # Track reward
            total_rewards[agent_id] += rewards[agent_id]


        # Move to next observation
        observations = next_observations


    # ========================================================
    # Decay exploration
    # ========================================================

    for agent in agents:
        agent.decay_epsilon()


    # ========================================================
    # Episode statistics
    # ========================================================

    total_reward = np.sum(total_rewards)


    if losses:
        average_loss = np.mean(losses)
    else:
        average_loss = 0.0


    # Save history
    reward_history.append(total_reward)
    loss_history.append(average_loss)


    # ========================================================
    # Display progress
    # ========================================================

    if (episode + 1) % 100 == 0:

        print(
            f"Episode {episode + 1:4d} | "
            f"Reward: {total_reward:9.2f} | "
            f"Loss: {average_loss:10.4f} | "
            f"Epsilon A0: {agents[0].epsilon:.3f} | "
            f"Epsilon A1: {agents[1].epsilon:.3f}"
        )


# ============================================================
# Save training history
# ============================================================

with open(
    "results/dqn_training_history.pkl",
    "wb"
) as file:

    pickle.dump(
        {
            "rewards": reward_history,
            "losses": loss_history
        },
        file
    )


print(
    "\nTraining history saved to "
    "results/dqn_training_history.pkl"
)


# ============================================================
# Close environment
# ============================================================

env.close()


# ============================================================
# Save trained DQN agents
# ============================================================

with open(
    "results/dqn_marl_agents.pkl",
    "wb"
) as file:

    pickle.dump(
        agents,
        file
    )


print("\nDQN-MARL training complete.")

print(
    "Agents saved to "
    "results/dqn_marl_agents.pkl"
)