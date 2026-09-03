import numpy as np

from agents.dqn_agent import DQNAgent


STATE_SIZE = 6
ACTION_SIZE = 2


agent = DQNAgent(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE
)


state = np.array(
    [3, 4, 2, 4, 0, 9],
    dtype=np.float32
)


print("Initial state:")
print(state)

print("\nInitial epsilon:")
print(agent.epsilon)


action = agent.choose_action(state)

print("\nSelected action:")
print(action)


# Add some experiences
for i in range(100):

    next_state = np.random.randint(
        0,
        10,
        size=STATE_SIZE
    ).astype(np.float32)

    reward = -float(
        np.sum(next_state[:4])
    )

    done = False

    agent.remember(
        state,
        action,
        reward,
        next_state,
        done
    )


print("\nReplay memory size:")
print(len(agent.memory))


loss = agent.replay()

print("\nTraining loss:")
print(loss)


agent.decay_epsilon()

print("\nEpsilon after decay:")
print(agent.epsilon)

print("\nDQN test completed successfully.")