from environment.traffic_env import TrafficSignalEnv

env = TrafficSignalEnv()

observation,info = env.reset()

print("Initial Observation: ")
print(observation)

for step in range(10):
    #random action for now

    action = env.action_space.sample()

    observation,reward,terminated,truncated,info = env.step(action)

    print(
        f"Step {step + 1}: "
        f"Action = {action},"
        f"Reward = {reward:.2f},"
        f"Queue = {info['total_queue']:.0f}"
    )

    env.render()

    if terminated or truncated:
        break
env.close()