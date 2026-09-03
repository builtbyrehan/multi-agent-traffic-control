from environment.multi_agent_env import MultiAgentTrafficEnv
import numpy as np

env = MultiAgentTrafficEnv()

observations = env.reset(seed=42)

print("Initial observations:")

for agent_id, observation in enumerate(observations):

    print(
        f"Agent {agent_id}: {observation}"
    )

print("\nEnvironment queues:")
print(env.queues)

print("\nAgent 0 sees Agent 1 queue:",
      np.sum(env.queues[1]))

print("Agent 1 sees Agent 0 queue:",
      np.sum(env.queues[0]))