import pickle

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Moving Average
# ============================================================

def moving_average(values, window=50):

    values = np.asarray(values)

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid"
    )


# ============================================================
# Load Training History
# ============================================================

with open(
    "results/dqn_training_history.pkl",
    "rb"
) as file:

    history = pickle.load(file)


rewards = history["rewards"]
losses = history["losses"]


# ============================================================
# Reward Plot
# ============================================================

reward_avg = moving_average(
    rewards,
    window=50
)

plt.figure(figsize=(10, 5))

plt.plot(
    range(50, len(rewards) + 1),
    reward_avg,
    label="50-Episode Moving Average"
)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN-MARL Training Reward")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# Loss Plot
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    losses,
    label="Training Loss"
)

plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("DQN-MARL Training Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# Reward + Loss Summary
# ============================================================

print("=" * 60)
print("          DQN-MARL TRAINING ANALYSIS")
print("=" * 60)

print(f"\nTotal Episodes: {len(rewards)}")

print(
    f"\nInitial Reward: "
    f"{rewards[0]:.2f}"
)

print(
    f"Final Reward:   "
    f"{rewards[-1]:.2f}"
)

print(
    f"\nInitial Loss: "
    f"{losses[0]:.4f}"
)

print(
    f"Final Loss:   "
    f"{losses[-1]:.4f}"
)

print("=" * 60)