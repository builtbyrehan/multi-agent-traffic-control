import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Final Controller Results
# ============================================================

controllers = [
    "Random",
    "Fixed-Time",
    "Tabular MARL",
    "DQN-MARL"
]

queues = [
    14.17,
    16.98,
    15.95,
    5.79
]


# ============================================================
# Calculate Improvements
# ============================================================

dqn_queue = queues[3]

improvement_random = (
    (queues[0] - dqn_queue)
    / queues[0]
    * 100
)

improvement_fixed = (
    (queues[1] - dqn_queue)
    / queues[1]
    * 100
)

improvement_marl = (
    (queues[2] - dqn_queue)
    / queues[2]
    * 100
)


# ============================================================
# Create Bar Chart
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    controllers,
    queues
)


# ============================================================
# Add Values Above Bars
# ============================================================

for bar, value in zip(bars, queues):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{value:.2f}",
        ha="center",
        fontsize=11
    )


# ============================================================
# Labels
# ============================================================

plt.xlabel("Controller")
plt.ylabel("Average Network Queue Per Step")

plt.title(
    "Multi-Agent Traffic Control: Controller Comparison"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# Print Final Analysis
# ============================================================

print("\n" + "=" * 60)
print("           FINAL EXPERIMENTAL RESULTS")
print("=" * 60)

print(
    f"\nRandom Controller:       {queues[0]:.2f}"
)

print(
    f"Fixed-Time Controller:   {queues[1]:.2f}"
)

print(
    f"Tabular MARL Controller: {queues[2]:.2f}"
)

print(
    f"DQN-MARL Controller:     {queues[3]:.2f}"
)

print("\n" + "=" * 60)
print("             DQN-MARL IMPROVEMENT")
print("=" * 60)

print(
    f"\nvs Random:       {improvement_random:.2f}%"
)

print(
    f"vs Fixed-Time:   {improvement_fixed:.2f}%"
)

print(
    f"vs Tabular MARL: {improvement_marl:.2f}%"
)

print("=" * 60)