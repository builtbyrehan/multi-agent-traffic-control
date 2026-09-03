import matplotlib.pyplot as plt


# ============================================================
# Final Experimental Results
# ============================================================

controllers = [
    "Random",
    "Fixed-Time",
    "Tabular MARL",
    "DQN-MARL",
    "Communication\nDQN-MARL"
]

queues = [
    14.17,
    16.98,
    15.95,
    5.79,
    5.80
]


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(11, 6))

bars = plt.bar(
    controllers,
    queues
)


# ============================================================
# Values above bars
# ============================================================

for bar, value in zip(bars, queues):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{value:.2f}",
        ha="center",
        fontsize=11
    )


plt.xlabel("Controller")
plt.ylabel("Average Network Queue Per Step")

plt.title(
    "Final Multi-Agent Traffic Control Results"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# Analysis
# ============================================================

dqn = 5.79
random = 14.17
fixed = 16.98
tabular = 15.95
communication = 5.80


print("\n" + "=" * 65)
print("             FINAL EXPERIMENTAL RESULTS")
print("=" * 65)

print(f"\nRandom Controller:              {random:.2f}")
print(f"Fixed-Time Controller:          {fixed:.2f}")
print(f"Tabular MARL Controller:        {tabular:.2f}")
print(f"DQN-MARL Controller:             {dqn:.2f}")
print(f"Communication DQN-MARL:          {communication:.2f}")


print("\n" + "=" * 65)
print("                 IMPROVEMENTS")
print("=" * 65)


print(
    f"\nDQN-MARL vs Random: "
    f"{(random - dqn) / random * 100:.2f}%"
)

print(
    f"DQN-MARL vs Fixed-Time: "
    f"{(fixed - dqn) / fixed * 100:.2f}%"
)

print(
    f"DQN-MARL vs Tabular MARL: "
    f"{(tabular - dqn) / tabular * 100:.2f}%"
)


print("\n" + "=" * 65)
print("             COMMUNICATION ABLATION")
print("=" * 65)

difference = communication - dqn

print(
    f"\nBaseline DQN-MARL:              {dqn:.2f}"
)

print(
    f"Communication DQN-MARL:        {communication:.2f}"
)

print(
    f"Difference:                     {difference:+.2f}"
)

print(
    "\nConclusion:"
)

print(
    "The explicit congestion-message feature "
    "did not produce a meaningful improvement."
)

print("=" * 65)