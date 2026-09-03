# 🚦 Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control

> **An intelligent traffic management system where multiple reinforcement learning agents independently control traffic signals and learn to coordinate with neighboring intersections to reduce congestion, waiting time, and traffic delays.**

---

## 📌 Overview

Traffic congestion is a major problem in modern cities. Traditional traffic lights typically operate using fixed schedules or manually engineered rules, which cannot effectively adapt to changing traffic conditions.

This project explores a different approach:

**What if every traffic intersection could act as an autonomous AI agent that observes its local environment, makes decisions, learns from the consequences, and coordinates with neighboring intersections?**

This project uses **Reinforcement Learning (RL)** and **Multi-Agent Reinforcement Learning (MARL)** to develop an adaptive traffic signal control system.

Each intersection is modeled as an autonomous agent.

The agents observe traffic conditions, select traffic-light actions, receive rewards based on traffic performance, and gradually learn policies that reduce congestion.

The project will progressively evolve from a simple single-agent RL environment into a cooperative **Multi-Agent RL system using MAPPO**.

---

# 🎯 Objectives

The main objectives are:

- Build a simulated traffic environment.
- Implement a single-agent RL traffic controller.
- Extend the system to multiple autonomous traffic-light agents.
- Implement cooperative Multi-Agent Reinforcement Learning.
- Experiment with different RL algorithms.
- Investigate communication and coordination between agents.
- Compare traditional traffic control with learned policies.
- Measure improvements using quantitative traffic metrics.
- Visualize agent behavior and learning progress.
- Develop a modular foundation for future autonomous multi-agent systems.

---

# 🧠 Core Idea

The system can be represented as:

```text
                    TRAFFIC ENVIRONMENT
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   INTERSECTION A     INTERSECTION B     INTERSECTION C
        │                  │                  │
        ▼                  ▼                  ▼
    RL AGENT A          RL AGENT B          RL AGENT C
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    Coordination
                           │
                           ▼
                  Adaptive Traffic Flow
```

Each traffic signal operates as an autonomous decision-making agent.

The overall learning loop is:

```text
Observe Traffic
      ↓
Determine State
      ↓
Select Action
      ↓
Change Traffic Signal
      ↓
Traffic Evolves
      ↓
Receive Reward
      ↓
Update Policy
      ↓
Repeat
```

---

# 🚦 Problem Formulation

The traffic control problem is modeled as a **Markov Decision Process (MDP)** for the single-agent case and as a **Multi-Agent Markov Decision Process (MMDP / Dec-POMDP)** for the multi-agent case.

Each agent interacts with the environment using:

### State

The state represents the current traffic conditions around an intersection.

Possible observations include:

- Number of vehicles on incoming lanes
- Queue length
- Average waiting time
- Current traffic-light phase
- Time spent in the current phase
- Vehicle density
- Neighboring intersection traffic

Example:

```text
State =
[
    lane_1_density,
    lane_2_density,
    lane_3_density,
    lane_4_density,
    queue_length,
    current_phase
]
```

---

### Action

The agent controls the traffic signal.

For the initial implementation:

```text
Action 0 → Keep current phase
Action 1 → Switch traffic phase
```

Later implementations may use:

```text
Action 0 → North/South green
Action 1 → East/West green
Action 2 → Extend current phase
Action 3 → Switch phase
```

---

### Reward

The primary objective is to minimize traffic congestion.

A simple reward function can be:

```text
Reward = - Total Waiting Time
```

A more advanced reward can combine multiple objectives:

```text
Reward =
    - α × Waiting Time
    - β × Queue Length
    + γ × Throughput
```

where:

- `α` controls the importance of waiting time
- `β` controls the importance of queue length
- `γ` controls the importance of throughput

The reward function will be experimentally tuned.

---

# 🤖 Multi-Agent Formulation

In the multi-agent version, each intersection becomes an autonomous agent.

For example:

```text
                  Agent B
                    🚦
                    │
                    │
             ───────┼───────
                    │
        Agent A 🚦──┼──🚦 Agent C
                    │
                    │
                    🚦
                  Agent D
```

Each agent:

1. Observes its local traffic.
2. Makes a traffic-light decision.
3. Receives a reward.
4. Learns from the outcome.
5. May exchange information with neighboring agents.

The goal is not only to optimize individual intersections but to improve the **overall traffic network**.

---

# 🧪 Experimental Progression

The project will be developed incrementally.

## Phase 1 — Environment

Build a simple traffic intersection simulation.

Goals:

- Create lanes
- Generate vehicles
- Implement traffic signals
- Simulate vehicle movement
- Track traffic metrics

---

## Phase 2 — Baseline Controller

Implement traditional traffic control.

Examples:

### Fixed-Time Control

Traffic lights change according to a predefined schedule.

This becomes the baseline against which RL agents are evaluated.

Metrics:

- Average waiting time
- Queue length
- Throughput
- Travel time

---

## Phase 3 — Single-Agent Q-Learning

Train one RL agent to control one intersection.

Architecture:

```text
Traffic Environment
        ↓
      State
        ↓
   Q-Learning Agent
        ↓
      Action
        ↓
Traffic Signal
        ↓
     Reward
        ↓
   Q-Table Update
```

This phase introduces the fundamental RL concepts.

---

## Phase 4 — Multi-Agent RL

Extend the environment to multiple intersections.

```text
             Environment
                  │
       ┌──────────┼──────────┐
       ↓          ↓          ↓
    Agent A    Agent B    Agent C
       │          │          │
       └──────────┼──────────┘
                  ↓
              Environment
```

Each intersection now has its own agent.

The agents initially learn independently.

---

## Phase 5 — Deep Reinforcement Learning

Replace the Q-table with neural networks.

Potential algorithms:

- DQN
- PPO

This allows the system to handle larger state spaces.

---

## Phase 6 — Cooperative MARL

Introduce explicit coordination between agents.

Agents may receive information about:

- Neighboring traffic density
- Neighboring queue lengths
- Incoming traffic
- Current neighboring signal phases

Example:

```text
Agent A
   │
   │ Traffic information
   ▼
Agent B
```

The goal is to determine whether communication and coordination improve global traffic performance.

---

# 🧠 MAPPO

The final major algorithmic stage will investigate:

> **Multi-Agent Proximal Policy Optimization (MAPPO)**

MAPPO is a multi-agent extension of PPO commonly used for cooperative MARL environments.

The project will investigate a **Centralized Training, Decentralized Execution (CTDE)** approach.

Conceptually:

```text
                 CENTRALIZED CRITIC
                        │
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
    Agent A          Agent B          Agent C
    Policy           Policy           Policy
        │               │               │
        ↓               ↓               ↓
 Traffic Signal   Traffic Signal   Traffic Signal
```

During training, the critic can use broader information about the environment.

During execution, each agent can make decisions using its local observations.

---

# 📊 Evaluation

The system will be evaluated using quantitative metrics.

## Primary Metrics

### Average Waiting Time

Measures how long vehicles remain waiting in traffic.

```text
Lower = Better
```

### Average Queue Length

Measures the average number of vehicles waiting at intersections.

```text
Lower = Better
```

### Throughput

Number of vehicles successfully passing through the network.

```text
Higher = Better
```

### Average Travel Time

Measures the time required for vehicles to reach their destinations.

```text
Lower = Better
```

### Collision Rate

Measures whether vehicles experience collisions within the simulation.

```text
Lower = Better
```

---

# 📈 Experiments

The project will compare multiple approaches.

| Approach | Agents | Learning | Coordination |
|---|---:|---|---|
| Fixed-Time Control | No | No | No |
| Random Control | No | No | No |
| Q-Learning | 1 | RL | No |
| Independent Q-Learning | Multiple | RL | Limited |
| DQN | 1 | Deep RL | No |
| PPO | 1 | Deep RL | No |
| Cooperative MARL | Multiple | MARL | Yes |
| MAPPO | Multiple | MARL | Yes |

The final results will compare the algorithms using:

- Waiting time
- Queue length
- Throughput
- Travel time
- Reward
- Training stability

---

# 🛠️ Technology Stack

## Programming

- Python

## Reinforcement Learning

- PyTorch
- Gymnasium
- Stable-Baselines3
- PettingZoo

## Traffic Simulation

The project can use:

- **SUMO (Simulation of Urban MObility)**

SUMO provides a realistic microscopic traffic simulation environment.

## Data & Visualization

- NumPy
- Pandas
- Matplotlib
- Seaborn

## Experiment Tracking

Potential tools:

- TensorBoard
- Weights & Biases

## Development

- VS Code
- Jupyter Notebook
- Git
- GitHub

---

# 🏗️ Proposed Architecture

```text
                         ┌─────────────────────┐
                         │   Traffic Simulator  │
                         │        (SUMO)        │
                         └──────────┬──────────┘
                                    │
                              Observations
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   ▼                   ▼
           ┌─────────┐         ┌─────────┐         ┌─────────┐
           │ Agent A │         │ Agent B │         │ Agent C │
           └────┬────┘         └────┬────┘         └────┬────┘
                │                   │                   │
                ▼                   ▼                   ▼
             Policy A            Policy B            Policy C
                │                   │                   │
                └───────────────────┼───────────────────┘
                                    │
                                 Actions
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Traffic Environment │
                         └──────────┬──────────┘
                                    │
                                  Reward
                                    │
                                    ▼
                            Learning Algorithm
                                    │
                                    ▼
                              Policy Update
```

---

# 📁 Project Structure

```text
multi-agent-traffic-control/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── configs/
│   ├── environment.yaml
│   ├── training.yaml
│   └── experiment.yaml
│
├── environment/
│   ├── __init__.py
│   ├── traffic_env.py
│   ├── intersection.py
│   ├── vehicle.py
│   └── reward.py
│
├── agents/
│   ├── __init__.py
│   ├── q_learning.py
│   ├── dqn.py
│   ├── ppo.py
│   └── mappo.py
│
├── training/
│   ├── train_qlearning.py
│   ├── train_dqn.py
│   ├── train_ppo.py
│   └── train_mappo.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   └── compare_models.py
│
├── visualization/
│   ├── traffic_visualizer.py
│   ├── plot_rewards.py
│   └── plot_metrics.py
│
├── experiments/
│   ├── baseline/
│   ├── single_agent/
│   └── multi_agent/
│
├── notebooks/
│   ├── 01_environment.ipynb
│   ├── 02_q_learning.ipynb
│   ├── 03_multi_agent.ipynb
│   └── 04_mappo.ipynb
│
└── results/
    ├── models/
    ├── figures/
    └── logs/
```

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/multi-agent-traffic-control.git

cd multi-agent-traffic-control
```

---

## 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🧪 Initial Training

The first implementation will train a simple Q-learning agent:

```bash
python training/train_qlearning.py
```

Later:

```bash
python training/train_dqn.py
```

and:

```bash
python training/train_mappo.py
```

---

# 📊 Results

Results will be added as experiments are completed.

Example target comparison:

```text
Average Waiting Time

Fixed-Time        ████████████████████
Q-Learning        ███████████████
Independent MARL  ███████████
MAPPO             ███████
```

Actual values will be reported from experimental results rather than predetermined.

---

# 🔬 Research Questions

This project will investigate several questions:

### RQ1

Can reinforcement learning outperform traditional fixed-time traffic signal control?

### RQ2

Does increasing the number of autonomous traffic agents improve network-level traffic management?

### RQ3

Does communication between neighboring agents improve performance?

### RQ4

How does independent learning compare with cooperative MARL?

### RQ5

Does MAPPO provide better coordination and training stability than simpler approaches?

### RQ6

How does the system perform under different traffic-demand patterns?

---

# 🌦️ Generalization Experiments

To make the project more realistic, agents will eventually be tested under different traffic conditions:

### Low Traffic

```text
🚗       🚗
```

### Medium Traffic

```text
🚗 🚗 🚙 🚗 🚙
```

### Heavy Traffic

```text
🚗 🚙 🚗 🚕 🚗 🚙 🚗 🚕 🚗
```

### Changing Traffic

The traffic distribution changes during an episode.

This tests whether the learned agents can adapt rather than simply memorize a fixed traffic pattern.

---

# 🧠 Future Extensions

The project can eventually evolve into a more advanced autonomous system.

### 1. Communication Learning

Allow agents to learn **what information to communicate and when to communicate it**.

### 2. Hierarchical Control

Introduce a higher-level coordinator:

```text
             City-Level Agent
                    │
          ┌─────────┼─────────┐
          ↓         ↓         ↓
       Zone A     Zone B     Zone C
          │         │         │
       Agents     Agents     Agents
```

### 3. LLM-Based Planning

An LLM could potentially operate as a high-level planning layer while RL agents handle low-level traffic control.

```text
                 LLM Planner
                     │
              Traffic Strategy
                     │
                     ▼
                 MARL Layer
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
     Agent A      Agent B      Agent C
        │            │            │
        ▼            ▼            ▼
    Traffic       Traffic      Traffic
    Signal        Signal       Signal
```

### 4. Real-Time Dashboard

Build a web dashboard showing:

- Current traffic
- Agent actions
- Queue lengths
- Waiting time
- Rewards
- Network throughput

Potential stack:

```text
React
   ↓
FastAPI
   ↓
MARL Environment
   ↓
SUMO
```

---

# 🏆 Project Goals

The final system aims to demonstrate:

- Reinforcement Learning
- Deep Reinforcement Learning
- Multi-Agent Reinforcement Learning
- Autonomous Decision Making
- Agent Coordination
- Cooperative Learning
- Simulation
- Experiment Design
- Model Evaluation
- AI Engineering

---

# 📚 Learning Path

The project is intentionally structured as a learning progression:

```text
Python
  ↓
MDP
  ↓
Q-Learning
  ↓
Deep RL
  ↓
DQN
  ↓
PPO
  ↓
Multi-Agent RL
  ↓
Coordination
  ↓
CTDE
  ↓
MAPPO
  ↓
Advanced Agentic Systems
```

---

# ⚠️ Project Philosophy

This project is **not intended to simply reproduce an existing tutorial**.

Existing implementations and research will be used as references for learning and benchmarking.

The goal is to:

1. Understand the underlying algorithms.
2. Build the environment ourselves where practical.
3. Establish meaningful baselines.
4. Run controlled experiments.
5. Analyze the results.
6. Document limitations.
7. Incrementally introduce more advanced MARL techniques.

The emphasis is on **understanding and experimentation**, not simply obtaining a working demo.

---

# 👨‍💻 Author

**Rehan**

Computer Science Student  
Interested in:

- Artificial Intelligence
- Machine Learning
- Reinforcement Learning
- Multi-Agent Systems
- Agentic AI
- LLM Engineering
- Autonomous Systems

---

# ⭐ Vision

The long-term goal of this project is to explore how autonomous AI agents can learn to make decisions, coordinate with one another, and optimize complex environments.

The traffic-control environment serves as the first step toward understanding more sophisticated **Multi-Agent AI and autonomous systems**.

```text
Single Agent
     ↓
Reinforcement Learning
     ↓
Multiple Agents
     ↓
Coordination
     ↓
Multi-Agent RL
     ↓
Planning
     ↓
Autonomous AI
     ↓
Agentic AI Systems
```

---

## 📜 License

This project is licensed under the MIT License.