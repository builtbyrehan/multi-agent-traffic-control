import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    Neural network that maps an observation to Q-values
    for each possible action.
    """

    def __init__(self, state_size, action_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, action_size)
        )

    def forward(self, state):
        return self.network(state)


class DQNAgent:
    """
    Deep Q-Network agent.

    Components:
    - Online Q-network
    - Target Q-network
    - Experience replay
    - Epsilon-greedy exploration
    """

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        memory_size=10000,
        batch_size=64,
        target_update_frequency=100,
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        # ---------------------------------------------
        # Experience replay memory
        # ---------------------------------------------

        self.memory = deque(maxlen=memory_size)

        # ---------------------------------------------
        # Device
        # ---------------------------------------------

        self.device = torch.device("cpu")

        # ---------------------------------------------
        # Online Q-network
        # ---------------------------------------------

        self.q_network = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        # ---------------------------------------------
        # Target Q-network
        # ---------------------------------------------

        self.target_network = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        # Initially make both networks identical
        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

        self.target_network.eval()

        # ---------------------------------------------
        # Optimizer
        # ---------------------------------------------

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.MSELoss()

        self.train_steps = 0

    # =================================================
    # ACTION SELECTION
    # =================================================

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.

        During training:
        - Sometimes explores a random action.
        - Sometimes chooses the action with the
          highest Q-value.
        """

        # ---------------------------------------------
        # Exploration
        # ---------------------------------------------

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # ---------------------------------------------
        # Exploitation
        # ---------------------------------------------

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.q_network(
                state_tensor
            )

        return int(
            torch.argmax(q_values).item()
        )

    # =================================================
    # GET Q-VALUES
    # =================================================

    def get_q_values(self, state):
        """
        Return Q-values for all possible actions.

        This method is used during:
        - Evaluation
        - Visualization
        - Testing

        It does NOT perform exploration.

        Example:

            Q-values = [-12.4, -8.7]

        Action 0 -> -12.4
        Action 1 ->  -8.7

        Therefore action 1 is selected because
        -8.7 is greater than -12.4.
        """

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.q_network(
                state_tensor
            )

        return q_values.cpu().numpy()[0]

    # =================================================
    # EXPERIENCE REPLAY MEMORY
    # =================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        Store an experience in replay memory.
        """

        self.memory.append(
            (
                np.array(
                    state,
                    dtype=np.float32
                ),

                action,

                reward,

                np.array(
                    next_state,
                    dtype=np.float32
                ),

                done
            )
        )

    # =================================================
    # TRAINING
    # =================================================

    def replay(self):
        """
        Train the Q-network using a random batch
        from the replay memory.
        """

        # Not enough experiences yet
        if len(self.memory) < self.batch_size:
            return None

        # ---------------------------------------------
        # Sample random batch
        # ---------------------------------------------

        batch = random.sample(
            self.memory,
            self.batch_size
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        # ---------------------------------------------
        # Convert to tensors
        # ---------------------------------------------

        states = torch.tensor(
            np.array(states),
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        # ---------------------------------------------
        # Current Q-values
        # ---------------------------------------------

        current_q_values = self.q_network(
            states
        )

        current_q_values = current_q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # ---------------------------------------------
        # Target Q-values
        # ---------------------------------------------

        with torch.no_grad():

            next_q_values = self.target_network(
                next_states
            )

            max_next_q_values = torch.max(
                next_q_values,
                dim=1
            ).values

            target_q_values = (
                rewards
                + self.gamma
                * max_next_q_values
                * (1 - dones)
            )

        # ---------------------------------------------
        # Calculate loss
        # ---------------------------------------------

        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        # ---------------------------------------------
        # Backpropagation
        # ---------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.train_steps += 1

        # ---------------------------------------------
        # Update target network
        # ---------------------------------------------

        if (
            self.train_steps
            % self.target_update_frequency
            == 0
        ):

            self.target_network.load_state_dict(
                self.q_network.state_dict()
            )

        return loss.item()

    # =================================================
    # EPSILON DECAY
    # =================================================

    def decay_epsilon(self):
        """
        Reduce exploration over time.
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )