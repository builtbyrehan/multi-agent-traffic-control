import numpy as np


class QLearningAgent:
    """
    Q-Learning agent for the traffic signal environment.
    """

    def __init__(
        self,
        action_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):

        self.action_size = action_size

        # Learning rate (alpha)
        self.learning_rate = learning_rate

        # Discount factor (gamma)
        self.discount_factor = discount_factor

        # Exploration probability
        self.epsilon = epsilon

        # How quickly exploration decreases
        self.epsilon_decay = epsilon_decay

        # Minimum exploration
        self.epsilon_min = epsilon_min

        # Q-table
        self.q_table = {}

    def _state_to_tuple(self, observation):
        """
        Convert NumPy observation into a hashable tuple.
        """

        return tuple(observation.astype(int))

    def get_q_values(self, observation):
        """
        Return Q-values for a state.
        If the state has never been seen, initialize it.
        """

        state = self._state_to_tuple(observation)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)

        return self.q_table[state]

    def choose_action(self, observation):
        """
        Epsilon-greedy action selection.

        Sometimes explore.
        Otherwise exploit the best known action.
        """

        q_values = self.get_q_values(observation)

        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        # Exploitation
        return int(np.argmax(q_values))

    def update(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
    ):
        """
        Update the Q-value using the Q-learning equation.
        """

        state = self._state_to_tuple(observation)
        next_state = self._state_to_tuple(next_observation)

        # Make sure both states exist
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size)

        current_q = self.q_table[state][action]

        # Best possible future Q-value
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state])

        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward
            + self.discount_factor * max_next_q
            - current_q
        )

        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        """
        Reduce exploration after each episode.
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )