import numpy as np


class MultiAgentTrafficEnv:
    """
    Two-intersection cooperative multi-agent traffic environment.

    Agent 0 controls Intersection A.
    Agent 1 controls Intersection B.

    Each agent receives:
        - Its own traffic queues
        - Its own signal phase
        - Neighbor's total queue
        - Neighbor's congestion message

    Observation:
        [N, S, E, W, phase, neighbor_queue, neighbor_message]
    """

    def __init__(self):

        self.num_agents = 2

        # ----------------------------------------------------
        # Traffic queues
        # ----------------------------------------------------

        # [Agent, Direction]
        #
        # Directions:
        # 0 = North
        # 1 = South
        # 2 = East
        # 3 = West

        self.queues = np.zeros(
            (self.num_agents, 4),
            dtype=int
        )


        # ----------------------------------------------------
        # Traffic-light phases
        # ----------------------------------------------------

        # 0 = North/South green
        # 1 = East/West green

        self.phases = np.zeros(
            self.num_agents,
            dtype=int
        )


        # ----------------------------------------------------
        # Environment time
        # ----------------------------------------------------

        self.time_step = 0
        self.max_steps = 100


    # ========================================================
    # RESET
    # ========================================================

    def reset(self, seed=None):
        """
        Reset both intersections.
        """

        if seed is not None:
            np.random.seed(seed)

        self.time_step = 0

        self.queues = np.random.randint(
            0,
            6,
            size=(self.num_agents, 4)
        )

        self.phases = np.zeros(
            self.num_agents,
            dtype=int
        )

        return self._get_observations()


    # ========================================================
    # STEP
    # ========================================================

    def step(self, actions):
        """
        Execute one action for each agent.

        actions:
            [action_agent_0, action_agent_1]

        Action:
            0 = keep current phase
            1 = switch phase
        """

        assert len(actions) == self.num_agents


        # ----------------------------------------------------
        # Apply actions
        # ----------------------------------------------------

        for agent_id, action in enumerate(actions):

            if action == 1:

                self.phases[agent_id] = (
                    1 - self.phases[agent_id]
                )


        # ----------------------------------------------------
        # Simulate traffic
        # ----------------------------------------------------

        self._simulate_traffic()


        self.time_step += 1


        # ----------------------------------------------------
        # Generate new observations
        # ----------------------------------------------------

        observations = self._get_observations()


        # ----------------------------------------------------
        # Calculate rewards
        # ----------------------------------------------------

        rewards = self._get_rewards()


        # ----------------------------------------------------
        # Check termination
        # ----------------------------------------------------

        done = (
            self.time_step >= self.max_steps
        )


        # ----------------------------------------------------
        # Information returned by environment
        # ----------------------------------------------------

        info = {

            "queues":
                self.queues.copy(),

            "phases":
                self.phases.copy(),

            "total_queue":
                int(np.sum(self.queues)),

        }


        return (
            observations,
            rewards,
            done,
            info
        )


    # ========================================================
    # TRAFFIC SIMULATION
    # ========================================================

    def _simulate_traffic(self):
        """
        Simulate vehicle arrivals and departures.
        """

        # ----------------------------------------------------
        # Random vehicle arrivals
        # ----------------------------------------------------

        arrivals = np.random.randint(
            0,
            3,
            size=(self.num_agents, 4)
        )

        self.queues += arrivals


        # ----------------------------------------------------
        # Vehicles pass through green directions
        # ----------------------------------------------------

        for agent_id in range(self.num_agents):

            # ----------------------------------------------
            # North/South green
            # ----------------------------------------------

            if self.phases[agent_id] == 0:

                self.queues[agent_id, 0] = max(
                    0,
                    self.queues[agent_id, 0] - 3
                )

                self.queues[agent_id, 1] = max(
                    0,
                    self.queues[agent_id, 1] - 3
                )


            # ----------------------------------------------
            # East/West green
            # ----------------------------------------------

            else:

                self.queues[agent_id, 2] = max(
                    0,
                    self.queues[agent_id, 2] - 3
                )

                self.queues[agent_id, 3] = max(
                    0,
                    self.queues[agent_id, 3] - 3
                )


    # ========================================================
    # CONGESTION MESSAGE
    # ========================================================

    def _get_congestion_message(self, total_queue):
        """
        Convert a numerical queue into a discrete
        communication message.

        0 = LOW congestion
        1 = MEDIUM congestion
        2 = HIGH congestion
        """

        if total_queue <= 8:

            return 0

        elif total_queue <= 15:

            return 1

        else:

            return 2


    # ========================================================
    # OBSERVATIONS
    # ========================================================

    def _get_observations(self):
        """
        Generate observations for both agents.

        Observation:

        [N, S, E, W,
         phase,
         neighbor_total_queue,
         neighbor_congestion_message]
        """

        observations = []


        for agent_id in range(self.num_agents):

            # ------------------------------------------------
            # Own queues
            # ------------------------------------------------

            own_queues = self.queues[agent_id]


            # ------------------------------------------------
            # Own signal phase
            # ------------------------------------------------

            own_phase = self.phases[agent_id]


            # ------------------------------------------------
            # Identify neighbor
            # ------------------------------------------------

            neighbor_id = 1 - agent_id


            # ------------------------------------------------
            # Neighbor's total queue
            # ------------------------------------------------

            neighbor_total_queue = int(
                np.sum(
                    self.queues[neighbor_id]
                )
            )


            # ------------------------------------------------
            # Neighbor's communication message
            # ------------------------------------------------

            neighbor_message = (
                self._get_congestion_message(
                    neighbor_total_queue
                )
            )


            # ------------------------------------------------
            # Construct observation
            # ------------------------------------------------

            observation = np.append(
                own_queues,
                [
                    own_phase,
                    neighbor_total_queue,
                    neighbor_message
                ]
            )


            observations.append(
                observation
            )


        return observations


    # ========================================================
    # REWARD
    # ========================================================

    def _get_rewards(self):
        """
        Cooperative reward.

        Both agents receive the same reward based on
        total network congestion.
        """

        total_network_queue = np.sum(
            self.queues
        )

        shared_reward = -float(
            total_network_queue
        )

        rewards = [
            shared_reward,
            shared_reward
        ]

        return rewards


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):
        """Close the environment."""
        pass