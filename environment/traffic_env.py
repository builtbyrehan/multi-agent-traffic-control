import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TrafficSignalEnv(gym.Env):
    """
    A simple single-intersection traffic signal environment.

    The agent controls a traffic light and learns to reduce
    the number of waiting vehicles.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Traffic on four approaches:
        # [North, South, East, West]
        self.max_vehicles = 20

        # Observation:
        # 4 traffic queues + current traffic-light phase
        self.observation_space = spaces.Box(
            low=0,
            high=self.max_vehicles,
            shape=(5,),
            dtype=np.float32,
        )

        # Actions:
        # 0 = keep current phase
        # 1 = switch phase
        self.action_space = spaces.Discrete(2)

        self.queues = None
        self.phase = None
        self.time_step = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start with a random number of vehicles
        self.queues = self.np_random.integers(
            low=0,
            high=6,
            size=4,
        ).astype(np.float32)

        # 0 = North/South green
        # 1 = East/West green
        self.phase = 0

        self.time_step = 0

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):

        # Change traffic-light phase if requested
        if action == 1:
            self.phase = 1 - self.phase

        # Generate new vehicles
        arrivals = self.np_random.integers(
            low=0,
            high=3,
            size=4,
        )

        self.queues += arrivals

        # Determine which roads currently have green
        if self.phase == 0:
            green_lanes = [0, 1]  # North/South
        else:
            green_lanes = [2, 3]  # East/West

        # Vehicles pass through green lanes
        for lane in green_lanes:
            vehicles_processed = min(self.queues[lane], 2)
            self.queues[lane] -= vehicles_processed

        # Prevent queues exceeding maximum
        self.queues = np.clip(
            self.queues,
            0,
            self.max_vehicles,
        )

        # Reward = negative total queue
        total_queue = np.sum(self.queues)
        reward = -float(total_queue)

        self.time_step += 1

        terminated = False
        truncated = self.time_step >= 100

        observation = self._get_observation()

        info = {
            "total_queue": float(total_queue),
            "phase": self.phase,
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self):

        return np.array(
            [
                self.queues[0],
                self.queues[1],
                self.queues[2],
                self.queues[3],
                self.phase,
            ],
            dtype=np.float32,
        )

    def render(self):

        phase_name = (
            "North/South GREEN"
            if self.phase == 0
            else "East/West GREEN"
        )

        print(
            f"Time: {self.time_step:03d} | "
            f"Queues: {self.queues.astype(int)} | "
            f"Signal: {phase_name}"
        )