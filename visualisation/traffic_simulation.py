import pickle
import random
import sys

import numpy as np
import torch
import pygame

from environment.multi_agent_env import MultiAgentTrafficEnv




# ============================================================
# CONFIGURATION
# ============================================================

WIDTH = 1500
HEIGHT = 850

FPS = 10

ROAD_COLOR = (55, 55, 55)
BACKGROUND_COLOR = (235, 235, 235)
LANE_COLOR = (245, 245, 245)

BLACK = (20, 20, 20)
WHITE = (255, 255, 255)

RED = (220, 50, 50)
GREEN = (50, 190, 90)
YELLOW = (245, 190, 50)

BLUE = (40, 120, 220)
DARK_BLUE = (25, 70, 130)

PANEL_COLOR = (248, 248, 248)

CAR_COLORS = [
    (40, 110, 200),
    (220, 70, 70),
    (50, 160, 90),
    (230, 150, 40),
    (130, 70, 180),
]


# ============================================================
# LOAD TRAINED AGENTS
# ============================================================

print("Loading trained DQN-MARL agents...")

try:

    with open(
        "results/dqn_marl_agents.pkl",
        "rb"
    ) as file:

        agents = pickle.load(file)

    print("DQN-MARL agents loaded successfully.")

except FileNotFoundError:

    print(
        "\nERROR: results/dqn_marl_agents.pkl "
        "was not found."
    )

    sys.exit()


# ============================================================
# INITIALIZE PYGAME
# ============================================================

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "DQN-MARL Multi-Agent Traffic Control"
)

clock = pygame.time.Clock()


# ============================================================
# FONTS
# ============================================================

TITLE_FONT = pygame.font.SysFont(
    "Arial",
    28,
    bold=True
)

HEADER_FONT = pygame.font.SysFont(
    "Arial",
    22,
    bold=True
)

NORMAL_FONT = pygame.font.SysFont(
    "Arial",
    18
)

SMALL_FONT = pygame.font.SysFont(
    "Arial",
    15
)


# ============================================================
# ENVIRONMENT
# ============================================================

env = MultiAgentTrafficEnv()


# ============================================================
# GLOBAL SIMULATION VARIABLES
# ============================================================

observations = None

simulation_time = 0

network_reward = 0.0

episode_reward = 0.0

last_actions = [0, 0]

last_q_values = [
    [0.0, 0.0],
    [0.0, 0.0]
]

running = True


# ============================================================
# VEHICLE VISUAL STATE
# ============================================================

vehicles = []


def create_vehicle(
    intersection,
    direction
):

    return {
        "intersection": intersection,
        "direction": direction,
        "progress": random.uniform(
            0,
            0.8
        ),
        "speed": random.uniform(
            0.015,
            0.035
        ),
        "color": random.choice(
            CAR_COLORS
        )
    }


def initialize_visual_vehicles():

    global vehicles

    vehicles = []

    for intersection in range(2):

        for direction in range(4):

            for _ in range(4):

                vehicles.append(
                    create_vehicle(
                        intersection,
                        direction
                    )
                )


initialize_visual_vehicles()


# ============================================================
# RESET SIMULATION
# ============================================================

def reset_simulation():

    global observations
    global simulation_time
    global network_reward
    global episode_reward
    global last_actions
    global last_q_values

    observations = env.reset(
        seed=random.randint(
            0,
            1000000
        )
    )

    simulation_time = 0

    network_reward = 0.0

    episode_reward = 0.0

    last_actions = [0, 0]

    last_q_values = [
        [0.0, 0.0],
        [0.0, 0.0]
    ]

    initialize_visual_vehicles()


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(
    text,
    x,
    y,
    font=NORMAL_FONT,
    color=BLACK
):

    surface = font.render(
        str(text),
        True,
        color
    )

    screen.blit(
        surface,
        (x, y)
    )


# ============================================================
# DRAW ROAD
# ============================================================

def draw_road_network():

    # Horizontal road
    pygame.draw.rect(
        screen,
        ROAD_COLOR,
        (
            0,
            350,
            WIDTH,
            180
        )
    )

    # Vertical roads
    pygame.draw.rect(
        screen,
        ROAD_COLOR,
        (
            400,
            0,
            180,
            HEIGHT
        )
    )

    pygame.draw.rect(
        screen,
        ROAD_COLOR,
        (
            1070,
            0,
            180,
            HEIGHT
        )
    )

    # Horizontal lane markings

    for x in range(
        0,
        WIDTH,
        50
    ):

        pygame.draw.rect(
            screen,
            LANE_COLOR,
            (
                x,
                438,
                30,
                4
            )
        )

    # Vertical lane markings

    for y in range(
        0,
        HEIGHT,
        50
    ):

        pygame.draw.rect(
            screen,
            LANE_COLOR,
            (
                488,
                y,
                4,
                30
            )
        )

        pygame.draw.rect(
            screen,
            LANE_COLOR,
            (
                1158,
                y,
                4,
                30
            )
        )


# ============================================================
# INTERSECTION COORDINATES
# ============================================================

INTERSECTIONS = [
    (490, 440),
    (1160, 440)
]


# ============================================================
# DRAW TRAFFIC LIGHT
# ============================================================

def draw_light(
    x,
    y,
    color,
    label
):

    pygame.draw.circle(
        screen,
        BLACK,
        (x, y),
        18
    )

    pygame.draw.circle(
        screen,
        color,
        (x, y),
        13
    )

    draw_text(
        label,
        x - 15,
        y - 42,
        SMALL_FONT
    )


# ============================================================
# DRAW INTERSECTION
# ============================================================

def draw_intersection(
    agent_id
):

    center_x, center_y = INTERSECTIONS[
        agent_id
    ]

    phase = int(
        env.phases[agent_id]
    )

    # Intersection label

    draw_text(
        f"Agent {agent_id}",
        center_x - 42,
        center_y - 12,
        HEADER_FONT,
        WHITE
    )

    # Traffic lights

    if phase == 0:

        # N/S green
        draw_light(
            center_x - 65,
            center_y - 80,
            GREEN,
            "N/S"
        )

        draw_light(
            center_x + 65,
            center_y - 80,
            RED,
            "E/W"
        )

    else:

        # E/W green
        draw_light(
            center_x - 65,
            center_y - 80,
            RED,
            "N/S"
        )

        draw_light(
            center_x + 65,
            center_y - 80,
            GREEN,
            "E/W"
        )


# ============================================================
# VEHICLE POSITIONS
# ============================================================

def vehicle_position(
    intersection,
    direction,
    progress
):

    cx, cy = INTERSECTIONS[
        intersection
    ]

    distance = 160 + (
        progress * 500
    )

    # North
    if direction == 0:

        return (
            cx - 25,
            cy - distance
        )

    # South
    if direction == 1:

        return (
            cx + 25,
            cy + distance
        )

    # East
    if direction == 2:

        return (
            cx + distance,
            cy + 25
        )

    # West
    return (
        cx - distance,
        cy - 25
    )


# ============================================================
# UPDATE VEHICLES
# ============================================================

def update_vehicles():

    for vehicle in vehicles:

        intersection = vehicle[
            "intersection"
        ]

        direction = vehicle[
            "direction"
        ]

        phase = int(
            env.phases[intersection]
        )

        green = False

        if phase == 0:

            if direction in [0, 1]:
                green = True

        else:

            if direction in [2, 3]:
                green = True

        if green:

            vehicle["progress"] += (
                vehicle["speed"]
            )

        else:

            vehicle["progress"] += (
                vehicle["speed"] * 0.15
            )

        if vehicle["progress"] > 1:

            vehicle["progress"] = 0

            vehicle["direction"] = random.randint(
                0,
                3
            )

            vehicle["intersection"] = random.randint(
                0,
                1
            )

            vehicle["color"] = random.choice(
                CAR_COLORS
            )


# ============================================================
# DRAW VEHICLES
# ============================================================

def draw_vehicles():

    for vehicle in vehicles:

        x, y = vehicle_position(
            vehicle["intersection"],
            vehicle["direction"],
            vehicle["progress"]
        )

        # Only draw vehicles inside screen

        if (
            -30 < x < WIDTH + 30
            and
            -30 < y < HEIGHT + 30
        ):

            pygame.draw.rect(
                screen,
                vehicle["color"],
                (
                    int(x),
                    int(y),
                    16,
                    10
                ),
                border_radius=3
            )


# ============================================================
# DRAW QUEUE VALUES
# ============================================================

def draw_queue_values(
    agent_id
):

    cx, cy = INTERSECTIONS[
        agent_id
    ]

    queues = env.queues[
        agent_id
    ]

    draw_text(
        f"N: {queues[0]}",
        cx - 15,
        cy - 170,
        SMALL_FONT
    )

    draw_text(
        f"S: {queues[1]}",
        cx - 15,
        cy + 155,
        SMALL_FONT
    )

    draw_text(
        f"W: {queues[2]}",
        cx - 205,
        cy - 10,
        SMALL_FONT
    )

    draw_text(
        f"E: {queues[3]}",
        cx + 170,
        cy - 10,
        SMALL_FONT
    )


# ============================================================
# DRAW INFORMATION PANEL
# ============================================================

def draw_information_panel():

    pygame.draw.rect(
        screen,
        PANEL_COLOR,
        (
            20,
            20,
            350,
            300
        ),
        border_radius=10
    )

    draw_text(
        "DQN-MARL Traffic Control",
        40,
        40,
        TITLE_FONT
    )

    draw_text(
        f"Simulation Time: {simulation_time}",
        40,
        85
    )

    total_queue = int(
        np.sum(env.queues)
    )

    draw_text(
        f"Network Queue: {total_queue}",
        40,
        115
    )

    draw_text(
        f"Network Reward: {network_reward:.2f}",
        40,
        145
    )

    draw_text(
        f"Episode Reward: {episode_reward:.2f}",
        40,
        175
    )

    draw_text(
        "Controls:",
        40,
        215,
        HEADER_FONT
    )

    draw_text(
        "R = Restart Episode",
        40,
        250,
        SMALL_FONT
    )

    draw_text(
        "ESC = Exit",
        200,
        250,
        SMALL_FONT
    )


# ============================================================
# DRAW AGENT INFORMATION
# ============================================================

def draw_agent_panel(
    agent_id,
    x,
    y
):

    pygame.draw.rect(
        screen,
        PANEL_COLOR,
        (
            x,
            y,
            470,
            135
        ),
        border_radius=8
    )

    draw_text(
        f"Agent {agent_id}",
        x + 15,
        y + 10,
        HEADER_FONT
    )

    queues = env.queues[
        agent_id
    ]

    phase = env.phases[
        agent_id
    ]

    action = last_actions[
        agent_id
    ]

    q0 = last_q_values[
        agent_id
    ][0]

    q1 = last_q_values[
        agent_id
    ][1]

    phase_name = (
        "North/South GREEN"
        if phase == 0
        else
        "East/West GREEN"
    )

    action_name = (
        "KEEP"
        if action == 0
        else
        "SWITCH"
    )

    draw_text(
        f"Queues: N={queues[0]} "
        f"S={queues[1]} "
        f"E={queues[2]} "
        f"W={queues[3]}",
        x + 15,
        y + 45,
        SMALL_FONT
    )

    draw_text(
        f"Phase: {phase_name}",
        x + 15,
        y + 70,
        SMALL_FONT
    )

    draw_text(
        f"Action: {action_name}",
        x + 15,
        y + 95,
        SMALL_FONT
    )

    draw_text(
        f"Q(KEEP): {q0:.2f}",
        x + 250,
        y + 45,
        SMALL_FONT
    )

    draw_text(
        f"Q(SWITCH): {q1:.2f}",
        x + 250,
        y + 70,
        SMALL_FONT
    )

# ============================================================
# CHOOSE DQN ACTIONS
# ============================================================

def choose_actions():

    actions = []
    q_values_all = []

    for agent_id in range(2):

        state = np.asarray(
            observations[agent_id],
            dtype=np.float32
        )

        # Convert observation to PyTorch tensor
        import torch

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=agents[agent_id].device
        ).unsqueeze(0)

        # DQN inference only
        with torch.no_grad():

            q_values_tensor = agents[
                agent_id
            ].q_network(
                state_tensor
            )

        q_values = (
            q_values_tensor
            .squeeze(0)
            .cpu()
            .numpy()
        )

        # Select best learned action
        action = int(
            np.argmax(q_values)
        )

        actions.append(action)

        q_values_all.append(
            q_values
        )

    return actions, q_values_all

# ============================================================
# STEP ENVIRONMENT
# ============================================================

def simulation_step():

    global observations
    global simulation_time
    global network_reward
    global episode_reward
    global last_actions
    global last_q_values

    actions, q_values = choose_actions()

    last_actions = actions

    last_q_values = [
        list(q_values[0]),
        list(q_values[1])
    ]

    observations, rewards, done, info = env.step(
        actions
    )

    reward = float(
        np.sum(rewards)
    )

    network_reward += reward

    episode_reward += reward

    simulation_time += 1

    if done:

        pygame.time.delay(
            700
        )

        reset_simulation()


# ============================================================
# INITIAL RESET
# ============================================================

reset_simulation()


# ============================================================
# MAIN LOOP
# ============================================================

while running:

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:

                running = False

            elif event.key == pygame.K_r:

                reset_simulation()

    # --------------------------------------------------------
    # DQN + ENVIRONMENT
    # --------------------------------------------------------

    simulation_step()

    # --------------------------------------------------------
    # VISUAL VEHICLES
    # --------------------------------------------------------

    update_vehicles()

    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    screen.fill(
        BACKGROUND_COLOR
    )

    draw_road_network()

    draw_vehicles()

    draw_intersection(0)

    draw_intersection(1)

    draw_queue_values(0)

    draw_queue_values(1)

    draw_information_panel()

    draw_agent_panel(
        0,
        40,
        650
    )

    draw_agent_panel(
        1,
        780,
        650
    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    draw_text(
        "LIVE DQN-MARL DECISION MAKING",
        570,
        25,
        TITLE_FONT,
        DARK_BLUE
    )

    pygame.display.flip()

    clock.tick(FPS)


# ============================================================
# CLEANUP
# ============================================================

env.close()

pygame.quit()

print("\nDQN-MARL simulation closed successfully.")