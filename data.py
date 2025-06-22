import math
import numpy as np

from Box2D.b2 import (
    fixtureDef,
    polygonShape,
    edgeShape,
)

# Rendering and simulation constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
FPS = 60
RENDER_MODE = "human"
SCALE = 30.0  # Scale factor for Box2D units to pixels

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = SCREEN_HEIGHT / SCALE / 4
TERRAIN_GRASS = 10  # low long are grass spots, in steps
TERRAIN_STARTPAD = 20  # in steps

LIDAR_RANGE = 160 / SCALE
FRICTION = 2.5

# Box2D Constants
SCALE = 30.0  # Scale factor for Box2D units to pixels
HULL_POLY = [(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

# Fixture definitions for Box2D
HULL_FD = fixtureDef(
    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in HULL_POLY]),
    density=5.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001,  # collide only with ground
    restitution=0.0,
)  # 0.99 bouncy

LEG_FD = fixtureDef(
    shape=polygonShape(box=(LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

LOWER_FD = fixtureDef(
    shape=polygonShape(box=(0.8 * LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

TERRAIN_EDGE_FD = fixtureDef(
            shape=edgeShape(vertices=[(0, 0), (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
        )

# Simulation constants
INITIAL_RANDOM_FORCE = 5
MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
#MAX_TILT_ANGLE = 0.5  # radians

# Training constants
MAX_STEPS = 500

# Observation Space
""" Hull, angle, angular velocity, linear velocity in X, linear velocity in Y
    Left Hip, angular velocity
    Left Knee, angular velocity, contact
    Right Hip, angular velocity
    Right Knee, angular velocity, contact
    If Lidar is used, 10 more observations are added for Lidar readings
"""
Observation = {
    "low": np.array([
                -math.pi, -5.0, -5.0, -5.0, # Hull
                -math.pi, -5.0, # Left Hip
                -math.pi, -5.0, -0.0, # Left Knee
                -math.pi, -5.0, # Right Hip
                -math.pi, -5.0, -0.0 # Right Knee
            ] + [-1.0]*10).astype(np.float32),  # 10 Lidar observations
    "high": np.array([
                math.pi, 5.0, 5.0, 5.0, # Hull
                math.pi, 5.0, # Left Hip
                math.pi, 5.0, 1.0, # Left Knee
                math.pi, 5.0, # Right Hip
                math.pi, 5.0, 1.0 # Right Knee
            ] + [1.0]*10).astype(np.float32)  # 10 Lidar observations
}

# Action Space
# Left-Hip, Left-Knee, Right-Hip, Right-Knee
Action = {
    "low": np.array([-1, -1, -1, -1]).astype(np.float32),
    "high": np.array([1, 1, 1, 1]).astype(np.float32)
}