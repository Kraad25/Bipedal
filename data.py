import math
import numpy as np

from Box2D.b2 import (
    fixtureDef,
    polygonShape,
)

# Rendering Constants
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
FRICTION = 2.5

# Box2D Constants
SCALE = 30.0  # Scale factor for Box2D units to pixels
HULL_POLY = [(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE


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


INITIAL_RANDOM_FORCE = 5
MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
#MAX_TILT_ANGLE = 0.5  # radians

MAX_STEPS = 500

# Standing Pose
# Observation Space
STANDING_OBSERVATION_LOW = np.array([
    # Hull
    -math.pi, # Hull angle
    -3.0, # Hull angular velocity
    -3.0, # Hull linear velocity in X
    -3.0, # Hull linear velocity in Y

    # Joints
    -math.pi, # Left Hip angle
    -3.0, # Left Hip angular velocity

    -math.pi, # Left Knee angle
    -3.0, # Left Knee angular velocity
    -0.0, # Left Foot contact (0.0 = no contact, 1.0 = contact)

    -math.pi, # Right Hip angle
    -3.0, # Right Hip angular velocity

    -math.pi, # Right Knee angle
    -3.0, # Right Knee angular velocity
    -0.0, # Right Foot contact
]).astype(np.float32)

STANDING_OBSERVATION_HIGH = np.array([
    # Hull
    math.pi, # Hull angle
    3.0, # Hull angular velocity
    3.0, # Hull linear velocity in X
    3.0, # Hull linear velocity in Y

    # Joints
    math.pi, # Left Hip angle
    3.0, # Left Hip angular velocity

    0.0, # Left Knee angle
    3.0, # Left Knee angular velocity
    1.0, # Left Foot contact

    math.pi, # Right Hip angle
    3.0, # Right Hip angular velocity

    0.0, # Right Knee angle
    3.0, # Right Knee angular velocity
    1.0, # Right Foot contact
]).astype(np.float32)

# Action Space
# Left-Hip, Left-Knee, Right-Hip, Right-Knee
STANDING_ACTION_LOW = np.array([-1, -1, -1, -1]).astype(np.float32)
STANDING_ACTION_HIGH = np.array([1, 0, 1, 0]).astype(np.float32)