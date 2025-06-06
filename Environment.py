import math
from typing import TYPE_CHECKING, List, Optional
import numpy as np

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle
import Box2D
from Box2D.b2 import (
    circleShape,
    contactListener,
    edgeShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
)

import pygame

# Rendering Constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
FPS = 60
RENDER_MODE = "human"
SCALE = 30.0  # Scale factor for Box2D units to pixels

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
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

class BipedalWalkerEnv(gym.Env):
    # Metadata -> Contains information about rendering options.
    metadata = {"render_modes": [RENDER_MODE], "render_fps": FPS}

    def __init__(self, render_mode="human"):
        EzPickle.__init__(self, render_mode)

        self.world = Box2D.b2World(gravity=(0, -10), doSleep=True)
        self.hull = None
        self.terrain = []
        
        self.render_mode = render_mode
        self.fps = self.metadata["render_fps"]
        self.clock = None
        self.screen = None

        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
            friction=FRICTION,
        )

        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0), (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
        )

    def reset(self):
        self._generate_terrain()

    def _generate_terrain(self):
        self.terrain = [] # # Stores Box2D static bodies for terrain (used for collisions)
        self.terrain_x = []
        self.terrain_y = []

        y = TERRAIN_HEIGHT
        delta_y = 0 # Rate of change in height

        for i in range(TERRAIN_LENGTH):
            x = i * TERRAIN_STEP
            self.terrain_x.append(x)    

            if i < TERRAIN_STARTPAD:
                delta_y = 0
            else:
                delta_y = 0.8*delta_y + 0.01*np.sign(TERRAIN_HEIGHT - y)
                delta_y += self.np_random.uniform(-1, 1)/SCALE
            
            y += delta_y

            self.terrain_y.append(y)

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH-1):
            x1, y1 = self.terrain_x[i], self.terrain_y[i]
            x2, y2 = self.terrain_x[i+1], self.terrain_y[i+1]
            
            poly = [(x1, y1), (x2, y2)]

            self.fd_edge.shape.vertices = poly
            terrain = self.world.CreateStaticBody(fixtures = self.fd_edge)
            color = (76, 255 if i % 2 == 0 else 204, 76)
            terrain.color1 = color
            terrain.color2 = color
            self.terrain.append(terrain)
            color = (102, 153, 76)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))     
        self.terrain.reverse() # Reverse the terrain to match Box2D's coordinate system

    def step(self, action):
        pass
    
    def render(self):
        if self.render_mode == "human" and self.render_mode in self.metadata["render_modes"]:
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                self.clock = pygame.time.Clock()

            pygame.display.set_caption("Bipedal Walker")
            self.screen.fill((25, 189, 255))

            # Draw terrain
            for poly, color in self.terrain_poly:
                scaled_poly = []
                for x, y in poly:
                    px = x * SCALE
                    py = SCREEN_HEIGHT - y * SCALE  # flip Y for Pygame
                    scaled_poly.append((px, py))
                pygame.draw.polygon(self.screen, color, scaled_poly)

            pygame.display.flip()
            self.clock.tick(FPS)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None

if __name__ == "__main__":
    env = BipedalWalkerEnv()
    running = True
    env.reset()
    env.render()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        env.render()

    env.close()