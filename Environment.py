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
from data import *

class BipedalWalkerEnv(gym.Env):
    # Metadata -> Contains information about rendering options.
    metadata = {"render_modes": [RENDER_MODE], "render_fps": FPS}

    def __init__(self, render_mode="human"):
        EzPickle.__init__(self, render_mode)

        self.world = Box2D.b2World(gravity=(0, -10), doSleep=True)
        self.hull = None
        self.terrain = []
        self.cloud_positions = []  # List to store cloud positions
        
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
        self._generate_background()

    def step(self, action):
        pass
    
    def render(self):
            self._render_setup()

            self._draw_background()

            pygame.display.flip()
            self.clock.tick(self.fps)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None

    def _render_setup(self):
        if self.render_mode == "human" and self.render_mode in self.metadata["render_modes"]:
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                self.clock = pygame.time.Clock()

            pygame.display.set_caption("Bipedal Walker")
            self.screen.fill((25, 189, 255))

    
    def _generate_terrain(self):
        self.terrain = [] # Stores Box2D static bodies for terrain (used for collisions)
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

    def _generate_clouds(self):
        self.cloud_positions = []
        sky_top = VIEWPORT_H / SCALE 
        sky_bottom = sky_top * 0.90  
        for _ in range(5):
            x = self.np_random.uniform(0, VIEWPORT_W/SCALE)
            y = self.np_random.uniform(sky_bottom, sky_top)
            self.cloud_positions.append((x, y))

    def _draw_terrain(self):
        for polygon, color in self.terrain_poly:
            scaled_poly = []
            for x, y in polygon:
                px = x * SCALE
                py = SCREEN_HEIGHT - y * SCALE  # flip Y for Pygame
                scaled_poly.append((px, py))
            pygame.draw.polygon(self.screen, color, scaled_poly)
    
    def _draw_clouds(self):
        cloud_img = pygame.image.load("assets/Cloud.png").convert_alpha()
        self.cloud = pygame.transform.scale(cloud_img, (128, 80))
        for cloud_x, cloud_y in self.cloud_positions:
            screen_x = cloud_x * SCALE
            screen_y = SCREEN_HEIGHT - cloud_y * SCALE
            self.screen.blit(self.cloud,(screen_x, screen_y))
    
    def _get_robot_intial_position(self):
        starting_area = TERRAIN_STARTPAD * TERRAIN_STEP
        init_x = starting_area / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H

        return init_x, init_y

    def _generate_background(self):
        self._generate_terrain()
        self._generate_clouds()

    def _draw_background(self):
        self._draw_terrain()
        self._draw_clouds()

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