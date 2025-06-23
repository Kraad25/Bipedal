import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle
import Box2D

import pygame

from data import *
from ContactDetector import ContactDetector
from Lidar import LidarCallback
from Background import Background
from Robot import Robot
from Renderer import Renderer

import time

class BipedalWalkerEnv(gym.Env):
    # Metadata -> Contains information about rendering options.
    metadata = {"render_modes": [RENDER_MODE], "render_fps": FPS}

    def __init__(self, render_mode="human", mode="Standing", MaxSteps=None):
        EzPickle.__init__(self, render_mode)
        
        # Rendering variables
        self.render_mode = render_mode
        self.fps = self.metadata["render_fps"]
        self.clock = None
        self.screen = None

        # Training Variables
        self.mode = mode
        self.max_steps = MaxSteps if MaxSteps is not None else None
        self.current_step = 0
        self.prev_shaping = None        

        # Action and observation spaces
        self.action_space = spaces.Box(Action["low"], Action["high"])
        self.observation_space = spaces.Box(Observation["low"], Observation["high"])

    def reset(self, seed=None):
        super().reset(seed=seed)


        self._reset_episode_variables()
        
        self.background.generate()
        self.robot.create()
        
        # Robot Variables
        self.legs = self.robot.get_legs()
        self.joints = self.robot.get_joints()
        self.hull = self.robot.get_hull()
        self.spawn_x = self.hull.position[0]
        self.position = self.hull.position
        self.velocity = self.hull.linearVelocity

        # if self.mode == "Standing":
        #     self.robot._apply_initial_random_force_to_hull()
            
        return np.array(self._get_state(), dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1

        self._apply_action(action)
        self._simulate_world()


        state = self._get_state()
        reward = self._calculate_reward(state, action)
        terminated, truncated = self._check_done_conditions(state)
        
        return np.array(state, dtype=np.float32), float(reward), terminated, truncated, {}
    
    def render(self):
        self._render_setup()

        self.drawlist = self.background.get_drawables() + self.robot.get_drawables()

        self.renderer.draw_terrain(self.background.get_terrain_polygons(), self.scroll)
        self.renderer.draw_clouds(self.background.get_cloud_positions(), self.cloud_img)
        self.renderer.draw_bodies(self.drawlist, self.scroll)

        self.lidar_render = (self.lidar_render + 1) % 100
        i = self.lidar_render
        self.renderer.render_lidars(self.lidars, i, self.scroll)

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
            self.renderer.set_screen(self.screen)
            self.cloud_img = pygame.image.load("assets/Cloud.png").convert_alpha()
 
    def _apply_action(self, action):
        # Torque Control
        # Left Hip
        self.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[0])) # Speed is in [-1, 1]
        self.joints[0].maxMotorTorque = float(MOTORS_TORQUE * np.clip(abs(action[0]), 0, 1)) # Torque is 0-100% of MOTORS_TORQUE(80)

        # Left Knee
        self.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[1]))
        self.joints[1].maxMotorTorque = float(MOTORS_TORQUE * np.clip(abs(action[1]), 0, 1))

        # Right Hip
        self.joints[2].motorSpeed = float(SPEED_HIP * np.sign(action[2]))
        self.joints[2].maxMotorTorque = float(MOTORS_TORQUE * np.clip(abs(action[2]), 0, 1))

        # Right Knee
        self.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[3]))
        self.joints[3].maxMotorTorque = float(MOTORS_TORQUE * np.clip(abs(action[3]), 0, 1))

    def _simulate_world(self):
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.scroll = self.position.x - (SCREEN_WIDTH / SCALE) / 5 # Offset for scrolling

    def _get_state(self):
        if self.mode == "Standing":
            return self._get_state_standing()
        elif self.mode == "Walking":
            return self._get_state_walking()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
           
    def _get_state_standing(self):
        state = [
            # Hull
            self.hull.angle,
            2.0* self.hull.angularVelocity/self.fps, # Normalized angular velocity(rad/frame)
            0.3*self.velocity.x*(SCREEN_WIDTH / SCALE) / self.fps, # Normalized linear velocity in X
            0.3*self.velocity.y*(SCREEN_HEIGHT / SCALE) / self.fps, # Normalized linear velocity in Y

            # Joints
            # Left Hip
            self.joints[0].angle,
            self.joints[0].speed/SPEED_HIP,

            # Left Knee
            self.joints[1].angle,
            self.joints[1].speed/SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0, # Left Foot contact

            # Right Hip
            self.joints[2].angle,
            self.joints[2].speed/SPEED_HIP,

            # Right Knee
            self.joints[3].angle,
            self.joints[3].speed/SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0, # Right Foot contact
            ] + [1.0]*10
        return state
    
    def _get_state_walking(self):
        # Lidar readings
        for i in range(10):
            self.lidars[i].fraction = 1.0
            self.lidars[i].p1 = self.position 
            self.lidars[i].p2 = (
                self.position [0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                self.position [1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
            )
            self.world.RayCast(self.lidars[i], self.lidars[i].p1, self.lidars[i].p2)

        state = [
            # Hull
            self.hull.angle,
            2.0* self.hull.angularVelocity/self.fps, # Normalized angular velocity(rad/frame)
            0.3*self.velocity.x*(SCREEN_WIDTH / SCALE) / self.fps, # Normalized linear velocity in X
            0.3*self.velocity.y*(SCREEN_HEIGHT / SCALE) / self.fps, # Normalized linear velocity in Y

            # Joints
            # Left Hip
            self.joints[0].angle,
            self.joints[0].speed/SPEED_HIP,

            # Left Knee
            self.joints[1].angle,
            self.joints[1].speed/SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0, # Left Foot contact

            # Right Hip
            self.joints[2].angle,
            self.joints[2].speed/SPEED_HIP,

            # Right Knee
            self.joints[3].angle,
            self.joints[3].speed/SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0, # Right Foot contact
            ]
        state += [lidar.fraction for lidar in self.lidars]  # Lidar readings
        return state
    
    def _calculate_reward(self, state, action):
        if self.mode == "Walking":
            reward = self._calculate_reward_walking(state)
        else:
            reward = self._calculate_reward_standing(state)       

        reward -= self._calculate_energy_penalty(action)
        
        if self.mode == "Standing" and abs(self.velocity.x)>0.2:
            reward -= 5.0
        if self.max_steps is not None and self.current_step >= self.max_steps:
            reward += 5.0
        if self._check_fallen() or self.position[0]<0:
            reward = -100.0

        return reward
    
    def _calculate_reward_walking(self, state):
        # moving forward is a way to receive reward
        shaping = (100 * self.position[0] / SCALE)

        # Keep head straight, other than that and falling, any behavior is unpunished
        shaping -= 5.0 * abs(state[0])  

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        return reward
    
    def _calculate_reward_standing(self, state):
        shaping = 5.0 * abs(state[0])
        
        reward = 0
        if self.prev_shaping is not None:
            reward = self.prev_shaping - shaping
        self.prev_shaping = shaping

        # Survival bonus: +0.1 for staying alive
        reward += 0.1

        return reward
    
    def _calculate_energy_penalty(self, action):
        return 0.00035 * MOTORS_TORQUE * np.sum(np.clip(np.abs(action), 0, 1))


    def _check_done_conditions(self, state):
        terminated, truncated = False, False

        if self.max_steps is not None:
            if self.current_step >= self.max_steps:
                truncated = True
        if self.position[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            terminated = True
        if self._check_fallen():
            terminated = True

        return terminated, truncated
    
    def _check_fallen(self):
        return self.game_over

    def _reset_episode_variables(self):
        # Box2D world setup
        self.world = Box2D.b2World(gravity=(0, -9.8), doSleep=True)
        self.hull = None

        # Object Creations
        self.background = Background(self.world, self.np_random)
        self.robot = Robot(self.world, self.np_random)
        self.renderer = Renderer()

        self.world.contactListener = ContactDetector(self)
        
        self.scroll = 0.0
        self.current_step = 0
        self.lidar_render = 0
        self.prev_shaping = None
        self.game_over = False

        self.lidars = [LidarCallback() for _ in range(10)]