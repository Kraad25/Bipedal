import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle
import Box2D
from Box2D.b2 import (
    circleShape,
    edgeShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
)

import pygame
from pygame import gfxdraw

from data import *
from ContactDetector import ContactDetector


class BipedalWalkerEnv(gym.Env):
    # Metadata -> Contains information about rendering options.
    metadata = {"render_modes": [RENDER_MODE], "render_fps": FPS}

    def __init__(self, render_mode="human", training_mode="Standing", MaxSteps=None):
        EzPickle.__init__(self, render_mode)

        # Box2D world setup
        self.world = Box2D.b2World(gravity=(0, -9.8), doSleep=True)
        self.hull = None
        self.max_steps = MaxSteps if MaxSteps is not None else None

        # World parameters
        self.terrain = [] # Stores Box2D static bodies for terrain (used for collisions)
        self.cloud_positions = []  # List to store cloud positions
        
        # Rendering parameters
        self.render_mode = render_mode
        self.fps = self.metadata["render_fps"]
        self.clock = None
        self.screen = None

        # Fixture definitions
        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
            friction=FRICTION,
        )
        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0), (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
        )

        # Action and observation spaces
        self.training_mode = training_mode
        self.current_step = 0
        self.prev_shaping = None

        if self.training_mode == "Standing":
            self.action_space = spaces.Box(STANDING_ACTION_LOW, STANDING_ACTION_HIGH)
            self.observation_space = spaces.Box(STANDING_OBSERVATION_LOW, STANDING_OBSERVATION_HIGH)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)        
        self._reset_episode_variables()
        self._generate_background()

        initial_x, initial_y = self._get_robot_intial_position()
        self.legs = []
        self.joints = []

        self.hull = self._create_hull(initial_x, initial_y)
        self._create_legs(initial_x, initial_y)
        #self._apply_initial_random_force_to_hull()

        self.drawlist = self.terrain + self.legs + [self.hull]

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

        self._draw_background()
        self._draw_robot()

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
            self.cloud_img = pygame.image.load("assets/Cloud.png").convert_alpha()

    
    def _generate_terrain(self):
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
        sky_top = SCREEN_HEIGHT / SCALE 
        sky_bottom = sky_top * 0.90  
        for _ in range(5):
            x = self.np_random.uniform(0, SCREEN_WIDTH/SCALE)
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
        self.cloud = pygame.transform.scale(self.cloud_img, (128, 80))
        for cloud_x, cloud_y in self.cloud_positions:
            screen_x = cloud_x * SCALE
            screen_y = SCREEN_HEIGHT - cloud_y * SCALE
            self.screen.blit(self.cloud,(screen_x, screen_y))

    def _draw_robot(self):
        for obj in self.drawlist:
            for fixture in obj.fixtures:
                transformation = fixture.body.transform

                if type(fixture.shape) is circleShape:
                    radius_point = fixture.shape.radius * SCALE
                    center_position = (transformation * fixture.shape.pos) * SCALE
                    center_position = (center_position[0], SCREEN_HEIGHT - center_position[1]) # flip Y for Pygame
                    
                    pygame.draw.circle(self.screen, 
                                       color=obj.color1, 
                                       center=center_position, 
                                       radius=radius_point
                                       )
                    
                    pygame.draw.circle(self.screen,
                                       color=obj.color2, 
                                       center=center_position, 
                                       radius=radius_point
                                       )
                else:
                    path = [transformation * vertice * SCALE for vertice in fixture.shape.vertices]
                    path = [(p[0], SCREEN_HEIGHT - p[1]) for p in path] # flip Y for Pygame
                    if len(path) > 2:
                        pygame.draw.polygon(self.screen, color=obj.color1, points=path)
                        gfxdraw.aapolygon(self.screen, path, obj.color1)
                        path.append(path[0])  # Close the polygon

                        pygame.draw.polygon(self.screen, color=obj.color2, points=path, width=1)
                        gfxdraw.aapolygon(self.screen, path, obj.color2)
                    else:
                        pygame.draw.aaline(self.screen,
                                           start_pos=path[0],
                                           end_pos=path[1],
                                           color=obj.color1,
                                           )

    
    def _get_robot_intial_position(self):
        starting_area = TERRAIN_STARTPAD * TERRAIN_STEP
        init_x = starting_area / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H

        return init_x, init_y
    
    def _create_hull(self,x ,y):
        hull = self.world.CreateDynamicBody(position=(x, y), fixtures=HULL_FD)
        hull.color1 = (127, 51, 229)
        hull.color2 = (76, 76, 127)
        return hull
    
    def _apply_initial_random_force_to_hull(self):
        random_force_in_x = self.np_random.uniform(-INITIAL_RANDOM_FORCE, INITIAL_RANDOM_FORCE)
        self.hull.ApplyForceToCenter((random_force_in_x, 0), True)

    def _create_legs(self, initial_x, initial_y):
        for i in [-1, 1]:
            self._create_leg(i, initial_x, initial_y)

    def _create_leg(self, side, initial_x, initial_y):
        upper_leg  = self.world.CreateDynamicBody(
            position=(initial_x, initial_y - LEG_H / 2 - LEG_DOWN),
            angle=(side*0.05),
            fixtures=LEG_FD,
        )
        upper_leg .color1 = (153 - side * 25, 76 - side * 25, 127 - side * 25)
        upper_leg .color2 = (102 - side * 25, 51 - side * 25, 76 - side * 25)
     
        hip_joint = revoluteJointDef(
            bodyA=self.hull,
            bodyB=upper_leg,
            localAnchorA=(0, LEG_DOWN),
            localAnchorB=(0, LEG_H / 2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=MOTORS_TORQUE,
            motorSpeed=side,  # ±1 to mirror the direction
            lowerAngle=-0.3927,
            upperAngle=1.1,
        )

        upper_leg.ground_contact = False

        self.legs.append(upper_leg)
        self.joints.append(self.world.CreateJoint(hip_joint))

        lower_leg = self.world.CreateDynamicBody(
            position=(initial_x, initial_y - LEG_H * 3 / 2 - LEG_DOWN),
            angle=(side*0.05),
            fixtures=LOWER_FD,
        )
        lower_leg.color1 = (153 - side * 25, 76 - side * 25, 127 - side * 25)
        lower_leg.color2 = (102 - side * 25, 51 - side * 25, 76 - side * 25)

        knee_joint = revoluteJointDef(
            bodyA=upper_leg,
            bodyB=lower_leg,
            localAnchorA=(0, -LEG_H / 2),
            localAnchorB=(0, LEG_H / 2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=MOTORS_TORQUE,
            motorSpeed=side,  # ±1 to mirror the direction
            lowerAngle=-0.785,
            upperAngle=-0.1,
        )

        lower_leg.ground_contact = False

        self.legs.append(lower_leg)
        self.joints.append(self.world.CreateJoint(knee_joint))

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
        position = self.hull.position
        self.scroll = position.x - (SCREEN_WIDTH / SCALE) / 5 # Offset for scrolling

    def _get_state(self):
        velocity = self.hull.linearVelocity

        state = [
            # Hull
            self.hull.angle,
            2.0* self.hull.angularVelocity/self.fps, # Normalized angular velocity(rad/frame)
            0.3*velocity.x*(SCREEN_WIDTH / SCALE) / self.fps, # Normalized linear velocity in X
            0.3*velocity.y*(SCREEN_HEIGHT / SCALE) / self.fps, # Normalized linear velocity in Y

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
        return state
    
    def _check_fallen(self, state):
        return self.game_over
    
    def _calculate_reward(self, state, action):
        shaping = 5.0 * abs(state[0])
        
        reward = 0.0
        if self.prev_shaping is not None:
            reward = self.prev_shaping - shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.001 * np.clip(np.abs(a), 0, 1)
        
        if self.current_step >= MAX_STEPS:
            reward += 10.0
        if self._check_fallen(state):
            reward -= 10.0

        return reward
    
    def _check_done_conditions(self, state):
        terminated = False
        truncated = False

        if self.max_steps is not None:
            if self.current_step >= self.max_steps:
                truncated = True
        
        if self._check_fallen(state):
            terminated = True

        return terminated, truncated
    
    def _reset_episode_variables(self):
        self.scroll = 0.0
        self.current_step = 0 
        self.prev_shaping = None
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False

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

    while running:
        action = env.action_space.sample()  # Random action for testing
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if terminated or truncated:
            print("Episode finished.")
            running = False


    env.close()