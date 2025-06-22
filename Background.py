from data import *
import numpy as np

class Background():
    def __init__(self, world, np_random):
        self.world = world
        self.np_random = np_random
        self.fd_edge = TERRAIN_EDGE_FD
        
        self.terrain_bodies = [] # Stores Box2D static bodies for terrain, for physics
        self.terrain_polygons = [] # Stores Shapes and Colors for Visuals
        self.clouds_positions  = [] # List to store cloud positions

    def generate(self):
        self._generate_terrain()
        self._generate_clouds()

    # Getters #
    def get_terrain_bodies(self):
        return self.terrain_bodies
    
    def get_terrain_polygons(self):
        return self.terrain_polygons
    
    def get_cloud_positions(self):
        return self.clouds_positions 
    
    def get_drawables(self):
        return self.terrain_bodies

    # Private Methods #
    def _generate_terrain(self):
        terrain_x = []
        terrain_y = []

        y = TERRAIN_HEIGHT
        delta_y = 0 # Rate of change in height

        for i in range(TERRAIN_LENGTH):
            x = i * TERRAIN_STEP
            terrain_x.append(x)    

            if i < TERRAIN_STARTPAD:
                delta_y = 0
            else:
                delta_y = 0.8*delta_y + 0.01*np.sign(TERRAIN_HEIGHT - y)
                delta_y += self.np_random.uniform(-1, 1)/SCALE
            
            y += delta_y
            terrain_y.append(y)

        for i in range(TERRAIN_LENGTH-1):
            x1, y1 = terrain_x[i], terrain_y[i]
            x2, y2 = terrain_x[i+1], terrain_y[i+1]
            
            poly = [(x1, y1), (x2, y2)]

            self.fd_edge.shape.vertices = poly
            terrain = self.world.CreateStaticBody(fixtures = self.fd_edge)

            # Box2D Rendering Colors
            color = (76, 255 if i % 2 == 0 else 204, 76)
            terrain.color1 = color
            terrain.color2 = color

            self.terrain_bodies.append(terrain)

            color = (102, 153, 76) 
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_polygons.append((poly, color))     
        self.terrain_bodies.reverse() # Reverse the terrain to match Box2D's coordinate system

    def _generate_clouds(self):
        sky_top = SCREEN_HEIGHT / SCALE 
        sky_bottom = sky_top * 0.90  

        for _ in range(5):
            x = self.np_random.uniform(0, SCREEN_WIDTH/SCALE)
            y = self.np_random.uniform(sky_bottom, sky_top)
            self.clouds_positions.append((x, y))