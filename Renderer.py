from data import *
import pygame
from Box2D.b2 import circleShape
from pygame import gfxdraw

class Renderer:
    def __init__(self):
        self.screen = None

    def draw_bodies(self, bodies, scroll):
        for obj in bodies:
            for fixture in obj.fixtures:
                transformation = fixture.body.transform

                if type(fixture.shape) is circleShape:
                    radius_point = fixture.shape.radius * SCALE
                    center_position = (transformation * fixture.shape.pos) * SCALE
                    center_position = (center_position[0], self._flip_y(center_position[1])) # flip Y for Pygame
                    
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
                    path = []
                    for vertice in fixture.shape.vertices:
                        world_point = transformation * vertice
                        screen_x = (world_point[0] - scroll) * SCALE
                        screen_y = self._flip_y(world_point[1] * SCALE)
                        path.append((screen_x, screen_y)) # flip Y for Pygame
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
                        
    def render_lidars(self, lidars, i, scroll):
        if i < 2 * len(lidars):
            single_lidar = (
                lidars[i]
                if i < len(lidars)
                else lidars[len(lidars) - i - 1]
            )
            if hasattr(single_lidar, "p1") and hasattr(single_lidar, "p2"):
                pygame.draw.line(
                    self.screen,
                    color=(255, 0, 0),
                    start_pos=((single_lidar.p1[0]-scroll) * SCALE, self._flip_y(single_lidar.p1[1] * SCALE)),
                    end_pos=((single_lidar.p2[0]-scroll) * SCALE, self._flip_y(single_lidar.p2[1] * SCALE)),
                    width=1,
                )
    
    def draw_terrain(self, terrain_polygons, scroll):
        for polygon, color in terrain_polygons:
            if polygon[1][0] < scroll:
                continue
            if polygon[0][0] > scroll + SCREEN_WIDTH/SCALE:
                continue

            scaled_poly = []
            for x, y in polygon:
                px = (x-scroll) * SCALE
                py = SCREEN_HEIGHT - y * SCALE  # flip Y for Pygame
                scaled_poly.append((px, py))
            pygame.draw.polygon(self.screen, color, scaled_poly)
            gfxdraw.aapolygon(self.screen, scaled_poly, color)

    def draw_clouds(self, cloud_positions, cloud_img):
        if not hasattr(self, "scaled_cloud"):
            self.scaled_cloud = pygame.transform.scale(cloud_img, (128, 80))

        for cloud_x, cloud_y in cloud_positions:
            screen_x = cloud_x * SCALE
            screen_y = SCREEN_HEIGHT - cloud_y * SCALE # flip Y for Pygame
            self.screen.blit(self.scaled_cloud,(screen_x, screen_y))

    def _flip_y(self, y):
        """Flip the Y coordinate for Pygame rendering."""
        return SCREEN_HEIGHT - y
    
    # Setters #
    def set_screen(self, screen):
        self.screen = screen