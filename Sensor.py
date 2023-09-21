import math
import pygame
from typing import Tuple, Type

class Sensor:
    def __init__(self, distance:int = 200, direction:pygame.math.Vector2 = pygame.math.Vector2(1, 0)) -> None:
        self.sensor_distance = distance
        self.sensor_direction = direction
        self.sensor_endpoint = None

    def reset(self) -> None:
        pass

    def move(self, rotation:float, object:Type) -> None:
        self.sensor_direction_rotated = self.sensor_direction.rotate(-rotation)
        self.sensor_endpoint = object.center + self.sensor_direction_rotated.normalize() * self.sensor_distance

    def draw(self, screen:pygame.display, object:Type, color:Tuple[int, int, int] = (255, 255, 255)) -> None:
        pygame.draw.line(screen, color, object.center, self.sensor_endpoint, 2)

    def draw_point(self, screen:pygame.display, sensor_point:Tuple[int, int], color:Tuple[int, int, int] = (255, 255, 255), size:float = 7):
        pygame.draw.circle(screen, color, (int(sensor_point[0]), int(sensor_point[1])), size)

    def collisions(self, background_image: pygame.Surface, obj: Type) -> Tuple:
        background_rect = background_image.get_rect()
        for i in range(self.sensor_distance):
            sensor_point = obj.center + self.sensor_direction_rotated * i
            pixel_color = background_rect.collidepoint(sensor_point) and background_image.get_at((int(sensor_point[0]), int(sensor_point[1])))
            if pixel_color == (0, 0, 0):
                dx = obj.center[0] - sensor_point[0]
                dy = obj.center[1] - sensor_point[1]
                collision_distance = round(math.sqrt(dx**2 + dy**2))
                if collision_distance < min(obj.width // 2, obj.height // 2):
                    return -1, ((255, 0, 0), (int(sensor_point[0]), int(sensor_point[1])), 10)
                return collision_distance, ((255, 255, 255), (int(sensor_point[0]), int(sensor_point[1])), 7)
        return self.sensor_distance, None




