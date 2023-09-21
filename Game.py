import gym
import math
import numpy as np
import pygame
from typing import Tuple, List
from Car import Car
from Checkpoint import Checkpoint
from Sensor import Sensor

class Game(gym.Env):
    def __init__(self, screen_width:int = 1200, screen_height:int = 800) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height

    def init(self) -> None:
        self.car = Car()
        self.car.set_position((110, 500))

        self.sensor_info = []
        self.sensors = [
            Sensor(500, pygame.math.Vector2(1, 0)),
            Sensor(500, pygame.math.Vector2(math.cos(-math.pi/7), math.sin(-math.pi/7))),
            Sensor(500, pygame.math.Vector2(math.cos(math.pi/7), math.sin(math.pi/7))),
            Sensor(500, pygame.math.Vector2(math.cos(-math.pi/3), math.sin(-math.pi/3))),
            Sensor(500, pygame.math.Vector2(math.cos(math.pi/3), math.sin(math.pi/3))),
            Sensor(500, pygame.math.Vector2(math.cos(-math.pi/2), math.sin(-math.pi/2))),
            Sensor(500, pygame.math.Vector2(math.cos(math.pi/2), math.sin(math.pi/2))),
            Sensor(500, pygame.math.Vector2(math.cos(-3*math.pi/4), math.sin(-3*math.pi/4))),
            Sensor(500, pygame.math.Vector2(math.cos(3*math.pi/4), math.sin(3*math.pi/4))),
            Sensor(500, pygame.math.Vector2(-1, 0)),
        ]

        self.edge_distances = {i: 500 for i in range(len(self.sensors))}

        self.checkpoint_id = np.array([0, 0])
        self.checkpoints = [
            Checkpoint(40, 450, 170, 450),
            Checkpoint(60, 370, 195, 370),
            Checkpoint(100, 280, 230, 290),
            Checkpoint(150, 190, 270, 220),
            Checkpoint(230, 105, 310, 170),
            Checkpoint(330, 70, 390, 140),
            Checkpoint(460, 45, 470, 125),
            Checkpoint(595, 30, 600, 115),
            Checkpoint(750, 35, 740, 115),
            Checkpoint(910, 55, 870, 130),
            Checkpoint(1050, 150, 930, 150),
            Checkpoint(870, 200, 965, 255),
            Checkpoint(790, 245, 875, 300),
            Checkpoint(700, 300, 795, 350),
            Checkpoint(700, 430, 800, 385),            
            Checkpoint(810, 480, 900, 420),
            Checkpoint(910, 515, 1000, 455),
            Checkpoint(990, 550, 1090, 505),
            Checkpoint(980, 600, 1090, 635),
            Checkpoint(910, 650, 990, 710),
            Checkpoint(815, 700, 860, 775),
            Checkpoint(670, 680, 700, 780),
            Checkpoint(540, 650, 540, 730),
            Checkpoint(400, 650, 385, 730),
            Checkpoint(270, 660, 230, 735),
            Checkpoint(100, 680, 180, 610),
            Checkpoint(40, 570, 155, 540)
        ]
    
    def init_render(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Car Game")
        self.clock = pygame.time.Clock()

        self.background_image = pygame.image.load("images/map.png")
        self.background_image = pygame.transform.scale(self.background_image, (self.screen_width, self.screen_height))

    def get_action(self, keys: Tuple[int, ...]) -> np.ndarray:
        actions = {
            pygame.K_LEFT:  np.array([0, 1]),   # Rotate left
            pygame.K_RIGHT: np.array([0, -1]),  # Rotate right
            pygame.K_UP:    np.array([1, 0]),   # Increase speed
            pygame.K_DOWN:  np.array([-1, 0])   # Decrease speed
        }

        action = np.array([0, 0])
        for key in actions:
            if keys[key]:
                action += actions[key]

        return action

    def step(self, action:Tuple[int, int]) -> List:
        reward = -0.2 if self.car.speed <= 0 else 0.1
        done = False

        self.car.move(action)
        self.car.keep_inbounds(self.screen_width, self.screen_height)
        
        self.sensor_info = []
        for i, sensor in enumerate(self.sensors):
            sensor.move(self.car.rotation, self.car.rect)
            self.edge_distances[i], sensor_inf  = sensor.collisions(self.background_image, self.car.rect)
            self.sensor_info.append(sensor_inf)

        for collision_distance in self.edge_distances.values():
            if collision_distance is not None and collision_distance < min(self.car.rect.width // 2, self.car.rect.height // 2):
                # self.reset()
                reward = -20
                done = True

        
        # cleared = False
        if self.checkpoint_id[0] == len(self.checkpoints):
            # cleared = True
            # self.checkpoint_id += 1
            self.reset_checkpoints()
        else:
            if self.checkpoints[self.checkpoint_id[0]].active == 1 and self.checkpoints[self.checkpoint_id[0]].intersect(self.car.rect_rotated):
                self.checkpoints[self.checkpoint_id[0]].active = -1
                self.checkpoint_id += 1
                reward += 10
        # if cleared:
        #     self.reset_checkpoints()
        #     self.checkpoint_id = 0
            # reward += 100
            # done = True

        x, y = self.car.get_position()
        observation = [x/self.screen_width, y/self.screen_height, self.car.speed/self.car.max_speed, self.car.rotation/360.0]
        for val in self.edge_distances.values():
            observation.append(val/500)

        return [np.array(observation), reward, done, None, None]

    def render(self) -> None:
        self.screen.blit(self.background_image, (0, 0))
        self.car.draw(self.screen)

        for sensor in self.sensors:
            sensor.draw(self.screen, self.car.rect)

        for sensor_inf in self.sensor_info:
            if sensor_inf:
                sensor.draw_point(self.screen, sensor_inf[1], sensor_inf[0], sensor_inf[2])
    
        for i, checkpoint in enumerate(self.checkpoints):
            if checkpoint.active == 1:
                checkpoint.color = (0, 0, 255)
            else:
                checkpoint.color = (0, 255, 0)
            checkpoint.draw(self.screen)

        pygame.display.update()

    def start(self) -> None:
        while True:
            self.check_quit()
            keys = pygame.key.get_pressed()
            action = self.get_action(keys)
            self.step(action)
            self.render()
                            
    def check_quit(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def reset_checkpoints(self, id:bool = False) -> None:
        self.checkpoint_id[0] = 0
        if id:
            self.checkpoint_id[1] = 0
        for checkpoint in self.checkpoints:
            checkpoint.reset()

    def reset_snesors(self) -> None:
        self.sensor_info = []
        for sensor in self.sensors:
            sensor.reset()
    
    def reset_car(self) -> None:
        self.car.set_position((110, 500))
        self.car.speed = 0
        self.car.rotation = 90
    
    def reset_edge_distances(self) -> None:
        for key in self.edge_distances:
            self.edge_distances[key] = 500

    def reset(self) -> List:
        self.reset_car()
        self.reset_snesors()
        self.reset_checkpoints(id = True)
        self.reset_edge_distances()
        
        x, y = self.car.get_position()
        observation = [x/self.screen_width, y/self.screen_height, self.car.speed/self.car.max_speed, self.car.rotation/360.0]
        for val in self.edge_distances.values():
            observation.append(val/500)

        return np.array(observation)
    
if __name__ == "__main__":
    screen_width = 1200
    screen_height = 800

    game = Game(screen_width, screen_height)
    game.init_render()
    game.clock.tick(60)
    game.init()
    game.start()
