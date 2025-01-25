import os
import sys
sys.stdout = open(os.devnull, 'w')
import pygame
sys.stdout = sys.__stdout__

import random
from .Colors import Colors
import numpy as np
from .direction import Direction
from collections import namedtuple
from .env_constant import *

pygame.init()
font = pygame.font.Font("font/arial.ttf", 27)

Point = namedtuple('Point' ,['x', 'y'])

class GameAgent:
    def __init__(self, width=640, height=480, speed=None):
        self.w = width
        self.h = height
        self.speed = speed

        #display game window
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.reset_func()

    def reset_func(self):
        #reset directoin to right
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, Point(self.head.x - block_size, self.head.y), Point(self.head.x - (2*block_size), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame = 0


    def _place_food(self):
        x = random.randint(0, (self.w - block_size)//block_size)*block_size
        y = random.randint(0, (self.h - block_size)//block_size)*block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play(self, action):
        self.frame += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
 
        self._move(action)
        self.snake.insert(0, self.head)

        game_over = False
        reward = 0
        if self.game_over() or self.frame > len(self.snake)*100:
            game_over = True
            reward = -10
            return game_over, self.score, reward
        
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        self._update()
        self.clock.tick(self.speed)

        return game_over, self.score, reward
    

    def game_over(self, pt=None):
        if pt == None:
            pt = self.head

        if pt.x > self.w - block_size or pt.x < 0 or pt.y > self.h - block_size or pt.y < 0:
            return True
        
        if pt in self.snake[1:]:
            return True
        
        return False
    

    def _update(self):
        self.display.fill(Colors.BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, Colors.BLUE, pygame.Rect(pt.x, pt.y, block_size, block_size))
            pygame.draw.rect(self.display, Colors.LIGHTBLUE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, Colors.RED, pygame.Rect(self.food.x, self.food.y, block_size, block_size))

        txt = font.render(f"Score: {self.score}", True, Colors.WHITE)
        self.display.blit(txt, [0,0])
        pygame.display.flip()
 

    def _move(self, action):
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clockwise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1)%4
            new_dir = clockwise[next_idx]
        else:
            next_idx = (idx - 1)%4
            new_dir = clockwise[idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += block_size
        elif self.direction == Direction.LEFT:
            x -= block_size
        elif self.direction == Direction.DOWN:
            y += block_size
        elif self.direction == Direction.UP:
            y -= block_size
        
        self.head = Point(x, y)