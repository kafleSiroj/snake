import os
import sys
sys.stdout = open(os.devnull, 'w')
import pygame
sys.stdout = sys.__stdout__

import random
from collections import namedtuple
from .direction import Direction
from .colors import Colors

pygame.init()
font = pygame.font.Font("font/arial.ttf", 27)

Point = namedtuple('Point' ,['x', 'y'])

block_size = 20

class Game:
    def __init__(self, width=640, height=480, speed=None):
        self.w = width
        self.h = height
        self.speed = speed

        #display game window
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()

        #init direction to right    
        self.direction = Direction.RIGHT

        #set up the snake    
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head, Point(self.head.x - block_size, self.head.y), Point(self.head.x - (2 * block_size), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        x = random.randint(0, (self.w - block_size) // block_size) * block_size
        y = random.randint(0, (self.h - block_size) // block_size) * block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN

        self._move(self.direction)
        self.snake.insert(0, self.head)

        game_over = False
        if self._game_over():
            game_over = True
            return game_over, self.score

        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        self._updadeOnGame()
        self.clock.tick(self.speed)

        return game_over, self.score

    def _game_over(self):
        if self.head.x > self.w - block_size or self.head.x < 0 or self.head.y > self.h - block_size or self.head.y < 0:
            return True

        if self.head in self.snake[1:]:
            return True

        return False

    def _updadeOnGame(self):
        self.display.fill(Colors.BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, Colors.BLUE, pygame.Rect(pt.x, pt.y, block_size, block_size))
            pygame.draw.rect(self.display, Colors.LIGHTBLUE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, Colors.RED, pygame.Rect(self.food.x, self.food.y, block_size, block_size))

        txt = font.render(f"Score: {self.score}", True, Colors.WHITE)
        self.display.blit(txt, [0, 0])
        pygame.display.flip()

    def replay(self):
        while True:
            self.display.fill(Colors.BLACK)

            game_over_txt = font.render("Game Over!", True, Colors.WHITE)
            self.display.blit(game_over_txt, [238, 125])
            replay_txt = font.render("Press [ENTER] to play again, Press [Q] to quit", True, Colors.WHITE)
            self.display.blit(replay_txt, [50, 225])
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        quit()
                    
                    elif event.key in (pygame.K_KP_ENTER, pygame.K_RETURN):
                        self.reset()
                        return

    def _move(self, direction):
        x = self.head.x
        y = self.head.y

        if direction == Direction.RIGHT:
            x += block_size
        elif direction == Direction.LEFT:
            x -= block_size
        elif direction == Direction.DOWN:
            y += block_size
        elif direction == Direction.UP:
            y -= block_size

        self.head = Point(x, y)

    def reset(self):
        #reset direction to right
        self.direction = Direction.RIGHT

        #reset the snake
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head, Point(self.head.x - block_size, self.head.y), Point(self.head.x - (2 * block_size), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()