import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import random
import numpy as np
from environment.env import Point, Direction, block_size, Direction
from collections import deque
from .agent_constants import *
from learning.dqn import DQNNetwork, DQNTrain

class Agent:
    def __init__(self):
        self.epsilon = 0
        self.gamma = 0.9
        self.n_games = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DQNNetwork(input_lay=11, hidden_lay=256, output_lay=3)
        self.trainer = DQNTrain(model=self.model, alpha=ALPHA, gamma=self.gamma)
        

    def state(self, env):
        head = env.snake[0]
        pl = Point(head.x - block_size, head.y)
        pr = Point(head.x + block_size, head.y)
        pu = Point(head.x, head.y - block_size)
        pd = Point(head.x, head.y + block_size)

        #directions
        dl = (env.direction == Direction.LEFT)
        dr = (env.direction == Direction.RIGHT)
        du = (env.direction == Direction.UP)
        dd = (env.direction == Direction.DOWN)

        #dangers
        dgs = (dr and env.game_over(pr)) or (dl and env.game_over(pl)) or (du and env.game_over(pu)) or (dd and env.game_over(pd))
        dgr = (du and env.game_over(pr)) or (dd and env.game_over(pl)) or (dl and env.game_over(pu)) or (dr and env.game_over(pd))
        dgl = (dd and env.game_over(pr)) or (du and env.game_over(pl)) or (dr and env.game_over(pu)) or (dl and env.game_over(pd))

        #food
        fl = env.food.x < env.head.x
        fr = env.food.x > env.head.x
        fu = env.food.y < env.head.y
        fd = env.food.y > env.head.y

        state = [dgs, dgr, dgl,
                 dl,dr,du,dd,
                 fl, fr, fu, fd]
        
        return np.array(state, dtype=int)

    def experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        states, actions, rewards, next_states, doneS = zip(*sample)
        self.trainer.train(states, actions, rewards, next_states, doneS)
        
    
    def train_short(self, state, action, reward, next_state, done):
        self.trainer.train(state, action, reward, next_state, done)

    def action(self, state):
        self.epsilon = 80 - self.n_games
        action = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            act = random.randint(0,2)
            action[act] = 1
        else:
            old_state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(old_state)
            act = torch.argmax(prediction).item()
            action[act] = 1

        return action