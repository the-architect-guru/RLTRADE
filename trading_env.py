import gymnasium as gym
from gymnasium import spaces
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=1000):
        super(TradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(data.columns) + 1,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.done = False
        return self._next_observation()

    def _next_observation(self):
        obs = np.append(self.data.iloc[self.current_step].values, [self.balance])
        return obs

    def step(self, action):
        self.current_step += 1
        reward = 0

        if action == 1:  # Buy
            self.position += 1
            self.balance -= self.data.iloc[self.current_step]['Close']
        elif action == 2:  # Sell
            self.position -= 1
            self.balance += self.data.iloc[self.current_step]['Close']

        if self.current_step >= len(self.data) - 1:
            self.done = True

        reward = self.balance + self.position * self.data.iloc[self.current_step]['Close']
        obs = self._next_observation()
        return obs, reward, self.done, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}')