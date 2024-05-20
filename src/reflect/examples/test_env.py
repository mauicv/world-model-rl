import torch
import matplotlib.pyplot as plt
from gymnasium.spaces import Box
import numpy as np


class SimpleEnvironment:
    def __init__(self, size=64) -> None:
        self.actor_loc = None
        self.screen = None
        self.size = size
        self.action_space = Box(
            low=-1,
            high=1,
            shape=(2, ),
            dtype=float
        )
        self.observation_space = Box(
            low=0,
            high=self.size,
            shape=(2, ),
            dtype=float
        )

    def reset(self):
        self.actor_loc = np.random.randint(2, self.size - 2, (2,), dtype=np.int16)
        return self.actor_loc, {} 

    def step(self, action):
        assert np.all(action > -1) and np.all(action < 1), \
            f'all {action} should be between -1 and 1'
        action = 5 * action
        self.actor_loc = (self.actor_loc + action).astype(np.int16)
        self.actor_loc = np.clip(self.actor_loc, 2, self.size - 2)
        reward = self.actor_loc[0]
        return self.actor_loc, reward, False

    def render(self):
        screen = np.ones((self.size, self.size, 3), dtype=np.float32)
        screen[
            self.actor_loc[0]-2:self.actor_loc[0]+2,
            self.actor_loc[1]-2:self.actor_loc[1]+2,
        ] = np.array([0, 0, 0])
        return screen
