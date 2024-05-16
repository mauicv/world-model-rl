import torch
import matplotlib.pyplot as plt
from gymnasium.spaces import Box
import numpy as np


class TestEnvironment:
    def __init__(self) -> None:
        self.actor_loc = None
        self.screen = None
        self.action_space = Box(
            low=-1,
            high=1,
            shape=(2, ),
            dtype=float
        )

    def reset(self):
        self.actor_loc = np.random.randint(2, 60, (2,), dtype=np.int16)
        return self.actor_loc, {} 

    def step(self, action):
        assert np.all(action > -1) and np.all(action < 1), \
            f'all {action} should be between -1 and 1'
        action = 5 * action
        self.actor_loc = (self.actor_loc + action).astype(np.int16)
        reward = self.actor_loc[0]
        return self.actor_loc, reward, False

    def render(self):
        screen = np.zeros((64, 64, 3), dtype=np.float32)
        screen[
            self.actor_loc[0]-2:self.actor_loc[0]+2,
            self.actor_loc[1]-2:self.actor_loc[1]+2,
        ] = np.array([256, 256, 256])
        return screen
