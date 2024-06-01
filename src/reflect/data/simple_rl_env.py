import torch
import matplotlib.pyplot as plt
from gymnasium.spaces import Box
import numpy as np


class Space():
    def __init__(self, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        if np.random.uniform() < 0.5:
            return np.array([0, 0])
        directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        return directions[np.random.choice(4)]


class SimpleRLEnvironment:
    def __init__(self, size=64, num_threats=1, moving_threats=False) -> None:
        self.actor_loc = None
        self.screen = None
        self.threats = None
        self.target = None
        self.moving_threats = moving_threats
        self.size = size
        self.num_threats = num_threats
        self.action_space = Space(
            low=-1,
            high=1,
            shape=(2, ),
            dtype=float
        )
        self.observation_space = Space(
            low=0,
            high=self.size,
            shape=(2, ),
            dtype=float
        )

    def reset(self):
        self.actor_loc = np.random.randint(0, self.size - 1, (2,), dtype=np.int16)
        self.threats = np.random.randint(0, self.size - 1, (self.num_threats, 2), dtype=np.int16)
        self.target = np.random.randint(0, self.size - 1, (2,), dtype=np.int16)
        return self.actor_loc, {} 

    def compute_reward(self):
        reward = 0
        for threat in self.threats:
            if np.all(threat == self.actor_loc):
                reward -= 100
        if np.all(self.target == self.actor_loc):
            reward += 1
        return reward

    def step(self, action):
        assert np.all(action >= -1) and np.all(action <= 1), \
            f'all {action} should be between -1 and 1'
        self.actor_loc = (self.actor_loc + action).astype(np.int16)
        self.actor_loc = np.clip(self.actor_loc, 0, self.size - 1)
        if self.moving_threats:
            directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
            direction_inds = np.random.choice(4, (self.num_threats,))
            threat_actions = directions[direction_inds]
        else:
            threat_actions = np.zeros((self.num_threats, 2), dtype=np.int16)
        self.threats = (self.threats + threat_actions).astype(np.int16)
        self.threats = np.clip(self.threats, 0, self.size - 1)
        return self.actor_loc, self.compute_reward(), False

    def render(self):
        screen = np.ones((self.size, self.size, 3), dtype=np.float32) * 255
        for threat in self.threats:
            screen[
                threat[0],
                threat[1],
            ] = np.array([255, 0, 0])
        screen[
            self.target[0],
            self.target[1],
        ] = np.array([0, 255, 0])
        screen[
            self.actor_loc[0],
            self.actor_loc[1],
        ] = np.array([0, 0, 0])
        return screen
