import numpy as np


class EnvNoise:
    def __init__(
            self,
            env,
            repeat=1
        ):
        self.repeat = repeat
        self.env = env
        self.current_action = None
        self.reset()

    def __call__(self):
        if self.count % self.repeat == 0:
            self.reset()
        self.count += 1
        return self.current_action

    def reset(self):
        self.count = 0
        self.current_action = self.env.action_space.sample()
        return self.current_action


class NoNoise:
    def __init__(self, dim):
        self.dim = dim
        self.reset()

    def __call__(self):
        return np.zeros(self.dim)

    def reset(self):
        return


class NormalNoise:
    def __init__(
            self,
            dim,
            sigma=0.2,
            dt=1e-2,
            repeat=2):
        self.dim = dim
        self.sigma = sigma
        self.dt = dt
        self.repeat = repeat
        self.reset()
        self.count=0
        self.current_action = None

    def __call__(self):
        if self.count % self.repeat == 0:
            self.reset()
        self.count += 1
        return self.current_action

    def reset(self):
        self.count = 0
        self.current_action = self.sigma * np.sqrt(self.dt) * \
            np.random.normal(loc=0, scale=1, size=(self.dim,))
        return self.current_action
