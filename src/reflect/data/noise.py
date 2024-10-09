import numpy as np
from scipy.interpolate import interp1d
import torch


class EnvNoise:
    def __init__(
            self,
            env,
        ):
        self.env = env
        self.current_action = None
        self.reset()

    def __call__(self):
        return self.env.action_space.sample()

    def reset(self):
        pass


class NormalNoise:
    def __init__(
            self,
            dim,
            noise_size=0.3
        ):
        self.noise_size = noise_size
        self.dim = dim
        self.reset()

    def __call__(self):
        return torch.normal(torch.zeros((self.dim, )), self.noise_size)

    def reset(self):
        pass


class NoNoise:
    def __init__(self, dim):
        self.dim = dim
        self.reset()

    def __call__(self):
        return np.zeros((self.dim, ))

    def reset(self):
        return
        

class LinearSegmentNoise1D:
    """1 Dimensional LinearSegmentNoise

    Generates noise values by interpolating between randomly sampled points
    along the orbit.
    """
    def __init__(
            self,
            steps=200,
            noise_size=0.3,
            num_interp_points=10,
            ):
        self.steps = steps
        self.noise_size = noise_size
        self.num_interp_points = num_interp_points
        self.orb = np.linspace(0, steps, num=steps+1, endpoint=True)
        self.points_x = None
        self.points_y = None
        self.step_ind = None
        self.f = None
        self.setup()

    def setup(self):
        self.points_x = np.array(
            [0, *np.random.choice(
                    self.orb[1:-1],
                    size=self.num_interp_points,
                    replace=False), self.steps+1])
        
        self.points_y = self.noise_size * \
            np.random.normal(size=(len(self.points_x)))
        self.step_ind = 0
        self.f = interp1d(
            self.points_x,
            self.points_y,
            kind='linear',
            fill_value="extrapolate")

    def __call__(self):
        self.step_ind += 1
        return self.f(self.orb[self.step_ind]).item()

    def reset(self):
        self.setup()


class LinearSegmentNoiseND:
    """N Dimensional SmoothSegmentNoise

    Generates multi Dimensional noise values by interpolating between randomly
    sampled points along the orbit.
    """
    def __init__(
            self,
            dim=2,
            steps=200,
            noise_size=0.3,
            num_interp_points=10,
            ):
        self.dim = dim
        self.generator = [
            LinearSegmentNoise1D(
                steps=steps,
                noise_size=noise_size,
                num_interp_points=num_interp_points
            ) for _ in range(self.dim)
        ]

    def __call__(self):
        data = list(g() for g in self.generator)
        return torch.tensor(data)

    def reset(self):
        return [g.reset() for g in self.generator]