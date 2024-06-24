import numpy as np
from scipy.interpolate import interp1d


class LinearSegmentNoise1D:
    """1 Dimensional LinearSegmentNoise

    Generates noise values by interpolating between randomly sampled points
    along the orbit.
    """
    def __init__(
            self,
            steps=200,
            sigma=0.2,
            num_interp_points=10,
            dt=1e-2):
        self.sigma = sigma
        self.steps = steps
        self.dt = dt
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
        self.points_y = self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=(len(self.points_x)))
        self.step_ind = 0
        self.f = interp1d(
            self.points_x,
            self.points_y,
            kind='linear',
            fill_value="extrapolate")

    def __call__(self):
        self.step_ind += 1
        return np.array([self.f(self.orb[self.step_ind])])

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
            sigma=0.2,
            num_interp_points=10,
            dt=1e-2):
        self.dim = dim
        self.generator = [
            LinearSegmentNoise1D(steps, sigma, num_interp_points, dt)
            for _ in range(self.dim)]

    def __call__(self):
        return np.concatenate([g() for g in self.generator])

    def reset(self):
        return [g.reset() for g in self.generator]


class SmoothNoise1D:
    """1 Dimensional SmoothSegmentNoise

    Generates noise values by interpolating between randomly sampled points
    along the orbit.
    """
    def __init__(
            self,
            steps=200,
            sigma=0.2,
            num_interp_points=10,
            dt=1e-2):
        self.sigma = sigma
        self.steps = steps
        self.dt = dt
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
        self.points_y = self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=(len(self.points_x)))
        self.step_ind = 0
        self.f = interp1d(
            self.points_x,
            self.points_y,
            kind='cubic',
            fill_value="extrapolate")

    def __call__(self):
        self.step_ind += 1
        return np.array([self.f(self.orb[self.step_ind])])

    def reset(self):
        self.setup()


class SmoothNoiseND:
    """N Dimensional SmoothSegmentNoise

    Generates multi Dimensional noise values by interpolating between randomly
    sampled points along the orbit.
    """
    def __init__(
            self,
            dim=2,
            steps=200,
            sigma=0.2,
            num_interp_points=10,
            dt=1e-2):
        self.dim = dim
        self.generator = [
            SmoothNoise1D(steps, sigma, num_interp_points, dt)
            for _ in range(self.dim)]

    def __call__(self):
        return np.concatenate([g() for g in self.generator])

    def reset(self):
        return [g.reset() for g in self.generator]