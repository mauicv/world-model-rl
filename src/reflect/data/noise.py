import numpy as np
from scipy.interpolate import interp1d


class EnvNoise:
    def __init__(self, env):
        self.env = env
        self.reset()

    def __call__(self):
        return self.env.action_space.sample()

    def reset(self):
        return


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


class OUNoise:
    """Ornstein-Uhlenbeck process.

    Taken from https://keras.io/examples/rl/ddpg_pendulum/
    Formula from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
    """
    def __init__(
            self,
            dim=1,
            sigma=0.15,
            theta=0.2,
            dt=1e-2,
            x_initial=None):
        self.theta = theta
        self.dim = dim
        self.sigma = sigma/10
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (- self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt)
            * np.random.normal(loc=0, scale=1, size=(self.dim,))
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros(self.dim)


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