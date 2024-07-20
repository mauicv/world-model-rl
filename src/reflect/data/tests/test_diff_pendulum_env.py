import torch
from reflect.data.differentiable_pendulum import DiffPendulumEnv


def test_diff_pendulum_env():
    env = DiffPendulumEnv()
    s, _ = env.reset(batch_size=10)
    assert s.shape == (10, 3)
    for _ in range(100):
        action = torch.randn(10)
        state, reward, done, _ = env.step(action)
        assert state.shape == (10, 3)
        assert reward.shape == (10, 1)
        assert done.shape == (10, 1)