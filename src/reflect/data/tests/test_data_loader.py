from reflect.data.loader import EnvDataLoader
from torchvision.transforms import Resize, Compose
import gymnasium as gym
import torch
import pytest


@pytest.mark.parametrize("env_name", [
    "InvertedPendulum-v4",
    "Ant-v4",
])
def test_data_loader(env_name):
    num_time_steps = 18
    env = gym.make(env_name, render_mode="rgb_array")
    action_shape, = env.action_space.shape
    data_loader = EnvDataLoader(
        num_time_steps=num_time_steps + 1,
        img_shape=(3, 64, 64),
        transforms=Compose([Resize((64, 64))]),
        env=env
    )
    for _ in range(4):
        data_loader.perform_rollout()
    for i in range(4):
        assert data_loader.end_index[i] >= num_time_steps, f'{data_loader.end_index=}'
        assert torch.all(data_loader.img_buffer[i, data_loader.end_index[i]+1:] == 0)
        assert torch.all(data_loader.action_buffer[i, data_loader.end_index[i]+1:] == 0)
        assert torch.all(data_loader.reward_buffer[i, data_loader.end_index[i]+1:] == 0)

    s, a, r, d = data_loader.sample(batch_size=3, num_time_steps=10)

    assert s.shape == (3, 10, 3, 64, 64)
    assert a.shape == (3, 10, action_shape)
    assert r.shape == (3, 10, 1)
    assert d.shape == (3, 10, 1)
    data_loader.close()
