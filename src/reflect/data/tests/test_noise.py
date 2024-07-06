from reflect.data.loader import EnvDataLoader
from reflect.models.world_model.observation_model import ObservationalModel
from torchvision.transforms import Resize, Compose
import gymnasium as gym
import torch
import numpy as np
import pytest
from reflect.data.noise import SmoothNoiseND, LinearSegmentNoiseND, NormalNoise

def test_smooth_noise_generators():
    generator = SmoothNoiseND(
        dim=2,
        steps=200,
        sigma=0.2,
        num_interp_points=10,
        dt=1e-2
    )
    noise = generator()
    assert len(noise) == 2


def test_linear_noise_generators():
    generator = LinearSegmentNoiseND(
        dim=8,
        steps=200,
        sigma=0.2,
        num_interp_points=10,
        dt=1e-2
    )
    noise = generator()
    assert len(noise) == 8


def test_normal_noise_generators():
    generator = NormalNoise(
        dim=2,
        repeat=2
    )
    generator.reset()
    noise1 = generator()
    noise2 = generator()
    assert np.all(noise1 == noise2)
    noise3 = generator()
    assert np.all(noise2 != noise3)


@pytest.mark.parametrize("env_name", [
    "InvertedPendulum-v4",
    "Ant-v4",
])
def test_data_loader(env_name, observation_model):
    num_time_steps = 18
    env = gym.make(env_name, render_mode="rgb_array")
    action_shape, = env.action_space.shape
    data_loader = EnvDataLoader(
        num_time_steps=num_time_steps + 1,
        img_shape=(3, 64, 64),
        transforms=Compose([Resize((64, 64))]),
        observation_model=observation_model,
        env=env,
        rollout_length=100,
        noise_generator=LinearSegmentNoiseND(
            dim=action_shape,
            steps=100,
            sigma=0.2,
            num_interp_points=10,
            dt=1e-2
        )
    )
    for _ in range(4):
        data_loader.perform_rollout()