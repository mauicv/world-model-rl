from reflect.data.loader import EnvDataLoader
from reflect.models.world_model.observation_model import ObservationalModel
from torchvision.transforms import Resize, Compose
import gymnasium as gym
import torch
import numpy as np
import pytest
from reflect.data.noise import NormalNoise

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
