from reflect.models.world_model import WorldModel
from reflect.models.world_model.embedder import Embedder as Embedder
from reflect.models.world_model.head import Head as Head
import gymnasium as gym
import torch
import pytest


@pytest.mark.parametrize("timesteps", [5, 16, 18])
def test_world_model(timesteps, observation_model, dynamic_model_8d_action):
    dm = dynamic_model_8d_action
    wm = WorldModel(
        observation_model=observation_model,
        dynamic_model=dm,
        num_ts=16,
    )

    o = torch.zeros((2, timesteps, 3, 64, 64))
    a = torch.zeros((2, timesteps, 8))
    r = torch.zeros((2, timesteps, 1))
    d = torch.zeros((2, timesteps, 1))

    z = wm.encode(o)
    assert z.shape == (2, timesteps, 1024)

    z, r, d = wm.step(z=z, a=a, r=r, d=d)
    assert z.shape == (2, timesteps+1, 1024)
    assert r.shape == (2, timesteps+1, 1)
    assert d.shape == (2, timesteps+1, 1)



@pytest.mark.parametrize("timesteps", [16])
def test_world_model(timesteps, observation_model, dynamic_model_8d_action):
    dm = dynamic_model_8d_action
    wm = WorldModel(
        observation_model=observation_model,
        dynamic_model=dm,
        num_ts=16,
    )

    o = torch.zeros((2, timesteps+1, 3, 64, 64))
    a = torch.zeros((2, timesteps+1, 8))
    r = torch.zeros((2, timesteps+1, 1))
    d = torch.zeros((2, timesteps+1, 1))

    results = wm.update(o, a, r, d)

    for key in ['recon_loss', 'reg_loss',
                'consistency_loss', 'dynamic_loss',
                'reward_loss', 'done_loss']:
        assert key in results


def test_save_load(tmp_path, observation_model, dynamic_model_8d_action):
    dm = dynamic_model_8d_action
    wm = WorldModel(
        observation_model=observation_model,
        dynamic_model=dm,
        num_ts=16,
    )    

    wm.save(tmp_path)
    wm.load(tmp_path)