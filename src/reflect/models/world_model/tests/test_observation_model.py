from reflect.models.world_model.observation_model import ObservationalModel
import torch


def test_observation_model(observation_model):
    assert observation_model is not None
    o = torch.zeros((1, 3, 64, 64))
    o_r, z, z_dist = observation_model(o)
    assert o_r.shape == (1, 3, 64, 64)
    assert z.shape == (1, 1024)
    assert z_dist.base_dist.mean.shape == (1, 1024)