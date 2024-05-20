from reflect.models.world_model.embedder import Embedder as Embedder
from reflect.models.world_model.head import Head as Head
import torch


def test_dynamic_model(observation_model, dynamic_model_8d_action):
    o = torch.zeros((2, 16, 3, 64, 64))
    _, z, z_dist = observation_model(o.flatten(0, 1))
    z = z.reshape(-1, 16, 32 * 32)
    a = torch.zeros((2, 16, 8))
    r = torch.zeros((2, 16, 1))
    z_dist, r, d = dynamic_model_8d_action((z, a, r))
    assert z_dist.base_dist.probs.shape == (2, 16, 32, 32)
    assert r.shape == (2, 16, 1)
    assert d.shape == (2, 16, 1)
