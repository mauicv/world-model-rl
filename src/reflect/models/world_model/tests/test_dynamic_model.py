from reflect.models.world_model.observation_model import ObservationalModel
from reflect.models.world_model import WorldModel
from reflect.models.world_model import DynamicsModel
from reflect.models.world_model.embedder import Embedder as Embedder
from reflect.models.world_model.head import Head as Head
import torch


def test_observation_model():
    om = ObservationalModel()
    o = torch.zeros((2, 16, 3, 64, 64))
    _, z, z_dist = om(o.flatten(0, 1))
    z = z.reshape(-1, 16, 32 * 32)
    a = torch.zeros((2, 16, 8))
    r = torch.zeros((2, 16, 1))

    dm = DynamicsModel(
        hdn_dim=256,
        num_heads=8,
        a_size=8,
    )

    z_dist, r, d = dm((z, a, r))
    assert z_dist.base_dist.probs.shape == (2, 16, 32, 32)
    assert r.shape == (2, 16, 1)
    assert d.shape == (2, 16, 1)
