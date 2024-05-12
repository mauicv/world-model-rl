from reflect.models.world_model.observation_model import ObservationalModel
import torch


def test_observation_model():
    model = ObservationalModel(num_classes=32, num_latent=32)
    assert model is not None
    o = torch.zeros((1, 3, 64, 64))
    o_r, z, z_dist = model(o)
    assert o_r.shape == (1, 3, 64, 64)
    assert z.shape == (1, 32, 32)
    assert z_dist.base_dist.probs.shape == (1, 32, 32)