from reflect.models.world_model.head import Head
import torch


def test_observation_model():
    head = Head(
        latent_dim=1024,
        hidden_dim=256,
    )
    z = torch.zeros((2, 16, 256))
    z_dist, r, d = head(z)
    assert z_dist.base_dist.mean.shape == (2, 16, 1024)
    assert r.shape == (2, 16, 1)
    assert d.shape == (2, 16, 1)