from reflect.models.world_model.head import Head
import torch


def test_observation_model():
    head = Head(
        latent_dim=32,
        num_cat=32,
        hidden_dim=256,
    )
    z = torch.zeros((2, 16, 256))
    z_dist, r, d = head(z)
    assert z_dist.base_dist.probs.shape == (2, 16, 32, 32)
    assert r.shape == (2, 16, 1)
    assert d.shape == (2, 16, 1)