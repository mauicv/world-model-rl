from reflect.components.transformer_world_model.embedder import Embedder
import torch


def test_embedder():
    om = Embedder(
        z_dim=1024,
        a_size=18,
        hidden_dim=256
    )
    z = torch.zeros((2, 16, 1024))
    a = torch.zeros((2, 16, 18))
    z_r = om((z, a))
    assert z_r.shape == (2, 16, 256)