from reflect.components.transformer_world_model.embedder import ConcatEmbedder, AddEmbedder, StackEmbedder
import torch

def test_stack_embedder():
    om = StackEmbedder(
        z_dim=1024,
        a_size=18,
        hidden_dim=256
    )
    z = torch.zeros((2, 16, 1024))
    a = torch.zeros((2, 16, 18))
    r = torch.zeros((2, 16, 1))
    z_r = om((z, a, r))
    assert z_r.shape == (2, 3*16, 256)


def test_concat_embedder():
    om = ConcatEmbedder(
        z_dim=1024,
        a_size=18,
        hidden_dim=256
    )
    z = torch.zeros((2, 16, 1024))
    a = torch.zeros((2, 16, 18))
    r = torch.zeros((2, 16, 1))
    z_r = om((z, a, r))
    assert z_r.shape == (2, 16, 3*256)


def test_add_embedder():
    om = AddEmbedder(
        z_dim=1024,
        a_size=18,
        hidden_dim=256
    )
    z = torch.zeros((2, 16, 1024))
    a = torch.zeros((2, 16, 18))
    r = torch.zeros((2, 16, 1))
    z_r = om((z, a, r))
    assert z_r.shape == (2, 16, 256)