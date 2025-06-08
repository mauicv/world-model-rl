from reflect.components.transformer_world_model.head import StackHead, ConcatHead, AddHead
import torch


def test_stack_head():
    head = StackHead(
        latent_dim=32,
        num_cat=32,
        hidden_dim=256,
    )
    z = torch.zeros((2, 48, 256))
    z_dist, (r, _), d = head(z)
    assert z_dist.base_dist.probs.shape == (2, 16, 32, 32)
    assert r.shape == (2, 16, 1)
    assert d.shape == (2, 16, 1)


def test_ensemble_stack_head():
    head = StackHead(
        ensemble_size=3,
        latent_dim=32,
        num_cat=32,
        hidden_dim=256,
    )
    z = torch.zeros((6, 48, 256))
    z_dist, (r, _), d = head(z)
    assert z_dist.base_dist.probs.shape == (6, 16, 32, 32)
    assert r.shape == (6, 16, 1)
    assert d.shape == (6, 16, 1)

def test_ensemble_stack_head_seed():
    head = StackHead(
        ensemble_size=3,
        latent_dim=32,
        num_cat=32,
        hidden_dim=256,
    )
    for layer, shape in zip(head.reward.layers, [524, 525, 513]):
        assert layer.layers[3].weight.shape[0] == shape


def test_concat_head():
    head = ConcatHead(
        latent_dim=32,
        num_cat=32,
        hidden_dim=256,
    )
    z = torch.zeros((2, 48, 3*256))
    z_dist, (r, _), d = head(z)
    assert z_dist.base_dist.probs.shape == (2, 48, 32, 32)
    assert r.shape == (2, 48, 1)
    assert d.shape == (2, 48, 1)


def test_add_head():
    head = AddHead(
        latent_dim=32,
        num_cat=32,
        hidden_dim=256,
    )
    z = torch.zeros((2, 48, 256))
    z_dist, (r, _), d = head(z)
    assert z_dist.base_dist.probs.shape == (2, 48, 32, 32)
    assert r.shape == (2, 48, 1)
    assert d.shape == (2, 48, 1)