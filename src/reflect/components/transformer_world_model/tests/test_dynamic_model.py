import torch
from reflect.components.transformer_world_model.transformer import PytfexTransformer

def test_stack_transformer():
    hdn_dim = 512
    transformer = PytfexTransformer(
        num_ts=12,
        num_cat=32,
        dropout=0.1,
        num_heads=8,
        latent_dim=32,
        action_size=6,
        num_layers=12,
        hdn_dim=hdn_dim,
        embedding_type='stack'
    )

    actions = torch.zeros(2, 12, 6)
    states = torch.zeros(2, 12, 32*32)
    rewards = torch.zeros(2, 12, 1)
    h, z_dist, r, d = transformer(states, actions, rewards)
    assert h.shape == (2, 12, 3*hdn_dim)
    assert z_dist.base_dist.logits.shape == (2, 12, 32, 32)
    assert r.shape == (2, 12, 1)
    assert d.shape == (2, 12, 1)


def test_concat_transformer():
    hdn_dim=512
    transformer = PytfexTransformer(
        num_ts=12,
        num_cat=32,
        dropout=0.1,
        num_heads=8,
        latent_dim=32,
        action_size=6,
        num_layers=12,
        hdn_dim=hdn_dim,
        embedding_type='concat'
    )

    actions = torch.zeros(2, 12, 6)
    states = torch.zeros(2, 12, 32*32)
    rewards = torch.zeros(2, 12, 1)
    h, z_dist, r, d = transformer(states, actions, rewards)
    assert h.shape == (2, 12, 3*hdn_dim)
    assert z_dist.base_dist.logits.shape == (2, 12, 32, 32)
    assert r.shape == (2, 12, 1)
    assert d.shape == (2, 12, 1)


def test_add_transformer():
    hdn_dim=512
    transformer = PytfexTransformer(
        num_ts=12,
        num_cat=32,
        dropout=0.1,
        num_heads=8,
        latent_dim=32,
        action_size=6,
        num_layers=12,
        hdn_dim=hdn_dim,
        embedding_type='add'
    )

    actions = torch.zeros(2, 12, 6)
    states = torch.zeros(2, 12, 32*32)
    rewards = torch.zeros(2, 12, 1)
    h, z_dist, r, d = transformer(states, actions, rewards)
    assert h.shape == (2, 12, hdn_dim)
    assert z_dist.base_dist.logits.shape == (2, 12, 32, 32)
    assert r.shape == (2, 12, 1)
    assert d.shape == (2, 12, 1)