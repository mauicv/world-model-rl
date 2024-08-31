from reflect.components.transformer_world_model.head import Head
import torch
import pytest


def test_head_model(reward_model, done_model, predictor):
    head = Head(
        predictor=predictor,
        reward_model=reward_model,
        done_model=done_model,
        continuous_latent_dim=16,
        discrete_latent_dim=8,
        num_cat=6,
        hidden_dim=64,
    )
    z = torch.zeros((2, 48, 64))
    state = head(z)
    assert state.continuous_state.mean.shape == (2, 16, 16)
    assert state.discrete_state.base_dist.probs.shape == (2, 16, 8, 6)
    assert state.reward_dist.base_dist.mean.shape == (2, 16, 1)
    assert state.done_dist.base_dist.mean.shape == (2, 16, 1)
