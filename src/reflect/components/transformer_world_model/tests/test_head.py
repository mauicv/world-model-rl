from reflect.components.transformer_world_model.head import Head
import torch
import pytest


def test_head_model(reward_model, done_model, predictor):
    head = Head(
        predictor=predictor,
        reward_model=reward_model,
        done_model=done_model,
        latent_dim=8,
        num_cat=8,
        hidden_dim=64,
    )
    z = torch.zeros((2, 48, 64))
    s_logits, r_mean, d_mean = head(z)
    assert s_logits.shape == (2, 16, 8, 8)
    assert r_mean.shape == (2, 16, 1)
    assert d_mean.shape == (2, 16, 1)