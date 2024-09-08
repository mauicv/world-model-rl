from reflect.components.transformer_world_model.head import Head
import torch
import pytest


def test_head_model(predictor):
    head = Head(
        predictor=predictor,
        latent_dim=8,
        num_cat=8,
        hidden_dim=64,
    )
    z = torch.zeros((2, 16, 64))
    s_logits, hdn_state = head(z)
    assert s_logits.shape == (2, 16, 8, 8)
    assert hdn_state.shape == (2, 16, 64)