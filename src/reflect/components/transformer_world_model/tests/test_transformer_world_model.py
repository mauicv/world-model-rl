from reflect.components.transformer_world_model import TransformerWorldModel
from reflect.components.transformer_world_model.embedder import Embedder as Embedder
from reflect.components.transformer_world_model.head import Head as Head
import gymnasium as gym
import torch
import pytest


def test_save_load(tmp_path, transformer_world_model):
    transformer_world_model.save(tmp_path)
    transformer_world_model.load(tmp_path)


def test_world_model(transformer_world_model: TransformerWorldModel):
    o, a, r, d = (
        torch.randn((2, 17, 3, 64, 64)) * 255,
        torch.randn((2, 17, 8)),
        torch.randn((2, 17, 1)),
        torch.randn((2, 17, 1))
    )
    first_seq, next_seq = transformer_world_model.observe_rollout(o, a, r, d)
    assert next_seq.state_dist.base_dist.probs.shape == (2, 16, 32, 32)
    assert next_seq.state_sample.shape == (2, 16, 32 * 32)
    assert next_seq.reward.base_dist.mean.shape == (2, 16, 1)
    assert next_seq.done.base_dist.mean.shape == (2, 16, 1)
    assert first_seq.state_dist.base_dist.logits.shape == (2, 16, 32, 32)
    assert first_seq.state_sample.shape == (2, 16, 1024)
    assert first_seq.action.shape == (2, 16, 8)
    assert first_seq.reward.base_dist.mean.shape == (2, 16, 1)
    assert first_seq.done.base_dist.mean.shape == (2, 16, 1)


def test_world_model_update(transformer_world_model: TransformerWorldModel):
    o, a, r, d = (
        torch.randn((2, 17, 3, 64, 64)) * 255,
        torch.randn((2, 17, 8)),
        torch.randn((2, 17, 1)),
        torch.randn((2, 17, 1))
    )
    target, output = transformer_world_model.observe_rollout(o, a, r, d)
    losses = transformer_world_model.update(target=target, output=output, observations=o)
    assert losses.dynamic_model_loss > 0
    assert losses.reward_loss > 0
    assert losses.done_loss > 0
    assert losses.recon_loss > 0
