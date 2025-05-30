from reflect.components.transformer_world_model.world_model import WorldModel
from reflect.components.transformer_world_model.world_model_actor import EncoderActor

from dataclasses import asdict
import torch
import pytest


def test_state_world_model_imagine_rollout(
        state_encoder,
        actor
    ):
    actor = EncoderActor(
        encoder=state_encoder,
        actor=actor,
        num_latent=32,
        num_cat=32
    )
    o = torch.zeros((1, 1, 27))
    a_pred = actor(o)
    assert a_pred.shape == (1, 1, 8)
    