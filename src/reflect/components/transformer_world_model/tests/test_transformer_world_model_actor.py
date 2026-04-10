from reflect.components.transformer_world_model.world_model import WorldModel
from reflect.components.transformer_world_model.world_model_actor import (
    EncoderActor,
    TransformerWorldModelActor,
    ObservationActor,
)
from reflect.components.models.actor import Actor

import torch


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


def test_transformer_world_model_actor_uses_world_model_encoder(
        state_encoder,
        state_decoder,
        dynamic_model_8d_action,
        actor
    ):
    wm = WorldModel(
        encoder=state_encoder,
        decoder=state_decoder,
        dynamic_model=dynamic_model_8d_action,
    )
    wrapped_actor = TransformerWorldModelActor(
        world_model=wm,
        actor=actor,
    )
    o = torch.zeros((1, 27))
    a_pred = wrapped_actor(o)
    assert a_pred.shape == (1, 1, 8)


def test_observation_actor_for_vector_observations(actor):
    vector_actor = Actor(
        input_dim=27,
        output_dim=8,
        bound=1,
        num_layers=2,
        hidden_dim=64,
    )
    wrapped_actor = ObservationActor(actor=vector_actor)
    o = torch.zeros((1, 27))
    a_pred = wrapped_actor(o)
    assert a_pred.shape == (1, 8)


def test_observation_actor_for_image_observations_with_flatten():
    image_actor = Actor(
        input_dim=3 * 8 * 8,
        output_dim=8,
        bound=1,
        num_layers=2,
        hidden_dim=64,
    )
    wrapped_actor = ObservationActor(
        actor=image_actor,
        flatten_observation=True,
    )
    o = torch.zeros((1, 3, 8, 8))
    a_pred = wrapped_actor(o)
    assert a_pred.shape == (1, 8)


def test_observation_actor_batch_time_vector_observations():
    vector_actor = Actor(
        input_dim=27,
        output_dim=8,
        bound=1,
        num_layers=2,
        hidden_dim=64,
    )
    wrapped_actor = ObservationActor(actor=vector_actor)
    o = torch.zeros((2, 1, 27))
    a_pred = wrapped_actor(o)
    assert a_pred.shape == (2, 1, 8)