from reflect.components.transformer_world_model.transformer import Transformer, Sequence, ImaginedRollout
from reflect.components.observation_model.decoder import ConvDecoder
from reflect.components.observation_model.encoder import ConvEncoder
from reflect.utils import create_z_dist
import torch
import pytest


def test_dynamic_model(transformer: Transformer, encoder: ConvEncoder):
    observation = torch.zeros((2, 16, 3, 64, 64))
    state = encoder(observation).reshape(2, 16, 32, 32)
    action = torch.zeros((2, 16, 8))
    reward = torch.zeros((2, 16, 1))
    done = torch.zeros((2, 16, 1))
    transformer_input = Sequence.from_sard(
        state=state,
        action=action,
        reward=reward,
        done=done
    )
    transformer_output: Sequence = transformer(transformer_input)
    assert transformer_output.state_dist.base_dist.probs.shape == (2, 16, 32, 32)
    assert transformer_output.state_sample.shape == (2, 16, 32*32)
    assert transformer_output.reward.base_dist.mean.shape == (2, 16, 1)
    assert transformer_output.done.base_dist.mean.shape == (2, 16, 1)


def test_dynamic_model_step(transformer: Transformer, encoder: ConvEncoder):
    observation = torch.zeros((2, 1, 3, 64, 64))
    state = encoder(observation).reshape(2, 1, 32, 32)
    state = create_z_dist(state).rsample()
    state = state.reshape(2, 1, 1024)
    action = torch.zeros((2, 1, 8))
    reward = torch.zeros((2, 1, 1))
    done = torch.zeros((2, 1, 1))
    transformer_input = ImaginedRollout(
        state=state,
        action=action,
        reward=reward,
        done=done
    )
    transformer_output: ImaginedRollout = transformer.step(transformer_input)
    assert transformer_output.state.shape == (2, 2, 1024)
    assert transformer_output.action.shape == (2, 1, 8)
    assert transformer_output.reward.shape == (2, 2, 1)
    assert transformer_output.done.shape == (2, 2, 1)
