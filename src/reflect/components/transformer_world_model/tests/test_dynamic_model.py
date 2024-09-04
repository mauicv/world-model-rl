from reflect.components.transformer_world_model.transformer import Transformer, Sequence, ImaginedRollout
from reflect.components.observation_model.encoder import ConvEncoder
from reflect.components.actor import Actor
from reflect.utils import create_z_dist
import torch
import pytest


def test_dynamic_model(transformer: Transformer):
    state = torch.zeros((2, 9, 64))
    action = torch.zeros((2, 9, 1))
    reward = torch.zeros((2, 9, 1))
    done = torch.zeros((2, 9, 1))
    transformer_input = Sequence.from_sard(
        state=state,
        action=action,
        reward=reward,
        done=done
    )
    transformer_output: Sequence = transformer(transformer_input)
    assert transformer_output.state_dist.base_dist.probs.shape == (2, 9, 8, 8)
    assert transformer_output.state_sample.shape == (2, 9, 64)
    assert transformer_output.reward.base_dist.mean.shape == (2, 9, 1)
    assert transformer_output.done.base_dist.mean.shape == (2, 9, 1)


def test_dynamic_model_step(transformer: Transformer):
    state = torch.zeros((2, 1, 64))
    state_logits = torch.zeros((2, 1, 64))
    action = torch.zeros((2, 1, 1))
    reward = torch.zeros((2, 1, 1))
    done = torch.zeros((2, 1, 1))
    transformer_input = ImaginedRollout(
        state_logits=state_logits,
        state=state,
        action=action,
        reward=reward,
        done=done
    )
    transformer_output: ImaginedRollout = transformer.step(transformer_input)
    assert transformer_output.state.shape == (2, 2, 64)
    assert transformer_output.action.shape == (2, 1, 1)
    assert transformer_output.reward.shape == (2, 2, 1)
    assert transformer_output.done.shape == (2, 2, 1)


def test_trasnformer_imagine_rollout(transformer: Transformer, actor: Actor):
    state_logits = torch.randn((2, 1, 64))
    state = torch.zeros((2, 1, 64))
    action = torch.zeros((2, 1, 1))
    reward = torch.zeros((2, 1, 1))
    done = torch.zeros((2, 1, 1))
    transformer_input = ImaginedRollout(
        state_logits=state_logits,
        state=state,
        action=action,
        reward=reward,
        done=done
    )
    transformer_output: ImaginedRollout = transformer.imagine_rollout(
        initial_state=transformer_input,
        actor=actor,
        n_steps=5,
    )
    assert transformer_output.state.shape == (2, 6, 64)
    assert transformer_output.action.shape == (2, 6, 1)
    assert transformer_output.reward.shape == (2, 6, 1)
    assert transformer_output.done.shape == (2, 6, 1)
    