from reflect.components.transformer_world_model.transformer import Transformer
from reflect.components.transformer_world_model.state import (
    Sequence,
    ImaginedRollout,
    StateSample,
    StateDistribution
)
from reflect.components.actor import Actor
import torch


def test_StateDistribution():
    continuous_mean = torch.zeros((2, 9, 12))
    continuous_std = torch.abs(torch.zeros((2, 9, 12))) + 0.01
    logits = torch.zeros((2, 9, 8, 6))
    reward = torch.zeros((2, 9, 1))
    done = torch.zeros((2, 9, 1))
    state = StateDistribution.from_sard(
        continuous_mean=continuous_mean,
        continuous_std=continuous_std,
        discrete=logits,
        reward=reward,
        done=done,
    )

    assert state.continuous_state.base_dist.mean.shape == (2, 9, 12)
    assert state.discrete_state.base_dist.probs.shape == (2, 9, 8, 6)
    assert state.reward_dist.base_dist.mean.shape == (2, 9, 1)
    assert state.done_dist.base_dist.mean.shape == (2, 9, 1)

    sample = state.rsample()

    assert sample.continuous_state.shape == (2, 9, 12)
    assert sample.discrete_state.shape == (2, 9, 8 * 6)
    assert sample.reward.shape == (2, 9, 1)
    assert sample.done.shape == (2, 9, 1)

def test_dynamic_model(transformer: Transformer):
    continuous_state = torch.zeros((2, 9, 16))
    discrete_state = torch.zeros((2, 9, 8 * 6))
    reward = torch.zeros((2, 9, 1))
    done = torch.zeros((2, 9, 1))
    action = torch.zeros((2, 9, 1))
    transformer_input = Sequence.from_sard(
        continuous_state=continuous_state,
        discrete_state=discrete_state,
        reward=reward,
        done=done,
        action=action
    )
    transformer_output: Sequence = transformer(transformer_input)
    assert transformer_output.state_distribution.continuous_state.base_dist.mean.shape == (2, 9, 16)
    assert transformer_output.state_distribution.discrete_state.base_dist.probs.shape == (2, 9, 8, 6)
    assert transformer_output.state_distribution.reward_dist.base_dist.mean.shape == (2, 9, 1)
    assert transformer_output.state_distribution.done_dist.base_dist.mean.shape == (2, 9, 1)
    assert transformer_output.state_sample.features.shape == (2, 9, 64)
    assert transformer_output.state_sample.reward.shape == (2, 9, 1)
    assert transformer_output.state_sample.done.shape == (2, 9, 1)


def test_dynamic_model_step(transformer: Transformer):
    state = torch.zeros((2, 3, 64))
    action = torch.zeros((2, 3, 1))
    reward = torch.zeros((2, 3, 1))
    done = torch.zeros((2, 3, 1))
    transformer_input = ImaginedRollout(
        state=state,
        action=action,
        reward=reward,
        done=done
    )
    transformer_output: ImaginedRollout = transformer.step(transformer_input)
    assert transformer_output.state.shape == (2, 4, 64)
    assert transformer_output.action.shape == (2, 3, 1)
    assert transformer_output.reward.shape == (2, 4, 1)
    assert transformer_output.done.shape == (2, 4, 1)


def test_trasnformer_imagine_rollout(transformer: Transformer, actor: Actor):
    state = torch.zeros((2, 1, 64))
    action = torch.zeros((2, 1, 1))
    reward = torch.zeros((2, 1, 1))
    done = torch.zeros((2, 1, 1))
    transformer_input = ImaginedRollout(
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
    