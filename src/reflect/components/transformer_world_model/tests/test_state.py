from reflect.components.transformer_world_model.state import Sequence, StateDistribution
import torch


def test_state():
    continuous_mean = torch.zeros((2, 16, 16))
    continuous_std = torch.zeros((2, 16, 16))
    discrete_state = torch.zeros((2, 16, 8, 6))
    reward = torch.zeros((2, 16, 1))
    done = torch.zeros((2, 16, 1))
    action = torch.zeros((2, 16, 1))
    state = StateDistribution.from_sard(
        continuous_mean=continuous_mean,
        continuous_std=continuous_std,
        discrete=discrete_state,
        reward=reward,
        done=done
    )

    sequence = Sequence.from_distribution(
        state=state,
        action=action
    )
    assert sequence.state_sample.continuous_state.shape == (2, 16, 16)
    assert sequence.state_sample.discrete_state.shape == (2, 16, 48)
    assert sequence.state_sample.reward.shape == (2, 16, 1)
    assert sequence.state_sample.done.shape == (2, 16, 1)
    initial_state = sequence.to_initial_state()
    assert initial_state.state_features.shape == (32, 1, 64)
    assert initial_state.dist_features.shape == (32, 1, 80)
    assert initial_state.action.shape == (32, 1, 1)
    assert initial_state.reward.shape == (32, 1, 1)
    assert initial_state.done.shape == (32, 1, 1)
