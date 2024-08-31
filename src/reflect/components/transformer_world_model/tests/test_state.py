from reflect.components.transformer_world_model.state import Sequence
import torch


def test_state():
    continuous_state = torch.zeros((2, 16, 16))
    discrete_state = torch.zeros((2, 16, 48))
    action = torch.zeros((2, 16, 1))
    reward = torch.zeros((2, 16, 1))
    done = torch.zeros((2, 16, 1))
    sequence = Sequence.from_sard(
        continuous_state=continuous_state,
        discrete_state=discrete_state,
        action=action,
        reward=reward,
        done=done
    )
    assert sequence.state_sample.continuous_state.shape == (2, 16, 16)
    assert sequence.state_sample.discrete_state.shape == (2, 16, 48)
    assert sequence.state_sample.reward.shape == (2, 16, 1)
    assert sequence.state_sample.done.shape == (2, 16, 1)
    initial_state = sequence.to_initial_state()
    assert initial_state.state.shape == (32, 1, 64)
    assert initial_state.action.shape == (32, 1, 1)
    assert initial_state.reward.shape == (32, 1, 1)
    assert initial_state.done.shape == (32, 1, 1)
