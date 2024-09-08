from reflect.components.observation_model.encoder import ConvEncoder
from reflect.components.transformer_world_model.state import Sequence, ImaginedRollout
import torch

def test_state(encoder: ConvEncoder):
    state = torch.zeros((2, 16, 64))
    action = torch.zeros((2, 16, 1))
    sequence = Sequence.from_sard(
        state=state,
        action=action,
    )
    assert sequence.state_dist.base_dist.probs.shape == (2, 16, 64)
    assert sequence.state_sample.shape == (2, 16, 64)
    initial_state = sequence.to_initial_state()
    assert initial_state.state.shape == (32, 1, 64)
    assert initial_state.action.shape == (32, 1, 1)
