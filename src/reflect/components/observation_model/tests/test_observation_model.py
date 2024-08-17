import torch
import pytest

@pytest.mark.skip(reason="Breaking changes, need to update tests")
def test_observation_model(observation_model):
    assert observation_model is not None
    o = torch.zeros((1, 3, 64, 64))
    o_r, z, z_logits = observation_model(o)
    assert o_r.shape == (1, 3, 64, 64)
    assert z.shape == (1, 32, 32)
    assert z_logits.shape == (1, 32, 32)