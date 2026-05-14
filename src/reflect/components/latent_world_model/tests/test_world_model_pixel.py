"""
Tests confirming LatentWorldModel works with ConvEncoder for pixel observations.
The world model itself is obs-shape-agnostic; the only requirement is that the
encoder handles (b, C, H, W) and (b, t, C, H, W) inputs, which ConvEncoder does.
"""
import copy

import torch
from reflect.components.latent_world_model.world_model import LatentWorldModel, LatentWorldModelLosses
from reflect.components.latent_world_model.models import MLPDynamicModel
from reflect.components.recon_world_model.models import ConvEncoder

IMAGE_SHAPE = (3, 64, 64)
LATENT_DIM = 512
ACTION_DIM = 4
BATCH, TIME = 4, 10


def make_wm():
    encoder = ConvEncoder(input_shape=IMAGE_SHAPE, latent_dim=LATENT_DIM)
    dynamic_model = MLPDynamicModel(
        latent_dim=LATENT_DIM, action_dim=ACTION_DIM, num_layers=3, hidden_dim=512
    )
    return LatentWorldModel(encoder=encoder, dynamic_model=dynamic_model)


def pixel_batch():
    o = torch.randn(BATCH, TIME, *IMAGE_SHAPE)
    a = torch.randn(BATCH, TIME, ACTION_DIM)
    r = torch.randn(BATCH, TIME, 1)
    d = torch.zeros(BATCH, TIME, 1)
    return o, a, r, d


def test_pixel_update():
    wm = make_wm()
    o, a, r, d = pixel_batch()

    losses = wm.update(o, a, r, d)

    assert isinstance(losses, LatentWorldModelLosses)
    assert torch.isfinite(torch.tensor(losses.consistency_loss))
    assert torch.isfinite(torch.tensor(losses.reward_loss))
    assert torch.isfinite(torch.tensor(losses.grad_norm))


def test_pixel_update_return_init_states():
    wm = make_wm()
    o, a, r, d = pixel_batch()

    losses, (z, a_out, r_out, d_out) = wm.update(o, a, r, d, return_init_states=True)

    assert isinstance(losses, LatentWorldModelLosses)
    assert z.shape == (BATCH, TIME, LATENT_DIM)
    assert a_out.shape == a.shape


def test_ema_encoder_is_identical_copy_at_init():
    """deepcopy must produce identical weights — requires Linear not LazyLinear."""
    encoder = ConvEncoder(input_shape=IMAGE_SHAPE, latent_dim=LATENT_DIM)
    ema_encoder = copy.deepcopy(encoder)

    for p_online, p_ema in zip(encoder.parameters(), ema_encoder.parameters()):
        assert torch.equal(p_online, p_ema), "EMA encoder must start as an exact copy"


def test_ema_encoder_updates_on_pixel_obs():
    wm = make_wm()
    o, a, r, d = pixel_batch()

    ema_params_before = [p.clone() for p in wm.ema_encoder.parameters()]
    wm.update(o, a, r, d)

    for before, after in zip(ema_params_before, wm.ema_encoder.parameters()):
        assert not torch.equal(before, after)

    for p in wm.ema_encoder.parameters():
        assert p.grad is None
