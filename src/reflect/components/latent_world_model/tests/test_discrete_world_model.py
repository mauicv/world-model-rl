import pytest
import torch
from reflect.components.latent_world_model.discrete_world_model import DiscreteLatentWorldModel, DiscreteLatentWorldModelLosses
from reflect.components.latent_world_model.tests.conftest import LATENT_DIM

FSQ_LEVELS = [8, 8]
NUM_GROUPS = LATENT_DIM // len(FSQ_LEVELS)   # 256
CODEBOOK_SIZE = 8 * 8                         # 64


@pytest.fixture
def discrete_world_model(discrete_encoder, discrete_dynamic_model):
    return DiscreteLatentWorldModel(
        encoder=discrete_encoder,
        dynamic_model=discrete_dynamic_model,
        latent_dim=LATENT_DIM,
        fsq_levels=FSQ_LEVELS,
    )


def test_step(discrete_world_model, env_bipedal_walker):
    b = 4
    action_dim = env_bipedal_walker.action_space.shape[0]
    z = torch.rand(b, LATENT_DIM) * 2 - 1
    a = torch.rand(b, action_dim)

    logits, codes, r, d = discrete_world_model._step(z, a)

    assert logits.shape == (b, NUM_GROUPS, CODEBOOK_SIZE)
    assert codes.shape == (b, LATENT_DIM)
    assert r.shape == (b, 1)
    assert d.shape == (b, 1)


def test_update(discrete_world_model, env_data_loader):
    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=4)

    losses = discrete_world_model.update(o, a, r, d)

    assert isinstance(losses, DiscreteLatentWorldModelLosses)
    assert torch.isfinite(torch.tensor(losses.consistency_loss))
    assert torch.isfinite(torch.tensor(losses.reward_loss))
    assert torch.isfinite(torch.tensor(losses.done_loss))
    assert torch.isfinite(torch.tensor(losses.grad_norm))


def test_update_return_init_states(discrete_world_model, env_data_loader):
    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=4)

    losses, (z, a_out, r_out, d_out) = discrete_world_model.update(o, a, r, d, return_init_states=True)

    b, t, *_ = o.shape
    assert isinstance(losses, DiscreteLatentWorldModelLosses)
    assert z.shape == (b, t, LATENT_DIM)
    assert a_out.shape == a.shape
    assert r_out.shape == r.shape
    assert d_out.shape == d.shape


def test_ema_encoder_updates(discrete_world_model, env_data_loader):
    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=4)

    ema_params_before = [p.clone() for p in discrete_world_model.ema_encoder.parameters()]

    discrete_world_model.update(o, a, r, d)

    for before, after in zip(ema_params_before, discrete_world_model.ema_encoder.parameters()):
        assert not torch.equal(before, after)

    for p in discrete_world_model.ema_encoder.parameters():
        assert p.grad is None


def test_imagine_rollout(discrete_world_model, actor, env_data_loader):
    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=4)

    num_timesteps = 8
    b = o.shape[0]
    z_init = discrete_world_model.encode(o[:, 0])

    z_traj, a_traj, r_traj, d_traj = discrete_world_model.imagine_rollout(
        z=z_init,
        actor=actor,
        num_timesteps=num_timesteps,
    )

    assert z_traj.shape == (b, num_timesteps + 1, LATENT_DIM)
    assert a_traj.shape[:2] == (b, num_timesteps + 1)
    assert r_traj.shape == (b, num_timesteps + 1, 1)
    assert d_traj.shape == (b, num_timesteps + 1, 1)
    assert torch.all(r_traj[:, -1] == 0)
    assert torch.all(d_traj[:, -1] == 0)
