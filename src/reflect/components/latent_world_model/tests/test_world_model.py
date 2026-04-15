import torch
from reflect.components.latent_world_model.world_model import LatentWorldModel, LatentWorldModelLosses
from reflect.components.latent_world_model.tests.conftest import LATENT_DIM


def test_update(encoder, dynamic_model, env_data_loader):
    wm = LatentWorldModel(
        encoder=encoder,
        dynamic_model=dynamic_model,
    )
    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=4)

    losses = wm.update(o, a, r, d)

    assert isinstance(losses, LatentWorldModelLosses)
    assert torch.isfinite(torch.tensor(losses.consistency_loss))
    assert torch.isfinite(torch.tensor(losses.reward_loss))
    assert torch.isfinite(torch.tensor(losses.done_loss))
    assert torch.isfinite(torch.tensor(losses.grad_norm))


def test_update_return_init_states(encoder, dynamic_model, env_data_loader):
    wm = LatentWorldModel(
        encoder=encoder,
        dynamic_model=dynamic_model,
    )
    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=4)

    losses, (z, a_out, r_out, d_out) = wm.update(o, a, r, d, return_init_states=True)

    b, t, *_ = o.shape
    assert isinstance(losses, LatentWorldModelLosses)
    assert z.shape == (b, t, 512)
    assert a_out.shape == a.shape
    assert r_out.shape == r.shape
    assert d_out.shape == d.shape


def test_imagine_rollout(encoder, dynamic_model, actor, env_data_loader):
    wm = LatentWorldModel(
        encoder=encoder,
        dynamic_model=dynamic_model,
    )
    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=4)

    num_timesteps = 8
    b = o.shape[0]
    z_init = encoder(o[:, 0])  # (b, latent_dim)

    z_traj, a_traj, r_traj, d_traj = wm.imagine_rollout(
        z=z_init,
        actor=actor,
        num_timesteps=num_timesteps,
    )

    assert z_traj.shape == (b, num_timesteps + 1, LATENT_DIM)
    assert a_traj.shape[:2] == (b, num_timesteps + 1)
    assert r_traj.shape == (b, num_timesteps + 1, 1)
    assert d_traj.shape == (b, num_timesteps + 1, 1)

    # Final reward and done should be zero-padded
    assert torch.all(r_traj[:, -1] == 0)
    assert torch.all(d_traj[:, -1] == 0)


def test_ema_encoder_updates(encoder, dynamic_model, env_data_loader):
    wm = LatentWorldModel(
        encoder=encoder,
        dynamic_model=dynamic_model,
        ema_tau=0.005,
    )
    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=4)

    # Capture EMA params before update
    ema_params_before = [p.clone() for p in wm.ema_encoder.parameters()]

    wm.update(o, a, r, d)

    # EMA encoder should have moved toward the online encoder
    for before, after in zip(ema_params_before, wm.ema_encoder.parameters()):
        assert not torch.equal(before, after)

    # EMA encoder should never receive gradients
    for p in wm.ema_encoder.parameters():
        assert p.grad is None
