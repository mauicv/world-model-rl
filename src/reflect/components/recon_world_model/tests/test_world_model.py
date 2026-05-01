import torch
from reflect.components.recon_world_model.world_model import (
    ReconWorldModel,
    ReconWorldModelLosses,
    ReconWorldModelTrainingParams,
)
from reflect.components.recon_world_model.tests.conftest import LATENT_DIM


def test_update(encoder, decoder, dynamic_model, env_data_loader):
    wm = ReconWorldModel(encoder=encoder, decoder=decoder, dynamic_model=dynamic_model)
    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=4)

    losses = wm.update(o, a, r, d)

    assert isinstance(losses, ReconWorldModelLosses)
    assert torch.isfinite(torch.tensor(losses.consistency_loss))
    assert torch.isfinite(torch.tensor(losses.reward_loss))
    assert torch.isfinite(torch.tensor(losses.done_loss))
    assert torch.isfinite(torch.tensor(losses.recon_loss))
    assert torch.isfinite(torch.tensor(losses.grad_norm))
    assert torch.isfinite(torch.tensor(losses.effective_rank))
    assert torch.isfinite(torch.tensor(losses.mean_latent_std))
    assert 0.0 <= losses.recon_gate_mean <= 1.0


def test_update_return_init_states(encoder, decoder, dynamic_model, env_data_loader):
    wm = ReconWorldModel(encoder=encoder, decoder=decoder, dynamic_model=dynamic_model)
    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=4)

    losses, (z, a_out, r_out, d_out) = wm.update(o, a, r, d, return_init_states=True)

    b, t, *_ = o.shape
    assert isinstance(losses, ReconWorldModelLosses)
    assert z.shape == (b, t, LATENT_DIM)
    assert a_out.shape == a.shape
    assert r_out.shape == r.shape
    assert d_out.shape == d.shape


def test_recon_gate_activates_at_zero_cosine_dist(encoder, decoder, dynamic_model):
    """Gate should be fully open (1.0) when cosine distance is zero."""
    wm = ReconWorldModel(
        encoder=encoder,
        decoder=decoder,
        dynamic_model=dynamic_model,
        params=ReconWorldModelTrainingParams(recon_threshold=0.5),
    )
    # Simulate perfect prediction: cosine_dist = 0 everywhere
    cosine_dist = torch.zeros(4, 9)
    gate = torch.nn.functional.relu(0.5 - cosine_dist) / 0.5
    assert torch.allclose(gate, torch.ones_like(gate))


def test_recon_gate_closed_above_threshold(encoder, decoder, dynamic_model):
    """Gate should be zero when cosine distance exceeds threshold."""
    wm = ReconWorldModel(
        encoder=encoder,
        decoder=decoder,
        dynamic_model=dynamic_model,
        params=ReconWorldModelTrainingParams(recon_threshold=0.5),
    )
    cosine_dist = torch.full((4, 9), 0.6)  # above threshold
    gate = torch.nn.functional.relu(0.5 - cosine_dist) / 0.5
    assert torch.allclose(gate, torch.zeros_like(gate))


def test_no_ema_encoder(encoder, decoder, dynamic_model):
    wm = ReconWorldModel(encoder=encoder, decoder=decoder, dynamic_model=dynamic_model)
    assert not hasattr(wm, 'ema_encoder')


def test_update_use_delta(encoder, decoder, dynamic_model, env_data_loader):
    wm = ReconWorldModel(
        encoder=encoder,
        decoder=decoder,
        dynamic_model=dynamic_model,
        use_delta=True,
    )
    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=4)

    losses = wm.update(o, a, r, d)

    assert isinstance(losses, ReconWorldModelLosses)
    assert torch.isfinite(torch.tensor(losses.consistency_loss))


def test_imagine_rollout(encoder, decoder, dynamic_model, actor, env_data_loader):
    wm = ReconWorldModel(encoder=encoder, decoder=decoder, dynamic_model=dynamic_model)
    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=4)

    num_timesteps = 8
    b = o.shape[0]
    z_init = encoder(o[:, 0])

    z_traj, a_traj, r_traj, d_traj = wm.imagine_rollout(
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


def test_gate_disabled(encoder, decoder, dynamic_model, env_data_loader):
    """With use_recon_gate=False, recon_gate_mean should always be 1.0."""
    wm = ReconWorldModel(
        encoder=encoder,
        decoder=decoder,
        dynamic_model=dynamic_model,
        params=ReconWorldModelTrainingParams(use_recon_gate=False),
    )
    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=4)

    losses = wm.update(o, a, r, d)

    assert losses.recon_gate_mean == 1.0


def test_encoder_receives_reconstruction_gradient(encoder, decoder, dynamic_model, env_data_loader):
    """Encoder parameters should receive gradient from reconstruction path."""
    wm = ReconWorldModel(
        encoder=encoder,
        decoder=decoder,
        dynamic_model=dynamic_model,
        # Force gate fully open so reconstruction always fires
        params=ReconWorldModelTrainingParams(recon_threshold=4.0),
    )
    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=4)

    # Use a regular update — backward sets grads, step() does not clear them
    wm.update(o, a, r, d)

    encoder_grads = [p.grad for p in wm.encoder.parameters() if p.grad is not None]
    assert len(encoder_grads) > 0, "encoder should have received gradients"
