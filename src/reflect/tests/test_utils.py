from reflect.utils import recon_loss_fn, reward_loss_fn, create_z_dist, kl_divergence_loss_fn
import torch


def test_recon_loss_fn_per_timestep():
    x = torch.randn(10, 24)
    y = torch.randn(10, 24)
    loss, ts_loss = recon_loss_fn(x, y)
    assert loss.shape == ()
    assert ts_loss.shape == (10, )

def test_reward_loss_fn_per_timestep():
    r = torch.randn(10, 12, 1)
    r_pred = torch.randn(10, 12, 1)
    loss, ts_loss = reward_loss_fn(r, r_pred)
    assert loss.shape == ()
    assert ts_loss.shape == (10, 12)

def test_cross_entropy_loss_fn_per_timestep():
    z = torch.randn(2, 16, 32, 32)
    z_hat = torch.randn(2, 16, 32, 32)
    z_dist = create_z_dist(z)
    z_hat_dist = create_z_dist(z_hat)
    loss, ts_loss = kl_divergence_loss_fn(z_dist, z_hat_dist)
    assert loss.shape == ()
    assert ts_loss.shape == (2, 16)
