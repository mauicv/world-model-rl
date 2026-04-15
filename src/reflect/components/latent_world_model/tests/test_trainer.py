import torch
from reflect.components.latent_world_model.world_model import LatentWorldModel
from reflect.components.latent_world_model.models import MLPActor
from reflect.components.trainers.td3.td3_trainer import TD3Trainer
from reflect.components.trainers.td3.critic import TD3Critic
from reflect.components.latent_world_model.tests.conftest import LATENT_DIM


def make_trainer(actor, env_bipedal_walker, n_steps=1):
    action_dim = env_bipedal_walker.action_space.shape[0]
    critic_1 = TD3Critic(state_dim=LATENT_DIM, action_dim=action_dim)
    critic_2 = TD3Critic(state_dim=LATENT_DIM, action_dim=action_dim)
    return TD3Trainer(
        actor=actor,
        critics=[critic_1, critic_2],
        n_steps=n_steps,
    )


def test_training_iteration(encoder, dynamic_model, actor, env_data_loader, env_bipedal_walker):
    wm = LatentWorldModel(encoder=encoder, dynamic_model=dynamic_model)
    trainer = make_trainer(actor, env_bipedal_walker)

    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=4)

    # World model update
    wm_losses, (z, _, _, _) = wm.update(o, a, r, d, return_init_states=True)

    # Flatten encoded states to use each timestep as an independent starting point
    b, t, latent_dim = z.shape
    z_init = z.reshape(b * t, latent_dim)

    # Imagine rollout from encoded starting states
    z_traj, a_traj, r_traj, d_traj = wm.imagine_rollout(
        z=z_init,
        actor=actor,
        num_timesteps=8,
        disable_gradients=True,
    )

    # Critic update
    critic_losses = trainer.update_critics(z_traj, r_traj, d_traj, a_traj)

    # Actor update (also updates target networks)
    actor_losses = trainer.update_actor(z_traj)

    assert torch.isfinite(torch.tensor(wm_losses.consistency_loss))
    assert torch.isfinite(torch.tensor(wm_losses.reward_loss))
    assert all(torch.isfinite(torch.tensor(l)) for l in critic_losses.value_losses)
    assert torch.isfinite(torch.tensor(actor_losses.actor_loss))


def test_training_iteration_n_step(encoder, dynamic_model, actor, env_data_loader, env_bipedal_walker):
    n_steps = 3
    num_timesteps = 8

    wm = LatentWorldModel(encoder=encoder, dynamic_model=dynamic_model)
    trainer = make_trainer(actor, env_bipedal_walker, n_steps=n_steps)

    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=4)

    _, (z, _, _, _) = wm.update(o, a, r, d, return_init_states=True)
    b, t, latent_dim = z.shape
    z_init = z.reshape(b * t, latent_dim)

    z_traj, a_traj, r_traj, d_traj = wm.imagine_rollout(
        z=z_init,
        actor=actor,
        num_timesteps=num_timesteps,
        disable_gradients=True,
    )

    # With n_steps=3 and T=num_timesteps transitions, expect T-n+1 valid critic targets
    critic_losses = trainer.update_critics(z_traj, r_traj, d_traj, a_traj)
    actor_losses = trainer.update_actor(z_traj)

    assert all(torch.isfinite(torch.tensor(l)) for l in critic_losses.value_losses)
    assert torch.isfinite(torch.tensor(actor_losses.actor_loss))
