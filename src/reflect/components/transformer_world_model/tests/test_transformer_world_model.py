from reflect.components.transformer_world_model import WorldModel
from reflect.components.flow_world_model.dynamic_model import DynamicFlowModel
from reflect.components.flow_world_model.world_model import WorldModel as FlowWorldModel
import gymnasium as gym
from dataclasses import asdict
import torch
import pytest


class RecordingActor(torch.nn.Module):
    def __init__(self, input_dim: int, action_dim: int = 8):
        super().__init__()
        self.policy = torch.nn.Linear(input_dim, action_dim)
        self.last_input_dim = None

    def forward(self, x, deterministic: bool = False):
        self.last_input_dim = x.shape[-1]
        mu = torch.tanh(self.policy(x))
        if deterministic:
            return mu
        std = torch.ones_like(mu) * 0.1
        dist = torch.distributions.normal.Normal(mu, std)
        return torch.distributions.independent.Independent(dist, 1)


@pytest.mark.parametrize("timesteps", [1])
def test_world_model_step(timesteps, encoder, decoder, dynamic_model_8d_action):
    dm = dynamic_model_8d_action
    wm = WorldModel(
        encoder=encoder, 
        decoder=decoder,
        dynamic_model=dm,
    )

    o = torch.zeros((2, timesteps, 3, 64, 64))
    a = torch.zeros((2, timesteps, 8))
    r = torch.zeros((2, timesteps, 1))
    d = torch.zeros((2, timesteps, 1))

    z, _ = wm.encode(o)
    z = z.reshape(2, timesteps, 1024)
    assert z.shape == (2, timesteps, 1024)

    z, r, d, kv_cache = wm.dynamic_model.step(z=z, a=a, r=r, d=d)
    assert z.shape == (2, timesteps+1, 1024)
    assert r.shape == (2, timesteps+1, 1)
    assert d.shape == (2, timesteps+1, 1)
    assert kv_cache is not None


@pytest.mark.parametrize("timesteps", [1])
def test_state_world_model_step(timesteps, state_encoder, state_decoder, dynamic_model_8d_action):
    dm = dynamic_model_8d_action
    wm = WorldModel(
        encoder=state_encoder, 
        decoder=state_decoder,
        dynamic_model=dm,
    )

    o = torch.zeros((2, timesteps, 27))
    a = torch.zeros((2, timesteps, 8))
    r = torch.zeros((2, timesteps, 1))
    d = torch.zeros((2, timesteps, 1))

    z, _ = wm.encode(o)
    z = z.reshape(2, timesteps, 1024)
    assert z.shape == (2, timesteps, 1024)

    z, r, d, kv_cache = wm.dynamic_model.step(z=z, a=a, r=r, d=d)
    assert z.shape == (2, timesteps+1, 1024)
    assert r.shape == (2, timesteps+1, 1)
    assert d.shape == (2, timesteps+1, 1)
    assert kv_cache is not None


@pytest.mark.parametrize("timesteps", [5, 16, 18])
def test_flatten_batch_time(timesteps, encoder, decoder, dynamic_model_8d_action):
    dm = dynamic_model_8d_action
    wm = WorldModel(
        encoder=encoder, 
        decoder=decoder,
        dynamic_model=dm,
    )

    o = torch.zeros((2, timesteps, 3, 64, 64))
    a = torch.zeros((2, timesteps, 8))
    r = torch.zeros((2, timesteps, 1))
    d = torch.zeros((2, timesteps, 1))

    z, _ = wm.encode(o)
    z = z.reshape(2, timesteps, 1024)
    
    z, a, r, d = wm.flatten_batch_time(z=z, a=a, r=r, d=d)
    assert z.shape == (2*timesteps, 1, 1024)
    assert a.shape == (2*timesteps, 1, 8)
    assert r.shape == (2*timesteps, 1, 1)
    assert d.shape == (2*timesteps, 1, 1)


@pytest.mark.parametrize("timesteps", [5, 16, 18])
def test_state_flatten_batch_time(timesteps, state_encoder, state_decoder, dynamic_model_8d_action):
    dm = dynamic_model_8d_action
    wm = WorldModel(
        encoder=state_encoder, 
        decoder=state_decoder,
        dynamic_model=dm,
    )

    o = torch.zeros((2, timesteps, 27))
    a = torch.zeros((2, timesteps, 8))
    r = torch.zeros((2, timesteps, 1))
    d = torch.zeros((2, timesteps, 1))

    z, _ = wm.encode(o)
    z = z.reshape(2, timesteps, 1024)
    
    z, a, r, d = wm.flatten_batch_time(z=z, a=a, r=r, d=d)
    assert z.shape == (2*timesteps, 1, 1024)
    assert a.shape == (2*timesteps, 1, 8)
    assert r.shape == (2*timesteps, 1, 1)
    assert d.shape == (2*timesteps, 1, 1)


@pytest.mark.parametrize("return_init_states", [True, False])
@pytest.mark.parametrize("training_mask", [None, torch.randint(0, 2, (2, 17))])
def test_world_model(return_init_states, training_mask, encoder, decoder, dynamic_model_8d_action):
    timesteps = 16
    dm = dynamic_model_8d_action
    wm = WorldModel(
        encoder=encoder, 
        decoder=decoder,
        dynamic_model=dm,
    )

    o = torch.zeros((2, timesteps+1, 3, 64, 64))
    a = torch.zeros((2, timesteps+1, 8))
    r = torch.zeros((2, timesteps+1, 1))
    d = torch.zeros((2, timesteps+1, 1))

    if return_init_states:
        results, (z, a, r, d) = wm.update(o, a, r, d, training_mask=training_mask, return_init_states=return_init_states)
        assert z.shape == (2*(timesteps + 1), 1, 1024)
        assert a.shape == (2*(timesteps + 1), 1, 8)
        assert r.shape == (2*(timesteps + 1), 1, 1)
        assert d.shape == (2*(timesteps + 1), 1, 1)
    else:
        results = wm.update(o, a, r, d, training_mask=training_mask,)

    for key in ['recon_loss', 'reg_loss',
                'consistency_loss', 'dynamic_loss',
                'reward_loss', 'done_loss']:
        assert key in asdict(results)


@pytest.mark.parametrize("return_init_states", [True, False])
@pytest.mark.parametrize("training_mask", [None, torch.randint(0, 2, (2, 17))])
def test_state_world_model(return_init_states, training_mask, state_encoder, state_decoder, dynamic_model_8d_action):
    timesteps = 16
    dm = dynamic_model_8d_action
    wm = WorldModel(
        encoder=state_encoder, 
        decoder=state_decoder,
        dynamic_model=dm,
    )

    o = torch.zeros((2, timesteps+1, 27))
    a = torch.zeros((2, timesteps+1, 8))
    r = torch.zeros((2, timesteps+1, 1))
    d = torch.zeros((2, timesteps+1, 1))

    if return_init_states:
        results, (z, a, r, d), o_pred = wm.update(
            o, a, r, d,
            training_mask=training_mask,
            return_init_states=return_init_states,
            return_decoded_predicted_latents=True,
        )
        assert z.shape == (2*(timesteps + 1), 1, 1024)
        assert a.shape == (2*(timesteps + 1), 1, 8)
        assert r.shape == (2*(timesteps + 1), 1, 1)
        assert d.shape == (2*(timesteps + 1), 1, 1)
        assert o_pred.shape == (2, timesteps, 27)
    else:
        results = wm.update(
            o, a, r, d,
            training_mask=training_mask,
        )

    for key in ['recon_loss', 'reg_loss',
                'consistency_loss', 'dynamic_loss',
                'reward_loss', 'done_loss']:
        assert key in asdict(results)

    assert results.recon_loss_per_timestep.shape == (2, timesteps+1)
    assert results.dynamic_loss_per_timestep.shape == (2, timesteps)
    assert results.reward_loss_per_timestep.shape == (2, timesteps)


def test_save_load(tmp_path, encoder, decoder, dynamic_model_8d_action):
    dm = dynamic_model_8d_action
    wm = WorldModel(
        encoder=encoder,
        decoder=decoder,
        dynamic_model=dm,
    )
    wm.save(tmp_path)
    wm.load(tmp_path)


def test_state_save_load(tmp_path, state_encoder, state_decoder, dynamic_model_8d_action):
    dm = dynamic_model_8d_action
    wm = WorldModel(
        encoder=state_encoder,
        decoder=state_decoder,
        dynamic_model=dm,
    )
    wm.save(tmp_path)
    wm.load(tmp_path)


@pytest.mark.parametrize("timesteps", [16])
@pytest.mark.parametrize("with_observations", [True, False])
def test_world_model_imagine_rollout(
        timesteps,
        with_observations,
        encoder,
        decoder,
        dynamic_model_8d_action,
        actor
    ):
    dm = dynamic_model_8d_action
    wm = WorldModel(
        encoder=encoder, 
        decoder=decoder,
        dynamic_model=dm,
    )
    o = torch.zeros((2, timesteps+1, 3, 64, 64))
    a = torch.zeros((2, timesteps+1, 8))
    r = torch.zeros((2, timesteps+1, 1))
    d = torch.zeros((2, timesteps+1, 1))
    _, (z, a, r, d) = wm.update(o, a, r, d, return_init_states=True)
    if not with_observations:
        z, a, r, d = wm.imagine_rollout(
            z=z, a=a, r=r, d=d,
            actor=actor,
            num_timesteps=16,
        )
    else:
        z, a, r, d, o = wm.imagine_rollout(
            z=z, a=a, r=r, d=d,
            actor=actor,
            with_observations=with_observations,
            num_timesteps=16,
        )
        assert o.shape == (34, 17, 3, 64, 64)
    assert z.shape == (34, 17, 1024)
    assert a.shape == (34, 17, 8)
    assert r.shape == (34, 17, 1)
    assert d.shape == (34, 17, 1)


@pytest.mark.parametrize("timesteps", [16])
@pytest.mark.parametrize("with_observations", [True, False])
def test_state_world_model_imagine_rollout(
        timesteps,
        with_observations,
        state_encoder,
        state_decoder,
        dynamic_model_8d_action,
        actor
    ):
    dm = dynamic_model_8d_action
    wm = WorldModel(
        encoder=state_encoder, 
        decoder=state_decoder,
        dynamic_model=dm,
    )
    o = torch.zeros((2, timesteps+1, 27))
    a = torch.zeros((2, timesteps+1, 8))
    r = torch.zeros((2, timesteps+1, 1))
    d = torch.zeros((2, timesteps+1, 1))
    _, (z, a, r, d) = wm.update(o, a, r, d, return_init_states=True)
    if not with_observations:
        z, a, r, d = wm.imagine_rollout(
            z=z, a=a, r=r, d=d,
            actor=actor,
            num_timesteps=16,
        )
    else:
        z, a, r, d, o = wm.imagine_rollout(
            z=z, a=a, r=r, d=d,
            actor=actor,
            with_observations=with_observations,
            num_timesteps=16,
        )
        assert o.shape == (34, 17, 27)
    assert z.shape == (34, 17, 1024)
    assert a.shape == (34, 17, 8)
    assert r.shape == (34, 17, 1)
    assert d.shape == (34, 17, 1)


@pytest.mark.parametrize("timesteps", [16])
def test_world_model_imagine_rollout_non_deterministic(
        timesteps,
        state_encoder,
        state_decoder,
        dynamic_model_8d_action,
        actor
    ):
    dm = dynamic_model_8d_action
    wm = WorldModel(
        encoder=state_encoder, 
        decoder=state_decoder,
        dynamic_model=dm,
    )
    o = torch.zeros((2, timesteps+1, 27))
    a = torch.zeros((2, timesteps+1, 8))
    r = torch.zeros((2, timesteps+1, 1))
    d = torch.zeros((2, timesteps+1, 1))
    _, (z, a, r, d) = wm.update(o, a, r, d, return_init_states=True)
    z, a, r, d, entropy = wm.imagine_rollout(
        z=z, a=a, r=r, d=d,
        actor=actor,
        with_entropies=True,
        num_timesteps=16,
    )
    assert z.shape == (34, 17, 1024)
    assert a.shape == (34, 17, 8)
    assert r.shape == (34, 17, 1)
    assert d.shape == (34, 17, 1)
    assert entropy.shape == (34, 17, 1)


@pytest.mark.parametrize("timesteps", [16])
def test_world_model_imagine_rollout_no_kvcache(
        timesteps,
        state_encoder,
        state_decoder,
        dynamic_model_8d_action,
        actor
    ):
    dm = dynamic_model_8d_action
    wm = WorldModel(
        encoder=state_encoder, 
        decoder=state_decoder,
        dynamic_model=dm,
    )
    o = torch.zeros((2, timesteps+1, 27))
    a = torch.zeros((2, timesteps+1, 8))
    r = torch.zeros((2, timesteps+1, 1))
    d = torch.zeros((2, timesteps+1, 1))
    _, (z, a, r, d) = wm.update(o, a, r, d, return_init_states=True)
    z, a, r, d = wm.imagine_rollout(
        z=z, a=a, r=r, d=d,
        use_kv_cache=False,
        actor=actor,
        num_timesteps=24,
    )
    assert z.shape == (34, 25, 1024)
    assert a.shape == (34, 25, 8)
    assert r.shape == (34, 25, 1)
    assert d.shape == (34, 25, 1)


def test_state_world_model_imagine_rollout_kvcache_actor_input_space_selection(
        state_encoder,
        state_decoder,
        dynamic_model_8d_action,
    ):
    timesteps = 16
    dm = dynamic_model_8d_action
    wm = WorldModel(
        encoder=state_encoder,
        decoder=state_decoder,
        dynamic_model=dm,
    )
    o = torch.zeros((2, timesteps + 1, 27))
    a = torch.zeros((2, timesteps + 1, 8))
    r = torch.zeros((2, timesteps + 1, 1))
    d = torch.zeros((2, timesteps + 1, 1))
    _, (z, a, r, d) = wm.update(o, a, r, d, return_init_states=True)

    latent_actor = RecordingActor(input_dim=1024)
    z_latent, a_latent, r_latent, d_latent = wm.imagine_rollout(
        z=z, a=a, r=r, d=d,
        actor=latent_actor,
        actor_in_latent_space=True,
        num_timesteps=8,
        use_kv_cache=True,
    )
    assert latent_actor.last_input_dim == 1024
    assert z_latent.shape == (34, 9, 1024)
    assert a_latent.shape == (34, 9, 8)
    assert r_latent.shape == (34, 9, 1)
    assert d_latent.shape == (34, 9, 1)

    reconstructed_actor = RecordingActor(input_dim=27)
    z_obs, a_obs, r_obs, d_obs = wm.imagine_rollout(
        z=z, a=a, r=r, d=d,
        actor=reconstructed_actor,
        actor_in_latent_space=False,
        num_timesteps=8,
        use_kv_cache=True,
    )
    assert reconstructed_actor.last_input_dim == 27
    assert z_obs.shape == (34, 9, 1024)
    assert a_obs.shape == (34, 9, 8)
    assert r_obs.shape == (34, 9, 1)
    assert d_obs.shape == (34, 9, 1)


def test_imagine_rollout_with_corrector(
        state_encoder,
        state_decoder,
        dynamic_model_8d_action,
    ):
    obs_dim = 27
    action_dim = 8
    num_positions = 3
    timesteps = 16

    wm = WorldModel(
        encoder=state_encoder,
        decoder=state_decoder,
        dynamic_model=dynamic_model_8d_action,
    )

    flow_dynamic = DynamicFlowModel(
        input_dim=obs_dim,
        conditioning_dim=num_positions * (obs_dim + action_dim),
        output_dim=obs_dim,
        time_embed_dim=16,
        hidden_dim=64,
        depth=2,
        use_layer_norm=True,
        num_positions=num_positions,
    )
    corrector = FlowWorldModel(
        dynamic_model=flow_dynamic,
        observation_dim=obs_dim,
        action_dim=action_dim,
        environment_action_bound=1.0,
    )

    o = torch.zeros((2, timesteps + 1, obs_dim))
    a = torch.zeros((2, timesteps + 1, action_dim))
    r = torch.zeros((2, timesteps + 1, 1))
    d = torch.zeros((2, timesteps + 1, 1))
    _, (z, a, r, d) = wm.update(o, a, r, d, return_init_states=True)

    actor = RecordingActor(input_dim=obs_dim, action_dim=action_dim)
    z_out, a_out, r_out, d_out, o_out = wm.imagine_rollout(
        z=z, a=a, r=r, d=d,
        actor=actor,
        actor_in_latent_space=False,
        corrector=corrector,
        with_observations=True,
        num_timesteps=8,
    )

    assert actor.last_input_dim == obs_dim
    assert z_out.shape == (34, 9, 1024)
    assert a_out.shape == (34, 9, action_dim)
    assert r_out.shape == (34, 9, 1)
    assert d_out.shape == (34, 9, 1)
    assert o_out.shape == (34, 9, obs_dim)


def test_state_world_model_imagine_rollout_no_kvcache_actor_input_space_selection(
        state_encoder,
        state_decoder,
        dynamic_model_8d_action,
    ):
    timesteps = 16
    dm = dynamic_model_8d_action
    wm = WorldModel(
        encoder=state_encoder,
        decoder=state_decoder,
        dynamic_model=dm,
    )
    o = torch.zeros((2, timesteps + 1, 27))
    a = torch.zeros((2, timesteps + 1, 8))
    r = torch.zeros((2, timesteps + 1, 1))
    d = torch.zeros((2, timesteps + 1, 1))
    _, (z, a, r, d) = wm.update(o, a, r, d, return_init_states=True)

    latent_actor = RecordingActor(input_dim=1024)
    z_latent, a_latent, r_latent, d_latent = wm.imagine_rollout(
        z=z, a=a, r=r, d=d,
        actor=latent_actor,
        actor_in_latent_space=True,
        num_timesteps=8,
        use_kv_cache=False,
    )
    assert latent_actor.last_input_dim == 1024
    assert z_latent.shape == (34, 9, 1024)
    assert a_latent.shape == (34, 9, 8)
    assert r_latent.shape == (34, 9, 1)
    assert d_latent.shape == (34, 9, 1)

    reconstructed_actor = RecordingActor(input_dim=27)
    z_obs, a_obs, r_obs, d_obs = wm.imagine_rollout(
        z=z, a=a, r=r, d=d,
        actor=reconstructed_actor,
        actor_in_latent_space=False,
        num_timesteps=8,
        use_kv_cache=False,
    )
    assert reconstructed_actor.last_input_dim == 27
    assert z_obs.shape == (34, 9, 1024)
    assert a_obs.shape == (34, 9, 8)
    assert r_obs.shape == (34, 9, 1)
    assert d_obs.shape == (34, 9, 1)
