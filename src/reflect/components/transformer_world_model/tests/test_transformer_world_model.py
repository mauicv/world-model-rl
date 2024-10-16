from reflect.components.transformer_world_model import WorldModel
import gymnasium as gym
from dataclasses import asdict
import torch
import pytest


@pytest.mark.parametrize("timesteps", [5, 16, 18])
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

    z, r, d = wm.dynamic_model.step(z=z, a=a, r=r, d=d)
    assert z.shape == (2, timesteps+1, 1024)
    assert r.shape == (2, timesteps+1, 1)
    assert d.shape == (2, timesteps+1, 1)


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


@pytest.mark.parametrize("timesteps", [16])
@pytest.mark.parametrize("return_init_states", [True, False])
def test_world_model(timesteps, return_init_states, encoder, decoder, dynamic_model_8d_action):
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
        results, (z, a, r, d) = wm.update(o, a, r, d, return_init_states=return_init_states)
        assert z.shape == (2*(timesteps + 1), 1, 1024)
        assert a.shape == (2*(timesteps + 1), 1, 8)
        assert r.shape == (2*(timesteps + 1), 1, 1)
        assert d.shape == (2*(timesteps + 1), 1, 1)
    else:
        results = wm.update(o, a, r, d)

    for key in ['recon_loss', 'reg_loss',
                'consistency_loss', 'dynamic_loss',
                'reward_loss', 'done_loss']:
        assert key in asdict(results)


def test_save_load(tmp_path, encoder, decoder, dynamic_model_8d_action):
    dm = dynamic_model_8d_action
    wm = WorldModel(
        encoder=encoder,
        decoder=decoder,
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
        z, a, r, d = wm.imagine_rollout(z=z, a=a, r=r, d=d, actor=actor)
    else:
        z, a, r, d, o = wm.imagine_rollout(
            z=z, a=a, r=r, d=d,
            actor=actor,
            with_observations=with_observations
        )
        assert o.shape == (34, 26, 3, 64, 64)
    assert z.shape == (34, 26, 1024)
    assert a.shape == (34, 26, 8)
    assert r.shape == (34, 26, 1)
    assert d.shape == (34, 26, 1)
