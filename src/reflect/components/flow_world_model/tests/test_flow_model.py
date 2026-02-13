from reflect.data.loader import EnvDataLoader
from reflect.components.flow_world_model.world_model import WorldModel
from reflect.components.models.actor import Actor
import torch
import pytest
from typing import Optional


def test_flow_model_update(
        env_data_loader: EnvDataLoader,
        world_model: WorldModel,
    ):
    for i in range(10):
        env_data_loader.perform_rollout()

    b_inds, t_inds, o, a, r, d = env_data_loader.sample(
        batch_size=3,
        num_time_steps=4
    )
    losses = world_model.update(
        o=o,
        r=r,
        d=d,
        a=a,
    )
    assert losses


@pytest.mark.parametrize("step_type", ["euler"])
@pytest.mark.parametrize("batch_size", [1, 3])
def test__step_flow(
        step_type: str,
        batch_size: int,
        env_data_loader: EnvDataLoader,
        world_model: WorldModel
    ):
    for i in range(10):
        env_data_loader.perform_rollout()

    b_inds, t_inds, s, a, r, d = env_data_loader.sample(
        batch_size=batch_size,
        num_time_steps=3
    )
    x_cond = world_model.get_conditioning(s, a, r, d)
    x = world_model.get_initial_x(s, r, d)
    t = torch.zeros(batch_size, 1, 1, device=x.device)

    x_next, t_next = world_model._step_flow(
        x_cond=x_cond,
        x=x,
        t=t,
        delta=0.25,
        step_type=step_type,
    )
    assert x_next.shape == (batch_size, 1, 6)
    assert t_next.shape == (batch_size, 1, 1)
    assert not torch.allclose(x_next, x)


def test_step_dynamics(
        env_data_loader: EnvDataLoader,
        world_model: WorldModel,
    ):
    for i in range(10):
        env_data_loader.perform_rollout()

    b_inds, t_inds, o, a, r, d = env_data_loader.sample(
        batch_size=3,
        num_time_steps=3
    )
    o, r, d = world_model.step_dynamics(
        o=o,
        a=a,
        r=r,
        d=d,
        num_flow_steps=10,
        noise_scale=0.05,
    )
    assert o.shape == (3, 1, 4)
    assert r.shape == (3, 1, 1)
    assert d.shape == (3, 1, 1)


# def test_imagine_rollout(
#         env_data_loader: EnvDataLoader,
#         world_model: WorldModel,
#         actor: Actor
#     ):
#     for i in range(10):
#         env_data_loader.perform_rollout()

#     b_inds, t_inds, o, a, r, d = env_data_loader.sample(
#         batch_size=6,
#         num_time_steps=3
#     )

#     rollout = world_model.imagine_rollout(
#         o=o,
#         a=a,
#         r=r,
#         d=d,
#         actor=actor,
#         num_timesteps=10,
#     )

#     assert rollout.o.shape == (6, 13, 4)
#     assert rollout.a.shape == (6, 13, 1)
#     assert rollout.r.shape == (6, 13, 1)
#     assert rollout.d.shape == (6, 13, 1)
