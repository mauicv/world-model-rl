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


# @pytest.mark.parametrize("mask", [None, torch.tensor([0, 0, 1])])
# def test_sample_generation(
#         env_data_loader: EnvDataLoader,
#         world_model: WorldModel,
#         mask: Optional[torch.Tensor]
#     ):
#     for i in range(10):
#         env_data_loader.perform_rollout()

#     b_inds, t_inds, o_in, a, r_in, d_in = env_data_loader.sample(
#         batch_size=3,
#         num_time_steps=3
#     )
#     o_out, r_out, d_out = world_model.sample(
#         o=o_in,
#         a=a,
#         r=r_in,
#         d=d_in,
#         mask=mask,
#     )
#     assert o_out.shape == (3, 3, 4)
#     assert r_out.shape == (3, 3, 1)
#     assert d_out.shape == (3, 3, 1)

#     if mask is not None:
#         assert not torch.allclose(o_out[:, -1, :], o_in[:, -1, :])
#         assert not torch.allclose(r_out[:, -1, :], r_in[:, -1, :])
#         assert not torch.allclose(d_out[:, -1, :], d_in[:, -1, :].to(torch.float32))
#         assert torch.allclose(o_out[:, 0:2, :], o_in[:, 0:2, :])
#         assert torch.allclose(r_out[:, 0:2, :], r_in[:, 0:2, :])
#         assert torch.allclose(d_out[:, 0:2, :], d_in[:, 0:2, :].to(torch.float32))
#     else:
#         assert not torch.allclose(o_out, o_in)
#         assert not torch.allclose(r_out, r_in)
#         assert not torch.allclose(d_out, d_in.to(torch.float32))


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
