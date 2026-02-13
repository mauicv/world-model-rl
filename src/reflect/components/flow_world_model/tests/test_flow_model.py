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


# @pytest.mark.parametrize("t_pred_ratio", [0, 0.1])
# def test_sample_generation_orbit(
#         env_data_loader: EnvDataLoader,
#         world_model: WorldModel,
#         t_pred_ratio: float
#     ):
#     for i in range(10):
#         env_data_loader.perform_rollout()

#     b_inds, t_inds, s, a, r, d = env_data_loader.sample(
#         batch_size=3,
#         num_time_steps=3
#     )
#     x = world_model.cat_states(s, r, d)
#     orbits, times = world_model._sample_orbit(
#         x=x,
#         a=a,
#         delta=0.25,
#         step_type='euler',
#     )
#     assert orbits.shape == (3, 3, 4, 6)

#     subsamples, subtimes = world_model._subsample_orbits(
#         orbits=orbits,
#         times=times,
#         num_samples=2,
#     )
#     assert subsamples.shape == (6, 3, 6)
#     assert subtimes.shape == (6, 3, 1)

# @pytest.mark.parametrize("step_type", ["euler", "rk2"])
# def test_step(
#         step_type: str,
#         env_data_loader: EnvDataLoader,
#         world_model: WorldModel
#     ):
#     for i in range(10):
#         env_data_loader.perform_rollout()

#     b_inds, t_inds, s, a, r, d = env_data_loader.sample(
#         batch_size=3,
#         num_time_steps=3
#     )
#     x = world_model.cat_states(s, r, d)
#     t = torch.zeros(3, 3, 1, device=x.device)

#     x_next, t_next = world_model._step(
#         x=x,
#         a=a,
#         t=t,
#         delta=0.25,
#         step_type=step_type,
#     )
#     assert x_next.shape == (3, 3, 6)
#     assert t_next.shape == (3, 3, 1)
#     assert not torch.allclose(x_next, x)


# @pytest.mark.parametrize("step_type", ["euler", "rk2"])
# def test_step_mask(
#         step_type: str,
#         env_data_loader: EnvDataLoader,
#         world_model: WorldModel
#     ):
#     for i in range(10):
#         env_data_loader.perform_rollout()

#     b_inds, t_inds, s, a, r, d = env_data_loader.sample(
#         batch_size=3,
#         num_time_steps=3
#     )
#     x = world_model.cat_states(s, r, d)
#     t = torch.zeros(3, 3, 1, device=x.device)

#     t = torch.ones(3, 3, 1, device=x.device)
#     t[:, [-1], :] = 0
    
#     x_next, t_next = world_model._step(
#         x=x,
#         a=a,
#         t=t,
#         delta=0.25,
#         step_type=step_type,
#         mask=torch.tensor([0, 0, 1], device=x.device),
#     )
#     assert torch.allclose(x_next[:, 0:2, :], x[:, 0:2, :])
#     assert not torch.allclose(x_next[:, 2:, :], x[:, -1, :])
#     assert x_next.shape == (3, 3, 6)
#     assert t_next.shape == (3, 3, 1)

# def test_rectified_update(
#         env_data_loader: EnvDataLoader,
#         world_model: WorldModel
#     ):
#     for i in range(10):
#         env_data_loader.perform_rollout()

#     b_inds, t_inds, s, a, r, d = env_data_loader.sample(
#         batch_size=3,
#         num_time_steps=3
#     )

#     losses = world_model.rectified_update(
#         o=s,
#         r=r,
#         d=d,
#         a=a,
#         delta=-0.25,
#     )

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
