from reflect.data.loader import EnvDataLoader
from reflect.components.flow_world_model.world_model import WorldModel
import torch
import pytest


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
    losses = world_model.update(o=o, a=a)
    assert losses


def test_flow_model_update_with_x_source(
        env_data_loader: EnvDataLoader,
        world_model: WorldModel,
    ):
    for i in range(10):
        env_data_loader.perform_rollout()

    b_inds, t_inds, o, a, r, d = env_data_loader.sample(
        batch_size=3,
        num_time_steps=4
    )
    x_source = o[:, [-2]] + 0.1 * torch.randn_like(o[:, [-2]])
    losses = world_model.update(o=o, a=a, x_source=x_source)
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
    x_cond = world_model.get_conditioning(s, a)
    x = world_model.get_initial_x(s)
    t = torch.zeros(batch_size, 1, 1, device=x.device)

    x_next, t_next = world_model._step_flow(
        x_cond=x_cond,
        x=x,
        t=t,
        delta=0.25,
        step_type=step_type,
    )
    assert x_next.shape == (batch_size, 1, world_model.observation_dim)
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
    o_pred = world_model.step_dynamics(o=o, a=a, num_flow_steps=10)
    assert o_pred.shape == (3, 1, world_model.observation_dim)


def test_correct(
        env_data_loader: EnvDataLoader,
        world_model: WorldModel,
    ):
    for i in range(10):
        env_data_loader.perform_rollout()

    b_inds, t_inds, o, a, r, d = env_data_loader.sample(
        batch_size=3,
        num_time_steps=5
    )
    o_decoded = o[:, [-1]] + 0.1 * torch.randn_like(o[:, [-1]])
    o_corrected = world_model.correct(o=o[:, :-1], a=a[:, :-1], o_decoded=o_decoded)
    assert o_corrected.shape == (3, 1, world_model.observation_dim)
