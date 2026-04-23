from reflect.data.loader import EnvDataLoader
from reflect.components.flow_world_model.dynamic_model import DynamicAttentionalFlowModel, DynamicFlowModel
from reflect.components.models.actor import Actor
import torch
import pytest
from typing import Optional



def test_dynamic_flow_model(
        env_data_loader: EnvDataLoader,
        dynamic_model: DynamicFlowModel,
    ):
    for i in range(10):
        env_data_loader.perform_rollout()

    b_inds, t_inds, s, a, r, d = env_data_loader.sample(
        batch_size=3,
        num_time_steps=4
    )

    x_cond = torch.cat([s[:, :3], a[:, :3]], dim=-1)
    x = torch.randn(s.shape[0], 1, dynamic_model.output_dim)
    t = torch.rand(3, 1, 1)
    u = dynamic_model.forward(x_cond, x, t)
    assert u.shape == (3, dynamic_model.output_dim)


def test_dynamic_attentional_flow_model(
        env_data_loader: EnvDataLoader,
        dynamic_attentional_model: DynamicAttentionalFlowModel,
    ):
    for i in range(10):
        env_data_loader.perform_rollout()

    b_inds, t_inds, s, a, r, d = env_data_loader.sample(
        batch_size=3,
        num_time_steps=4
    )
    x_cond = torch.cat([s[:, :3], a[:, :3]], dim=-1)
    x = torch.randn(s.shape[0], 1, dynamic_attentional_model.output_dim)
    t = torch.rand(3, 1, 1)
    u = dynamic_attentional_model.forward(x_cond, x, t)
    assert u.shape == (3, dynamic_attentional_model.output_dim)