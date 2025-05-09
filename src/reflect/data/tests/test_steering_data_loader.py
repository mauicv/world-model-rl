from reflect.data.steered_loader import SteeredEnvDataLoader
from reflect.components.transformer_world_model.world_model_actor import EncoderActor
from reflect.components.models import ConvEncoder
from reflect.components.models import Actor

import gymnasium as gym
import torch
import pytest


@pytest.mark.parametrize("env_name", [
    "InvertedPendulum-v4",
    "Ant-v4",
])
def test_data_loader_imgs(env_name):
    num_time_steps = 18
    env = gym.make(env_name, render_mode="rgb_array")
    action_shape, = env.action_space.shape
    data_loader = SteeredEnvDataLoader(
        num_time_steps=num_time_steps + 1,
        state_shape=(3, 64, 64),
        use_imgs_as_states=True,
        # transforms=Compose([Resize((64, 64))]),
        env=env,
        num_steering=6,
        max_steering_std=25
    )
    for _ in range(25):
        data_loader.perform_rollout()
    
    for i in range(4):
        assert data_loader.end_index[i] >= num_time_steps, f'{data_loader.end_index=}'
        assert torch.all(data_loader.state_buffer[i, data_loader.end_index[i]+1:] == 0)
        assert torch.all(data_loader.action_buffer[i, data_loader.end_index[i]+1:] == 0)
        assert torch.all(data_loader.reward_buffer[i, data_loader.end_index[i]+1:] == 0)

    s, a, r, d = data_loader.sample(batch_size=3, num_time_steps=10)

    print(s.shape, a.shape, r.shape, d.shape)

    assert s.shape == (3, 10, 3, 64, 64)
    assert a.shape == (3, 10, action_shape)
    assert r.shape == (3, 1)
    assert d.shape == (3, 10, 1)
    data_loader.close()


@pytest.mark.parametrize("env_name", [
    "Ant-v4",
])
def test_data_loader_states(env_name):
    num_time_steps = 18
    env = gym.make(env_name, render_mode="rgb_array")
    action_shape, = env.action_space.shape
    data_loader = SteeredEnvDataLoader(
        num_time_steps=num_time_steps + 1,
        state_shape=(27,),
        use_imgs_as_states=False,
        env=env,
        num_steering=6,
        max_steering_std=25,
        steering_ratio=0.75
    )

    for _ in range(25):
        data_loader.perform_rollout()

    # TODO: add something to check that the steering labels are correct

    for i in range(4):
        assert data_loader.end_index[i] >= num_time_steps, f'{data_loader.end_index=}'
        assert torch.all(data_loader.state_buffer[i, data_loader.end_index[i]+1:] == 0)
        assert torch.all(data_loader.action_buffer[i, data_loader.end_index[i]+1:] == 0)
        assert torch.all(data_loader.reward_buffer[i, data_loader.end_index[i]+1:] == 0)

    s, a, r, d = data_loader.sample(batch_size=8, num_time_steps=10)

    assert s.shape == (8, 10, 27)
    assert a.shape == (8, 10, action_shape)
    assert r.shape == (8, 1)
    assert d.shape == (8, 10, 1)

    assert torch.all(r[:6, 0] == 1)
    assert torch.all(r[6:, 0] == 0)
    data_loader.close()

@pytest.mark.parametrize("env_name", [
    "Ant-v4",
])
def test_data_loader_weight_perturbation(env_name):
    num_time_steps = 18
    env = gym.make(env_name, render_mode="rgb_array")
    action_shape, = env.action_space.shape

    encoder = ConvEncoder(
        input_shape=(3, 64, 64),
        embed_size=1024,
        activation=torch.nn.ReLU(),
        depth=32
    )

    actor = Actor(
        input_dim=1024,
        output_dim=8,
        bound=1,
        num_layers=3,
        hidden_dim=512,
    )

    policy = EncoderActor(
        encoder=encoder,
        actor=actor,
        num_latent=32,
        num_cat=32
    )

    weight_perturbation_size = 0.01
    data_loader = SteeredEnvDataLoader(
        num_time_steps=num_time_steps + 1,
        state_shape=(3, 64, 64),
        use_imgs_as_states=True,
        # transforms=Compose([Resize((64, 64))]),
        env=env,
        num_steering=6,
        max_steering_std=25
    )

    for _ in range(25):
        data_loader.perform_rollout()
    for i in range(4):
        assert data_loader.end_index[i] >= num_time_steps, f'{data_loader.end_index=}'
        assert torch.all(data_loader.state_buffer[i, data_loader.end_index[i]+1:] == 0)
        assert torch.all(data_loader.action_buffer[i, data_loader.end_index[i]+1:] == 0)
        assert torch.all(data_loader.reward_buffer[i, data_loader.end_index[i]+1:] == 0)

    s, a, r, d = data_loader.sample(batch_size=3, num_time_steps=10)

    print(s.shape, a.shape, r.shape, d.shape)

    assert s.shape == (3, 10, 3, 64, 64)
    assert a.shape == (3, 10, action_shape)
    assert r.shape == (3, 1)
    assert d.shape == (3, 10, 1)
    data_loader.close()
