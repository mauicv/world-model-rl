# from reflect.models.world_model.observation_model import ObservationalModel
from reflect.data.loader import EnvDataLoader
from reflect.models.world_model import WorldModel
from reflect.models.world_model.environment import Environment
# from reflect.models.world_model import DynamicsModel
from conftest import make_dynamic_model
from reflect.models.world_model.embedder import Embedder as Embedder
from reflect.models.world_model.head import Head as Head
from torchvision.transforms import Resize, Compose
import gymnasium as gym
import torch
import pytest


@pytest.mark.parametrize("env_name", [
    "InvertedPendulum-v4",
    "Ant-v4",
])
def test_environment(env_name, observation_model):
    batch_size=10
    real_env = gym.make(env_name, render_mode="rgb_array")
    action_size = real_env.action_space.shape[0]

    dm = make_dynamic_model(a_size=action_size)

    wm = WorldModel(
        observation_model=observation_model,
        dynamic_model=dm,
        num_ts=16,
    )

    dl = EnvDataLoader(
        num_time_steps=17,
        img_shape=(3, 64, 64),
        transforms=Compose([
            Resize((64, 64))
        ]),
        observation_model=observation_model,
        env=real_env
    )

    dl.perform_rollout()

    env = Environment(
        world_model=wm,
        data_loader=dl,
        batch_size=batch_size
    )
    state, _ = env.reset()
    assert state.shape == (batch_size, 1, 1024)
    
    cur_batch_size = batch_size
    for i in range(2, 25):
        actions = torch.zeros((cur_batch_size, 1, action_size))
        states, rewards, dones = env.step(actions)

        random_done = torch.randint(0, cur_batch_size, (1,)).item()
        dones[random_done, -1, 0] = 1
        env.dones[random_done, -1, 0] = 1
        num_not_done = (dones < 0.5).sum().item()

        if torch.all((dones>=0.5)):
            break

        assert states.shape == (cur_batch_size, 1, 1024)
        assert rewards.shape == (cur_batch_size, 1, 1)
        assert dones.shape == (cur_batch_size, 1, 1)

        assert env.states.shape == (cur_batch_size, i, 1024)
        assert env.rewards.shape == (cur_batch_size, i, 1)
        assert env.dones.shape == (cur_batch_size, i, 1)
        assert env.actions.shape == (cur_batch_size, i - 1, action_size)

        assert env.not_done.sum() == num_not_done
        cur_batch_size = num_not_done
