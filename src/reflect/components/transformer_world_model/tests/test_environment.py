# from reflect.models.world_model.observation_model import ObservationalModel
from reflect.data.loader import EnvDataLoader
# from reflect.components.transformer_world_model import WorldModel
from reflect.components.transformer_world_model.environment import Environment
# from reflect.models.world_model import DynamicsModel
from reflect.components.transformer_world_model.embedder import Embedder as Embedder
from reflect.components.transformer_world_model.head import Head as Head
from torchvision.transforms import Resize, Compose
import gymnasium as gym
import torch
import pytest


@pytest.mark.skip(reason="Breaking changes, need to update tests")
@pytest.mark.parametrize("env_name", [
    "InvertedPendulum-v4",
    "Ant-v4",
])
def test_environment_step_filter(env_name, observation_model):
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
    env.dones = torch.zeros((batch_size, 1, 1))
    cur_batch_size = batch_size
    for i in range(2, 25):
        actions = torch.zeros((cur_batch_size, 1, action_size))
        states, rewards, dones = env.step_filter(actions)
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


@pytest.mark.skip(reason="Breaking changes, need to update tests")
@pytest.mark.parametrize("env_name", [
    "InvertedPendulum-v4",
    "Ant-v4",
])
def test_environment_step(env_name, observation_model):
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
    env.dones = torch.zeros((batch_size, 1, 1))
    for i in range(2, 25):
        actions = torch.zeros((batch_size, 1, action_size))
        states, rewards, dones = env.step(actions)

        assert states.shape == (batch_size, 1, 1024)
        assert rewards.shape == (batch_size, 1, 1)
        assert dones.shape == (batch_size, 1, 1)

        assert env.states.shape == (batch_size, i, 1024)
        assert env.rewards.shape == (batch_size, i, 1)
        assert env.dones.shape == (batch_size, i, 1)
        assert env.actions.shape == (batch_size, i - 1, action_size)

    s, a, r, d = env.get_rollouts()
    assert s.shape == (batch_size, 23, 1024)
    assert a.shape == (batch_size, 22, action_size)
    assert r.shape == (batch_size, 23, 1)
    assert d.shape == (batch_size, 23, 1)

    # test that once done (done >= 0.5), all future done values are 1
    for t in d:
        t = t[:, 0]
        done = False
        for item in t:
            if not done and item > 0.5:
                done = True
            if done:
                assert item > 0.5
            else:
                assert item < 0.5