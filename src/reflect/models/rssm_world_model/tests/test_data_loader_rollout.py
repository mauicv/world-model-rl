import torch
from reflect.models.rssm_world_model.memory_actor import WorldModelActor
from reflect.data.loader import EnvDataLoader

def test_actor_with_world_model(world_model_actor: WorldModelActor):
    world_model_actor.reset()
    obs = torch.zeros(1, 1, 3, 64, 64)
    action = world_model_actor(obs)
    assert action.shape == (1, 1)

def test_actor_with_world_model_nv_data_loader(env_data_loader: EnvDataLoader):
    env_data_loader.perform_rollout()
    i, a, r, d = env_data_loader.sample(num_time_steps=10)
    assert i.shape == (64, 10, 3, 64, 64)
    assert a.shape == (64, 10, 1)
    assert r.shape == (64, 10, 1)
    assert d.shape == (64, 10, 1)
