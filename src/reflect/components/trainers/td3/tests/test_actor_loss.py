
import torch
from reflect.components.trainers.td3.tests.reference_fn import compute_actor_loss
from reflect.components.trainers.td3.actor import TD3Actor
import copy


def test_actor_loss_computation(env, replay_buffer, trainer):
    current_state, *_ = env.reset(seed=1)
    action = env.action_space.sample()

    for _ in range(100):
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.push((current_state, next_state, action, reward, done))
        if done: break
        current_state = next_state
        action = env.action_space.sample()

    z, nz, a, r, d = replay_buffer.sample()

    z = torch.tensor(z)
    nz = torch.tensor(nz)
    a = torch.tensor(a)
    r = torch.tensor(r)
    d = torch.tensor(d)

    loss_2 = compute_actor_loss(trainer.actor, trainer.critics[0], z)
    print('loss_2', loss_2.item())
    loss_1, _ = trainer.update_actor(z)
    print('loss_1', loss_1)

    assert torch.allclose(loss_2, torch.tensor(loss_1))
    

def prep_test_actor(env, trainer):
    actor_state_dict = trainer.actor.state_dict()
    target_actor_state_dict = trainer.target_actor.state_dict()
    actor = TD3Actor(
        input_dim=env.observation_space.shape[0],
        output_dim=env.action_space.shape[0],
    )
    target_actor = TD3Actor(
        input_dim=env.observation_space.shape[0],
        output_dim=env.action_space.shape[0],
    )
    actor.load_state_dict(copy.deepcopy(actor_state_dict))
    target_actor.load_state_dict(copy.deepcopy(target_actor_state_dict))
    return actor, target_actor


def test_actor_loss_gn(env, replay_buffer, trainer):
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    current_state, *_ = env.reset(seed=1)
    action = env.action_space.sample()

    for _ in range(100):
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.push((current_state, next_state, action, reward, done))
        if done: break
        current_state = next_state
        action = env.action_space.sample()

    z, nz, a, r, d = replay_buffer.sample()

    z = torch.tensor(z)
    nz = torch.tensor(nz)
    a = torch.tensor(a)
    r = torch.tensor(r)
    d = torch.tensor(d)

    nactor, target_actor = prep_test_actor(env, trainer)
    loss_2 = compute_actor_loss(nactor, trainer.critics[0], z)
    loss_2.backward(retain_graph=True)
    gn2 = torch.nn.utils.clip_grad_norm_(nactor.parameters(), torch.inf)
    print('loss_2', loss_2.item(), gn2.item())
    
    loss_1, actor_gn = trainer.update_actor(z)
    print('loss_1', loss_1, actor_gn)
    assert torch.allclose(loss_2, torch.tensor(loss_1))
    assert torch.allclose(gn2, torch.tensor(actor_gn))