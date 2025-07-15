
import torch
from reflect.components.trainers.td3.tests.reference_fn import compute_TD_target


def test_target_computation(env, replay_buffer, trainer):
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

    targets_1 = compute_TD_target(nz, r, d, trainer.target_actor, trainer.target_critics[0])
    targets_2 = trainer.compute_TD_target(nz, r, d, trainer.target_critics[0])
    print('targets_1', targets_1)
    print('targets_2', targets_2)
    assert torch.allclose(targets_1, targets_2)


    
    