import torch
from reflect.components.trainers.td3.tests.reference_fn import compute_Q_loss
import copy
from reflect.components.trainers.td3.critic import TD3Critic

def test_critic_loss_computation(env, replay_buffer, trainer):

    current_state, *_ = env.reset(seed=1)
    action = env.action_space.sample()

    for _ in range(100):
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.push((current_state, next_state, action, reward, done))
        if done: break
        current_state = next_state
        action = env.action_space.sample()

    z, nz, a, r, d = replay_buffer.sample()

    # TODO: TEST clone?
    z = torch.tensor(z)
    nz = torch.tensor(nz)
    a = torch.tensor(a)
    r = torch.tensor(r)
    d = torch.tensor(d)

    loss_1, loss_2 = compute_Q_loss(
        trainer.critics[0],
        trainer.target_critics[0],
        trainer.critics[1],
        trainer.target_critics[1],
        trainer.actor,
        trainer.target_actor,
        z, nz, a, r, d
    )

    losses, _ = trainer.update_critics(
        z, nz, a, r, d
    )
    assert torch.allclose(loss_1, torch.tensor(losses[0]))
    assert torch.allclose(loss_2, torch.tensor(losses[1]))


def prep_test_critics(env, trainer):
    critic_1_state_dict = trainer.critics[0].state_dict()
    critic_2_state_dict = trainer.critics[1].state_dict()
    target_critic_1_state_dict = trainer.target_critics[0].state_dict()
    target_critic_2_state_dict = trainer.target_critics[1].state_dict()
    critic_1 = TD3Critic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )
    critic_2 = TD3Critic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )
    target_critic_1 = TD3Critic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )
    target_critic_2 = TD3Critic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )
    critic_1.load_state_dict(copy.deepcopy(critic_1_state_dict))
    critic_2.load_state_dict(copy.deepcopy(critic_2_state_dict))
    target_critic_1.load_state_dict(copy.deepcopy(target_critic_1_state_dict))
    target_critic_2.load_state_dict(copy.deepcopy(target_critic_2_state_dict))
    return critic_1, critic_2, target_critic_1, target_critic_2



def test_critic_loss_gn(env, replay_buffer, trainer):
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

    # TODO: TEST clone?
    z = torch.tensor(z)
    nz = torch.tensor(nz)
    a = torch.tensor(a)
    r = torch.tensor(r)
    d = torch.tensor(d)

    critic_1, critic_2, target_critic_1, target_critic_2 = prep_test_critics(env, trainer)

    loss_1, loss_2 = compute_Q_loss(
        critic_1,
        target_critic_1,
        critic_2,
        target_critic_2,
        trainer.actor,
        trainer.target_actor,
        z, nz, a, r, d
    )
    loss_1.backward(retain_graph=True)
    gn1 = torch.nn.utils.clip_grad_norm_(critic_1.parameters(), torch.inf)
    loss_2.backward(retain_graph=True)
    gn2 = torch.nn.utils.clip_grad_norm_(critic_2.parameters(), torch.inf)

    losses, gn = trainer.update_critics(
        z, nz, a, r, d
    )

    assert torch.allclose(gn1, torch.tensor(gn[0]))
    assert torch.allclose(gn2, torch.tensor(gn[1]))
    assert torch.allclose(loss_1, torch.tensor(losses[0]))
    assert torch.allclose(loss_2, torch.tensor(losses[1]))
