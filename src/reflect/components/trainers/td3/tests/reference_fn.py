import torch

GAMMA=0.98


def compute_action(actor, state, eps=0):
    actor.eval()
    state = torch.tensor(state, dtype=torch.float)
    if len(state.shape) == 1:
        state=state[None, :]
    action = actor(state)
    action = torch.clip(action + torch.randn_like(action) * eps, -1, 1)
    actor.train()
    return action.squeeze().detach()


def compute_TD_target(
        next_states,
        rewards,
        dones,
        target_actor,
        target_critic
    ):
    dones = torch.tensor(dones)
    next_states = torch.tensor(next_states)
    next_state_actions = compute_action(
        target_actor,
        next_states,
        eps=0
    )
    next_state_action_values = target_critic(
        next_states,
        next_state_actions
    )
    rewards = torch.tensor(rewards)
    targets = rewards + GAMMA * (1 - dones) * next_state_action_values[:, 0]
    return targets.detach()

def compute_Q_loss(
        critic_1,
        target_critic_1,
        critic_2,
        target_critic_2,
        actor,
        target_actor,
        current_states,
        next_states,
        current_actions,
        rewards,
        dones
    ):
    targets_1 = compute_TD_target(
        next_states,
        rewards,
        dones,
        target_actor,
        target_critic_1
    )

    targets_2 = compute_TD_target(
        next_states,
        rewards,
        dones,
        target_actor,
        target_critic_2
    )

    cat_targets = torch.cat(
        (
            targets_1[:, None],
            targets_2[:, None]
        ),
        dim=-1
    )
    targets = torch.min(
        cat_targets,
        dim=-1
    ).values

    current_states = torch.tensor(current_states)
    current_actions = torch.tensor(current_actions)
    current_state_action_values_1 = critic_1(
        current_states,
        current_actions
    )
    current_state_action_values_2 = critic_2(
        current_states,
        current_actions
    )

    loss_1 = (targets.detach() - current_state_action_values_1[:, 0])**2
    loss_1 = loss_1.mean()

    loss_2 = (targets.detach() - current_state_action_values_2[:, 0])**2
    loss_2 = loss_2.mean()

    return loss_1, loss_2


def compute_actor_loss(actor, critic, states):
    # states = torch.tensor(states)
    actions = actor(states)
    action_values = - critic(states, actions)
    return action_values.mean()