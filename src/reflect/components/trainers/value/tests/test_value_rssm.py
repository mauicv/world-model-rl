from reflect.data.loader import EnvDataLoader
from reflect.components.rssm_world_model.world_model import WorldModel
from reflect.components.models.actor import Actor
from reflect.components.trainers.value.value_trainer import ValueGradTrainer


def test_world_model_value_trainer(
        env_data_loader: EnvDataLoader,
        world_model: WorldModel,
        actor: Actor,
        value_grad_trainer: ValueGradTrainer
    ):
    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=32)
    o_emb, prior_sequence, posterior_sequence \
        = world_model.observe_rollout(o, a)

    losses = world_model.update(
        prior_state_sequence=prior_sequence,
        posterior_state_sequence=posterior_sequence,
        obs=o,
        reward=r,
        done=d
    )
    assert losses

    initial_states = posterior_sequence.flatten_batch_time().detach()
    o_emb = o_emb[:, 1:].reshape(-1, *o_emb.shape[2:])
    rollout = world_model.imagine_rollout(
        initial_states=initial_states,
        actor=actor,
        n_steps=10
    )

    assert rollout.rewards.shape == (288, 10, 1)

    losses = value_grad_trainer.update(
        state_samples=rollout.features,
        reward_samples=rollout.rewards,
        done_samples=rollout.dones,
    )

    assert losses.actor_loss
    assert losses.actor_grad_norm
    assert losses.value_loss
    assert losses.value_grad_norm

def test_world_model_value_trainer_critic_only(
        env_data_loader: EnvDataLoader,
        world_model: WorldModel,
        actor: Actor,
        value_grad_trainer: ValueGradTrainer
    ):
    env_data_loader.perform_rollout()
    _, _, o, a, r, d = env_data_loader.sample(batch_size=32)
    o_emb, prior_sequence, posterior_sequence \
        = world_model.observe_rollout(o, a)

    losses = world_model.update(
        prior_state_sequence=prior_sequence,
        posterior_state_sequence=posterior_sequence,
        obs=o,
        reward=r,
        done=d
    )
    assert losses

    initial_states = posterior_sequence.flatten_batch_time().detach()
    o_emb = o_emb[:, 1:].reshape(-1, *o_emb.shape[2:])
    rollout = world_model.imagine_rollout(
        initial_states=initial_states,
        actor=actor,
        n_steps=10
    )

    assert rollout.rewards.shape == (288, 10, 1)

    losses = value_grad_trainer.update(
        state_samples=rollout.features,
        reward_samples=rollout.rewards,
        done_samples=rollout.dones,
        critic_only=True
    )

    assert losses.actor_loss is None
    assert losses.actor_grad_norm is None
    assert losses.value_loss
    assert losses.value_grad_norm is not None

