import torch
from reflect.data.loader import EnvDataLoader
from reflect.models.rssm_world_model.world_model import WorldModel
from reflect.models.agent.actor import Actor
from reflect.models.agent.reward_trainer import RewardGradTrainer


def test_world_model_update(
        env_data_loader: EnvDataLoader,
        world_model: WorldModel,
    ):
    env_data_loader.perform_rollout()
    o, a, r, d = env_data_loader.sample(batch_size=32)
    _, prior_sequence, posterior_sequence \
        = world_model.observe_rollout(o, a)

    losses = world_model.update(
        prior_state_sequence=prior_sequence,
        posterior_state_sequence=posterior_sequence,
        obs=o,
        reward=r,
        done=d
    )
    assert losses


def test_world_model_imagine_rollout(
        env_data_loader: EnvDataLoader,
        world_model: WorldModel,
        actor: Actor
    ):
    env_data_loader.perform_rollout()
    o, a, *_ = env_data_loader.sample(batch_size=32)
    o_emb, _, posterior_sequence \
        = world_model.observe_rollout(o, a)

    initial_states = posterior_sequence.flatten_batch_time().detach()
    o_emb = o_emb.reshape(-1, *o_emb.shape[2:])
    rollout = world_model.imagine_rollout(
        initial_states=initial_states,
        actor=actor,
        o_emb=o_emb,
        n_steps=10
    )

    assert rollout.rewards.shape == (320, 10, 1)


def test_world_model_train_step(
        env_data_loader: EnvDataLoader,
        world_model: WorldModel,
        actor: Actor,
        reward_grad_trainer: RewardGradTrainer
    ):
    env_data_loader.perform_rollout()
    o, a, r, d = env_data_loader.sample(batch_size=32)
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
    o_emb = o_emb.reshape(-1, *o_emb.shape[2:])
    rollout = world_model.imagine_rollout(
        initial_states=initial_states,
        actor=actor,
        o_emb=o_emb,
        n_steps=10
    )

    assert rollout.rewards.shape == (320, 10, 1)

    losses = reward_grad_trainer.update(
        reward_samples=rollout.rewards,
        done_samples=rollout.dones,
    )

    assert losses.actor_loss