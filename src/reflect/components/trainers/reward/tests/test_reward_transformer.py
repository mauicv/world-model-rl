from reflect.data.loader import EnvDataLoader
from reflect.components.transformer_world_model.world_model import TransformerWorldModel
from reflect.components.actor import Actor
from reflect.components.trainers.reward.reward_trainer import RewardGradTrainer


def test_world_model_reward_trainer(
        env_data_loader: EnvDataLoader,
        transformer_world_model: TransformerWorldModel,
        actor: Actor,
        reward_grad_trainer: RewardGradTrainer
    ):
    env_data_loader.perform_rollout()
    o, a, r, d = env_data_loader.sample(batch_size=32)

    target, output \
        = transformer_world_model.observe_rollout(o, a)

    losses = transformer_world_model.update(
        target=target,
        output=output,
        observations=o,
        global_step=2301,
        reward=r,
        done=d
    )
    assert losses

    initial_state = target.to_initial_state().detach()
    rollout, rewards, dones = transformer_world_model.imagine_rollout(
        initial_state=initial_state,
        actor=actor,
        n_steps=10
    )

    assert rewards.shape == (288, 11, 1)

    losses = reward_grad_trainer.update(
        reward_samples=rewards,
        done_samples=dones,
    )

    assert losses.actor_loss
