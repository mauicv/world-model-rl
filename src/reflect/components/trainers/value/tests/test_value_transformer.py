from reflect.data.loader import EnvDataLoader
from reflect.components.transformer_world_model.world_model import TransformerWorldModel
from reflect.components.actor import Actor
from reflect.components.trainers.value.value_trainer import ValueGradTrainer


def test_world_model_reward_trainer(
        env_data_loader: EnvDataLoader,
        transformer_world_model: TransformerWorldModel,
        actor: Actor,
        value_grad_trainer: ValueGradTrainer
    ):
    env_data_loader.perform_rollout()
    o, a, r, d = env_data_loader.sample(batch_size=32)

    target, output \
        = transformer_world_model.observe_rollout(o, a, r, d)

    losses = transformer_world_model.update(
        target=target,
        output=output,
        observations=o,
    )
    assert losses

    initial_state = target.to_initial_state().detach()
    rollout = transformer_world_model.imagine_rollout(
        initial_state=initial_state,
        actor=actor,
        n_steps=10
    )

    assert rollout.reward.shape == (288, 11, 1)

    losses = value_grad_trainer.update(
        state_samples=rollout.state,
        reward_samples=rollout.reward,
        done_samples=rollout.done,
    )

    assert losses.actor_loss
