import gymnasium as gym
from reflect.components.trainers.value.value_trainer import ValueGradTrainer
from reflect.components.models.actor import Actor
from reflect.components.trainers.value.critic import ValueCritic
from reflect.components.transformer_world_model.tests.conftest import make_dynamic_model
from reflect.data.loader import EnvDataLoader, GymRenderImgProcessing
from reflect.components.transformer_world_model import WorldModel
from reflect.components.transformer_world_model.environment import Environment
from torchvision.transforms import Resize, Compose
from dataclasses import asdict


def test_update(encoder, decoder, actor):
    real_env = gym.make("Ant-v4", render_mode="rgb_array")
    action_size = real_env.action_space.shape[0]

    dm = make_dynamic_model(a_size=action_size)

    wm = WorldModel(
        encoder=encoder,
        decoder=decoder,
        dynamic_model=dm,
    )

    dl = EnvDataLoader(
        num_time_steps=17,
        state_shape=(3, 64, 64),
        processing=GymRenderImgProcessing(
            transforms=Compose([
                Resize((64, 64))
            ])
        ),
        env=real_env
    )

    dl.perform_rollout()

    actor = Actor(
        input_dim=32*32,
        output_dim=real_env.action_space.shape[0],
        bound=real_env.action_space.high,
    )
    critic = ValueCritic(
        state_dim=32*32,
    )
    trainer = ValueGradTrainer(
        actor=actor,
        critic=critic,
    )

    o, a, r, d = dl.sample(batch_size=2)
    _, (z, a, r, d) = wm.update(o, a, r, d, return_init_states=True)

    z, a, r, d = wm.imagine_rollout(
        z=z, a=a, r=r, d=d,
        actor=actor,
        with_observations=False
    )

    history = trainer.update(
        state_samples=z,
        reward_samples=r,
        done_samples=d
    )
    history_dict = asdict(history)
    for key in  [
            'actor_grad_norm',
            'value_grad_norm',
            'value_loss',
            'actor_loss'
        ]:
        assert key in history_dict


def test_update_state(state_encoder, state_decoder, actor):
    real_env = gym.make("Ant-v4", render_mode="rgb_array")
    action_size = real_env.action_space.shape[0]

    dm = make_dynamic_model(a_size=action_size)

    wm = WorldModel(
        encoder=state_encoder,
        decoder=state_decoder,
        dynamic_model=dm,
    )

    dl = EnvDataLoader(
        num_time_steps=17,
        state_shape=(27,),
        env=real_env,
        use_imgs_as_states=False,
    )

    dl.perform_rollout()

    actor = Actor(
        input_dim=32*32,
        output_dim=real_env.action_space.shape[0],
        bound=real_env.action_space.high,
    )
    critic = ValueCritic(
        state_dim=32*32,
    )
    trainer = ValueGradTrainer(
        actor=actor,
        critic=critic,
    )

    o, a, r, d = dl.sample(batch_size=2)
    _, (z, a, r, d) = wm.update(o, a, r, d, return_init_states=True)

    z, a, r, d = wm.imagine_rollout(
        z=z, a=a, r=r, d=d,
        actor=actor,
        with_observations=False
    )

    history = trainer.update(
        state_samples=z,
        reward_samples=r,
        done_samples=d
    )
    history_dict = asdict(history)
    for key in  [
            'actor_grad_norm',
            'value_grad_norm',
            'value_loss',
            'actor_loss'
        ]:
        assert key in history_dict

