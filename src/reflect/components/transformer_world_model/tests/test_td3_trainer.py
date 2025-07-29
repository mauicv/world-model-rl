import gymnasium as gym
from reflect.components.trainers.td3.td3_trainer import TD3Trainer
from reflect.components.trainers.td3.actor import TD3Actor
from reflect.components.trainers.td3.critic import TD3Critic
from reflect.components.transformer_world_model.tests.conftest import make_dynamic_model
from reflect.data.loader import EnvDataLoader, GymRenderImgProcessing
from reflect.components.transformer_world_model import WorldModel
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

    actor = TD3Actor(
        input_dim=32*32,
        output_dim=real_env.action_space.shape[0],
        # bound=real_env.action_space.high,
    )
    critic_1 = TD3Critic(
        state_dim=32*32,
        action_dim=real_env.action_space.shape[0],
    )
    critic_2 = TD3Critic(
        state_dim=32*32,
        action_dim=real_env.action_space.shape[0],
    )
    trainer = TD3Trainer(
        actor=actor,
        critics=[critic_1, critic_2],
        actor_lr=1e-5,
        critic_lr=1e-5,
        grad_clip=0.5,
        actor_update_frequency=2,
        tau=5e-3,
    )

    _, _, o, a, r, d = dl.sample(batch_size=4)
    _, (z, a, r, d) = wm.update(o, a, r, d, return_init_states=True)

    z, a, r, d = wm.imagine_rollout(
        z=z, a=a, r=r, d=d,
        actor=actor,
        with_observations=False,
        disable_gradients=True
    )

    history = trainer.update(
        state_samples=z,
        reward_samples=r,
        done_samples=d,
        action_samples=a
    )
    history_dict = asdict(history)
    for key in  [
            'actor_grad_norm',
            'value_grad_norms',
            'value_losses',
            'actor_loss',
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

    actor = TD3Actor(
        input_dim=32*32,
        output_dim=real_env.action_space.shape[0],
        noise_std=0.2,
    )
    critic_1 = TD3Critic(
        state_dim=32*32,
        action_dim=real_env.action_space.shape[0],
    )
    critic_2 = TD3Critic(
        state_dim=32*32,
        action_dim=real_env.action_space.shape[0],
    )
    trainer = TD3Trainer(
        actor=actor,
        critics=[critic_1, critic_2],
        actor_lr=1e-5,
        critic_lr=1e-5,
        grad_clip=0.5,
        actor_update_frequency=2,
        tau=5e-3,
        alpha=2.5,
        num_actor_updates=5,
    )

    dl.perform_rollout()
    _, _, o, a, r, d = dl.sample(batch_size=4)
    _, (z, a, r, d) = wm.update(o, a, r, d, return_init_states=True)

    z, a, r, d, _ = wm.imagine_rollout(
        z=z, a=a, r=r, d=d,
        actor=actor,
        with_observations=False,
        disable_gradients=True,
        with_entropies=True,
    )

    history = trainer.update(
        state_samples=z,
        reward_samples=r,
        done_samples=d,
        action_samples=a
    )
    history_dict = asdict(history)
    for key in  [
            'actor_grad_norm',
            'value_grad_norms',
            'value_losses',
            'actor_loss'
        ]:
        assert key in history_dict


def test_save_load(tmp_path):
    actor = TD3Actor(
        input_dim=32*32,
        output_dim=8,
        # bound=1.0,
    )
    critic_1 = TD3Critic(
        state_dim=32*32,
        action_dim=8,
    )
    critic_2 = TD3Critic(
        state_dim=32*32,
        action_dim=8,
    )
    trainer = TD3Trainer(
        actor=actor,
        critics=[critic_1, critic_2],
        actor_lr=1e-5,
        critic_lr=1e-5,
        grad_clip=0.5,
        actor_update_frequency=2,
        tau=5e-3,
    )
    trainer.save(tmp_path)
    trainer.load(tmp_path)