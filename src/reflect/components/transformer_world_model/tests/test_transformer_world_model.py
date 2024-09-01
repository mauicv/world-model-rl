from reflect.components.transformer_world_model import TransformerWorldModel
from reflect.components.transformer_world_model.embedder import Embedder as Embedder
from reflect.components.transformer_world_model.head import Head as Head
from reflect.components.actor import Actor
import torch


def test_save_load(tmp_path, transformer_world_model):
    transformer_world_model.save(tmp_path)
    transformer_world_model.load(tmp_path)


def test_world_model(transformer_world_model: TransformerWorldModel):
    o, a, r, d = (
        torch.randn((2, 10, 3, 64, 64)) * 255,
        torch.randn((2, 10, 1)),
        torch.randn((2, 10, 1)),
        torch.randn((2, 10, 1))
    )
    first_seq, next_seq = transformer_world_model.observe_rollout(o, a, r, d)
    assert first_seq.state_distribution.continuous_state.mean.shape == (2, 9, 16)
    assert first_seq.state_sample.features.shape == (2, 9, 64)
    assert next_seq.state_distribution.continuous_state.mean.shape == (2, 9, 16)
    assert next_seq.state_sample.features.shape == (2, 9, 64)


def test_world_model_update(transformer_world_model: TransformerWorldModel):
    o, a, r, d = (
        torch.randn((2, 10, 3, 64, 64)) * 255,
        torch.randn((2, 10, 1)),
        torch.randn((2, 10, 1)),
        torch.randn((2, 10, 1))
    )
    target, output = transformer_world_model.observe_rollout(o, a, r, d)
    losses = transformer_world_model.update(target=target, output=output, observations=o)
    assert losses.dynamic_model_loss > 0
    assert losses.reward_loss > 0
    assert losses.done_loss > 0
    assert losses.recon_loss > 0


def test_imagine_rollout(transformer_world_model: TransformerWorldModel, mixed_state_actor: Actor):
    observation, action, reward, done = (
        torch.randn((2, 10, 3, 64, 64)) * 255,
        torch.randn((2, 10, 1)),
        torch.randn((2, 10, 1)),
        torch.randn((2, 10, 1))
    )
    target, _ = transformer_world_model.observe_rollout(
        observation=observation,
        action=action,
        reward=reward,
        done=done
    )
    initial_state = target.to_initial_state()
    imagined_rollout = transformer_world_model.imagine_rollout(
        initial_state=initial_state,
        actor=mixed_state_actor,
        n_steps=10,
        with_observations=True
    )
    assert imagined_rollout.state_features.shape == (18, 11, 64)
    assert imagined_rollout.dist_features.shape == (18, 11, 80)
    assert imagined_rollout.reward.shape == (18, 11, 1)
    assert imagined_rollout.done.shape == (18, 11, 1)
    assert imagined_rollout.observations.shape == (18, 11, 3, 64, 64)
    assert imagined_rollout.action.shape == (18, 11, 1)
