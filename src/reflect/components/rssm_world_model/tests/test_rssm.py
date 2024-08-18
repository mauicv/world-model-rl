import torch
from reflect.components.rssm_world_model.rssm import (
    ContinuousRSSM,
    InternalStateContinuous,
    DiscreteRSSM,
    InternalStateDiscrete
)
from reflect.components.observation_model import (
    ConvEncoder,
    ConvDecoder
)
from reflect.components.actor import Actor


def test_continuous_rssm_observe_rollout(
        encoder: ConvEncoder,
        decoder: ConvDecoder,
        continuous_rssm: ContinuousRSSM
    ):

    action_batch = torch.randn(32, 10, 1)
    obs_batch = torch.randn(32, 10, 3, 64, 64)
    obs_embed = encoder(obs_batch)

    prior_state_sequence, posterior_state_sequence = continuous_rssm.observe_rollout(
        obs_embed,
        action_batch,
    )

    assert prior_state_sequence.stoch_states.shape == (32, 10, 30)
    assert posterior_state_sequence.stoch_states.shape == (32, 10, 30)

    features = posterior_state_sequence.get_features()
    decoder_output = decoder(features)
    assert decoder_output.shape == (32, 9, 3, 64, 64)


def test_continuous_rssm_imagine_rollout(
        actor: Actor,
        continuous_rssm: ContinuousRSSM
    ):

    initial_states = InternalStateContinuous(
        deter_state=torch.randn(32, 200),
        stoch_state=torch.randn(32, 30),
        mean=torch.randn(32, 30),
        std=torch.ones(32, 30),
    )

    imagined_state_sequence = continuous_rssm.imagine_rollout(
        initial_states=initial_states,
        actor=actor,
        n_steps=10
    )

    assert imagined_state_sequence.stoch_states.shape == (32, 11, 30)


def test_discrete_rssm_observe_rollout(
        encoder: ConvEncoder,
        discrete_decoder: ConvDecoder,
        discrete_rssm: DiscreteRSSM
    ):

    action_batch = torch.randn(32, 10, 1)
    obs_batch = torch.randn(32, 10, 3, 64, 64)
    obs_embed = encoder(obs_batch)

    prior_state_sequence, posterior_state_sequence = discrete_rssm.observe_rollout(
        obs_embed,
        action_batch,
    )

    assert prior_state_sequence.stoch_states.shape == (32, 10, 1024)
    assert posterior_state_sequence.stoch_states.shape == (32, 10, 1024)

    features = posterior_state_sequence.get_features()
    decoder_output = discrete_decoder(features)
    assert decoder_output.shape == (32, 9, 3, 64, 64)


def test_discrete_rssm_imagine_rollout(
        discrete_actor: Actor,
        discrete_rssm: DiscreteRSSM
    ):

    initial_states = InternalStateDiscrete(
        deter_state=torch.randn(32, 200),
        stoch_state=torch.randn(32, 32*32),
        logits=torch.randn(32, 32, 32),
    )

    imagined_state_sequence = discrete_rssm.imagine_rollout(
        initial_states=initial_states,
        actor=discrete_actor,
        n_steps=10
    )

    assert imagined_state_sequence.stoch_states.shape == (32, 11, 1024)