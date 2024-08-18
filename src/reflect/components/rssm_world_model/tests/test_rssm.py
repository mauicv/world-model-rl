import torch
from reflect.components.rssm_world_model.rssm import RSSM, InternalStateContinuous
from reflect.components.observation_model import ConvEncoder, ConvDecoder
from reflect.components.actor import Actor


def test_rssm_observe_rollout(
        encoder: ConvEncoder,
        decoder: ConvDecoder,
        rssm: RSSM
    ):

    action_batch = torch.randn(32, 10, 1)
    obs_batch = torch.randn(32, 10, 3, 64, 64)
    obs_embed = encoder(obs_batch)

    prior_state_sequence, posterior_state_sequence = rssm.observe_rollout(
        obs_embed,
        action_batch,
    )

    assert prior_state_sequence.stoch_states.shape == (32, 10, 30)
    assert posterior_state_sequence.stoch_states.shape == (32, 10, 30)

    z = torch.cat([posterior_state_sequence.deter_states, posterior_state_sequence.stoch_states], dim=-1)
    decoder_output = decoder(z)
    assert decoder_output.shape == (32, 10, 3, 64, 64)


def test_rssm_imagine_rollout(
        encoder: ConvEncoder,
        actor: Actor,
        rssm: RSSM
    ):

    obs_batch = torch.randn(32, 1, 3, 64, 64)
    obs_embed = encoder(obs_batch)
    initial_states = InternalStateContinuous(
        deter_state=torch.randn(32, 200),
        stoch_state=torch.randn(32, 30),
        mean=torch.randn(32, 30),
        std=torch.ones(32, 30),
    )

    imagined_state_sequence = rssm.imagine_rollout(
        initial_states=initial_states,
        actor=actor,
        n_steps=10
    )

    assert imagined_state_sequence.stoch_states.shape == (32, 11, 30)