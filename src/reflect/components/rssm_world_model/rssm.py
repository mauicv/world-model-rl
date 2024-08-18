from typing import Tuple
import torch
from reflect.components.rssm_world_model.state.continuous import InternalStateContinuous, InternalStateContinuousSequence


class RSSM(torch.nn.Module):
    def __init__(
            self,
            hidden_size: int,
            deter_size: int,
            stoch_size: int,
            obs_embed_size: int,
            action_size: int,
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.act = torch.nn.ELU()
        self.rnn = torch.nn.GRUCell(deter_size, deter_size)
        self.fc_state_action_embed = torch.nn.Linear(
            stoch_size+action_size,
            deter_size
        )
        self.state_prior = torch.nn.Sequential(
            torch.nn.Linear(deter_size, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, 2 * stoch_size)
        )
        self.state_posterior = torch.nn.Sequential(
            torch.nn.Linear(deter_size+obs_embed_size, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, 2 * stoch_size)
        )

    def initial_state_sequence(self, batch_size):
        return InternalStateContinuousSequence.from_init(
            init_state=self.initial_state(batch_size)
        )

    def initial_state(self, batch_size):
        return InternalStateContinuous(
            deter_state=torch.zeros(batch_size, self.deter_size),
            stoch_state=torch.zeros(batch_size, self.stoch_size),
            mean=torch.zeros(batch_size, self.stoch_size),
            std=torch.zeros(batch_size, self.stoch_size)
        )

    def prior(self, action_emb: torch.Tensor, state: InternalStateContinuous):
        """Computes the prior distribution of the next state given the current
        state and action

        h_{t} = f(s_{t-1}, a_{t-1}, h_{t-1})
        p(s_{t} | h_{t})
        """
        state_action = torch.cat([state.stoch_state, action_emb], dim=-1)
        embedded_state_action = self.act(self.fc_state_action_embed(state_action))
        prior_deter_state = self.rnn(embedded_state_action, state.deter_state)
        stoch_mean_std = self.state_prior(prior_deter_state)
        mean, std = torch.split(stoch_mean_std, self.stoch_size, dim=-1)
        std = torch.nn.functional.softplus(std) + 0.1
        stoch_state = mean + std * torch.randn_like(mean)
        return InternalStateContinuous(
            deter_state=prior_deter_state,
            stoch_state=stoch_state,
            mean=mean,
            std=std
        )

    def posterior(self, obs_embed: torch.Tensor, state: InternalStateContinuous):
        """Computes the posterior distribution given the current hidden state
        and the embedding observation.

        p(s_{t} | h_{t}, o_{t})
        """
        hidden = torch.cat([state.deter_state, obs_embed], dim=-1)
        posterior_mean_std = self.state_posterior(hidden)
        mean, std = torch.split(posterior_mean_std, self.stoch_size, dim=-1)
        std = torch.nn.functional.softplus(std) + 0.1
        stoch_state = mean + std * torch.randn_like(mean)
        return InternalStateContinuous(
            deter_state=state.deter_state,
            stoch_state=stoch_state,
            mean=mean,
            std=std
        )

    def observe_step(self, obs_embed, action_emb, state: InternalStateContinuous):
        prior_state = self.prior(action_emb, state)
        posterior_state = self.posterior(obs_embed, prior_state)
        return prior_state, posterior_state

    def observe_rollout(
            self,
            obs_embeds,
            action_embs,
        ) -> Tuple[InternalStateContinuousSequence, InternalStateContinuousSequence]:
        batch, n_steps, *_ = obs_embeds.shape
        prior_state_sequence = self.initial_state_sequence(batch)
        prior_state_sequence.to(obs_embeds.device)
        posterior_state_sequence = self.initial_state_sequence(batch)
        posterior_state_sequence.to(obs_embeds.device)
        state = posterior_state_sequence.get_last()
        for t in range(n_steps - 1):
            # The observe step computes the t+1 prior from the t-th
            # state. We then use this and the t+1-th observation to
            # compute the posterior. 
            obs_embed = obs_embeds[:, t+1]
            action_emb = action_embs[:, t]
            prior, posterior = self.observe_step(
                obs_embed,
                action_emb,
                state
            )
            prior_state_sequence.append_(prior)
            posterior_state_sequence.append_(posterior)
            state = posterior
        return prior_state_sequence, posterior_state_sequence

    def imagine_rollout(
            self,
            initial_states: InternalStateContinuous,
            actor: torch.nn.Module,
            n_steps: int,
        ) -> InternalStateContinuousSequence:
        prior_state_sequence = InternalStateContinuousSequence.from_init(initial_states)
        device = next(actor.parameters()).device
        prior_state_sequence.to(device)
        state = prior_state_sequence.get_last()
        for i in range(n_steps):
            # TODO: do we detach action input here?
            action_input = state.get_features().detach()
            action_emb = actor(action_input, deterministic=True)
            prior = self.prior(action_emb, state)
            prior_state_sequence.append_(prior)
            state = prior
        return prior_state_sequence
