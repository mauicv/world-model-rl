from typing import Tuple
import torch
from dataclasses import dataclass
import torch.distributions as D


@dataclass
class InternalState:
    deter_state: torch.Tensor
    stoch_state: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def detach(self) -> 'InternalState':
        return InternalState(
            deter_state=self.deter_state.detach(),
            stoch_state=self.stoch_state.detach(),
            mean=self.mean.detach(),
            std=self.std.detach()
        )

    @property
    def shapes(self) -> Tuple[int, int]:
        return self.deter_state.shape, self.stoch_state.shape
    
    def get_features(self):
        return torch.cat([
            self.deter_state,
            self.stoch_state
        ], dim=-1)

    def to(self, device: torch.device):
        self.deter_state=self.deter_state.to(device)
        self.stoch_state=self.stoch_state.to(device)
        self.mean=self.mean.to(device)
        self.std=self.std.to(device)


@dataclass
class InternalStateSequence:
    deter_states: torch.Tensor
    stoch_states: torch.Tensor
    means: torch.Tensor
    stds: torch.Tensor

    @classmethod
    def from_init(cls, init_state: InternalState):
        return cls(
            deter_states=init_state.deter_state.unsqueeze(1),
            stoch_states=init_state.stoch_state.unsqueeze(1),
            means=init_state.mean.unsqueeze(1),
            stds=init_state.std.unsqueeze(1)
        )

    @property
    def shapes(self) -> Tuple[int, int]:
        return self.deter_states.shape, self.stoch_states.shape

    def append_(self, other: InternalState):
        self.deter_states = torch.cat([self.deter_states, other.deter_state.unsqueeze(1)], dim=1)
        self.stoch_states = torch.cat([self.stoch_states, other.stoch_state.unsqueeze(1)], dim=1)
        self.means = torch.cat([self.means, other.mean.unsqueeze(1)], dim=1)
        self.stds = torch.cat([self.stds, other.std.unsqueeze(1)], dim=1)

    def __getitem__(self, index):
        return InternalState(
            deter_state=self.deter_states[:, index],
            stoch_state=self.stoch_states[:, index],
            mean=self.means[:, index],
            std=self.stds[:, index]
        )

    def get_last(self):
        return self[-1]

    def get_features(self):
        return torch.cat([
            self.deter_states[:, 1:],
            self.stoch_states[:, 1:]
        ], dim=-1)

    def get_dist(self):
        normal = D.Normal(
            self.means[:, 1:],
            self.stds[:, 1:]
        )
        return D.Independent(normal, 1)

    def flatten_batch_time(self) -> InternalState:
        deter_states = self.deter_states[:, 1:]
        stoch_states = self.stoch_states[:, 1:]
        means = self.means[:, 1:]
        stds = self.stds[:, 1:]
        return InternalState(
            deter_state=deter_states.reshape(-1, *deter_states.shape[2:]),
            stoch_state=stoch_states.reshape(-1, *stoch_states.shape[2:]),
            mean=means.reshape(-1, *means.shape[2:]),
            std=stds.reshape(-1, *stds.shape[2:])
        )

    def to(self, device):
        self.deter_states = self.deter_states.to(device)
        self.stoch_states = self.stoch_states.to(device)
        self.means = self.means.to(device)
        self.stds = self.stds.to(device)


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
        self.act = torch.nn.ReLU()
        self.rnn = torch.nn.GRUCell(deter_size, deter_size)
        self.fc_state_action_embed = torch.nn.Linear(
            stoch_size+action_size,
            deter_size
        )
        self.state_prior = torch.nn.Sequential(
            torch.nn.Linear(deter_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 2 * stoch_size)
        )
        self.state_posterior = torch.nn.Sequential(
            torch.nn.Linear(deter_size+obs_embed_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 2 * stoch_size)
        )

    def initial_state_sequence(self, batch_size):
        return InternalStateSequence.from_init(
            init_state=self.initial_state(batch_size)
        )

    def initial_state(self, batch_size):
        return InternalState(
            deter_state=torch.zeros(batch_size, self.deter_size),
            stoch_state=torch.zeros(batch_size, self.stoch_size),
            mean=torch.zeros(batch_size, self.stoch_size),
            std=torch.zeros(batch_size, self.stoch_size)
        )

    def prior(self, action_emb: torch.Tensor, state: InternalState):
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
        std = torch.nn.functional.softplus(std) + 1e-5
        stoch_state = mean + std * torch.randn_like(mean)
        return InternalState(
            deter_state=prior_deter_state,
            stoch_state=stoch_state,
            mean=mean,
            std=std
        )

    def posterior(self, obs_embed: torch.Tensor, state: InternalState):
        """Computes the posterior distribution given the current hidden state
        and the embedding observation.

        p(s_{t} | h_{t}, o_{t})
        """
        hidden = torch.cat([state.deter_state, obs_embed], dim=-1)
        posterior_mean_std = self.state_posterior(hidden)
        mean, std = torch.split(posterior_mean_std, self.stoch_size, dim=-1)
        std = torch.nn.functional.softplus(std) + 1e-5
        stoch_state = mean + std * torch.randn_like(mean)
        return InternalState(
            deter_state=state.deter_state,
            stoch_state=stoch_state,
            mean=mean,
            std=std
        )

    def observe_step(self, obs_embed, action_emb, state: InternalState):
        prior_state = self.prior(action_emb, state)
        posterior_state = self.posterior(obs_embed, prior_state)
        return prior_state, posterior_state

    def observe_rollout(
            self,
            obs_embeds,
            action_embs,
        ) -> Tuple[InternalStateSequence, InternalStateSequence]:
        batch, n_steps, *_ = obs_embeds.shape
        prior_state_sequence = self.initial_state_sequence(batch)
        prior_state_sequence.to(obs_embeds.device)
        posterior_state_sequence = self.initial_state_sequence(batch)
        posterior_state_sequence.to(obs_embeds.device)
        state = posterior_state_sequence.get_last()
        for i in range(n_steps):
            obs_embed = obs_embeds[:, i]
            action_emb = action_embs[:, i]
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
            initial_states: InternalState,
            actor: torch.nn.Module,
            n_steps: int,
            obs_embed: torch.Tensor
        ) -> InternalStateSequence:
        prior_state_sequence = InternalStateSequence.from_init(initial_states)
        prior_state_sequence.to(obs_embed.device)
        state = prior_state_sequence.get_last()
        for i in range(n_steps):
            action_input = torch.cat([
                    state.stoch_state,
                    state.deter_state
            ], dim=-1)
            # TODO: do we detach action input here?
            action_emb = actor(action_input.detach(), deterministic=True)
            prior, _ = self.observe_step(obs_embed, action_emb, state)
            prior_state_sequence.append_(prior)
            state = prior
        return prior_state_sequence
