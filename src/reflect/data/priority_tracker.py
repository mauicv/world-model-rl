from typing import Optional

import torch


class PriorityTracker:
    def __init__(
            self,
            num_runs: int,
            rollout_length: int,
            ema_decay: float = 0.95,
            min_priority: float = 1e-4,
            max_priority: float = 100.0,
            epsilon: float = 1e-6,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32,
        ):
        self.num_runs = num_runs
        self.rollout_length = rollout_length
        self.ema_decay = ema_decay
        self.min_priority = min_priority
        self.max_priority = max_priority
        self.epsilon = epsilon
        self.device = device
        self.dtype = dtype

        # Current priorities used for sampling [run_index, time_index].
        self.priorities = torch.ones(
            self.num_runs,
            self.rollout_length,
            dtype=self.dtype,
            device=self.device,
        )
        # EMA loss estimate per sample for smoothing noisy updates.
        self.ema_loss = torch.zeros(
            self.num_runs,
            self.rollout_length,
            dtype=self.dtype,
            device=self.device,
        )
        # Tracks best/worst observed EMA to support normalization.
        self.best_ema_loss = torch.full(
            (self.num_runs, self.rollout_length),
            float("inf"),
            dtype=self.dtype,
            device=self.device,
        )
        self.worst_ema_loss = torch.zeros(
            self.num_runs,
            self.rollout_length,
            dtype=self.dtype,
            device=self.device,
        )
        # Marks samples that have received at least one update.
        self.has_observation = torch.zeros(
            self.num_runs,
            self.rollout_length,
            dtype=torch.bool,
            device=self.device,
        )

    def reset_run(self, run_index: int, end_index: Optional[int] = None):
        """Reset tracking state for a rollout slot before refilling it."""
        raise NotImplementedError

    def update_with_losses(self, b_inds, t_inds, per_sample_loss: torch.Tensor):
        """Update EMA + best/worst trackers from per-sample model errors."""
        raise NotImplementedError

    def normalized_loss(self, b_inds, t_inds):
        """Return loss normalized from best->worst (0 to 1) for selected indices."""
        raise NotImplementedError

    def refresh_priorities(self, b_inds=None, t_inds=None):
        """Recompute priorities from normalized losses and clamp to valid range."""
        raise NotImplementedError

    def run_priority_scores(self, run_indices=None):
        """Aggregate timestep priorities into per-run scores for batch sampling."""
        raise NotImplementedError

    def step_priority_distribution(self, run_index: int, temperature: float):
        """Build a timestep sampling distribution for one run."""
        raise NotImplementedError

