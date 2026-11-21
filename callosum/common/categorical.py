"""Categorical actor-critic for the skill-level envs (task 3.1).

A `FaceTurnSkills-v0` policy chooses ONE discrete primitive per decision, so its actor is a
categorical head over `n_actions` logits instead of the diagonal Gaussian of
`_ppo_core.ActorCritic`. The torso and heads are deliberately identical to that class (3
hidden layers of 256, tanh, orthogonal init, critic out-std 1.0): only the distribution
changes. The actor's output layer keeps the small `0.01 * sqrt(2)` init, so the initial logits
are near zero -- a near-uniform initial policy, the discrete counterpart of the Gaussian's
initially small action std.

`get_value` / `get_action` / `get_action_and_value` mirror `_ppo_core.ActorCritic`, so the
IPPO rollout loop and `ppo_update` work with either class unchanged. Actions are primitive
indices, shape `(batch,)` and dtype long; log-probs and entropies are per-sample scalars (the
Gaussian sums over action dims, the categorical has none to sum). The DAPG/BC auxiliary loss
is not implemented for this head (`bc_mse` needs a Gaussian mean vector); the trainer rejects
those flags for discrete envs (skill-level lines need no BC by design).

Pure torch (no mani_skill): unit-tested on macOS/CI like `_ppo_core`.
"""

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

from callosum.common.ppo_core import _mlp


def _as_ids(action: torch.Tensor) -> torch.Tensor:
    """Primitive ids `(batch,)` from an action that may carry a trailing dim (`(batch, 1)`,
    how the rollout buffer stores actions)."""
    ids = action.reshape(action.shape[0], -1)
    if ids.shape[-1] != 1:
        raise ValueError(f"categorical actions hold one primitive id, got width {ids.shape[-1]}")
    return ids[:, 0]


class CategoricalPolicy(nn.Module):
    """One hand's actor-critic: a categorical policy over primitive ids and a state-value head.

    Args:
        obs_dim: width of the agent's policy input (`_skills_agent_obs.agent_obs_dim`).
        n_actions: size of the role's primitive registry (`_skills.num_primitives`).
    """

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.critic = _mlp(obs_dim, 1, out_std=1.0)
        self.actor_logits = _mlp(obs_dim, n_actions, out_std=0.01 * float(np.sqrt(2)))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(logits, value)`: primitive logits `(batch, n_actions)` and state values `(batch, 1)`."""
        return self.actor_logits(obs), self.critic(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """State values, shape `(batch, 1)`."""
        return self.critic(obs)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """A sampled primitive id `(batch,)`, or the logits' argmax when `deterministic` (eval)."""
        logits = self.actor_logits(obs)
        if deterministic:
            return logits.argmax(dim=-1)
        return Categorical(logits=logits).sample()

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(action, log_prob, entropy, value)`; samples an action unless one is given.

        Given actions may carry a trailing action dim (`(batch, 1)`, how the rollout buffer
        stores them); the ids are squeezed, so the returned log-probs are `(batch,)`.
        """
        logits = self.actor_logits(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        else:
            action = _as_ids(action)
        return action, dist.log_prob(action), dist.entropy(), self.critic(obs)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(action, log_prob, entropy)` of a fresh decision (argmax logits when deterministic)."""
        logits = self.actor_logits(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(log_prob, entropy, value)` of the taken `action` under the current policy.

        The action may carry a trailing action dim (`(batch, 1)`); see `get_action_and_value`.
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(_as_ids(action)), dist.entropy(), value
