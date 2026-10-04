"""Pure-torch PPO building blocks for the IPPO trainer: actor-critic, GAE, clipped update.

The network, the GAE recursion (with the `final_values` bootstrap across partial resets) and the
clipped-surrogate update are those of ManiSkill's `examples/baselines/ppo/ppo.py` @ v3.0.1.
They live apart from `callosum.training.ippo` because that module needs ManiSkill (not
importable on macOS/CI) while this one is unit-testable everywhere.
"""

import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal

from callosum.configs.ippo import IPPOConfig


def layer_init(layer: nn.Linear, std: float = float(np.sqrt(2)), bias_const: float = 0.0):
    """Orthogonal weight / constant bias initialisation (as in ManiSkill's PPO)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _mlp(in_dim: int, out_dim: int, out_std: float) -> nn.Sequential:
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, out_dim), std=out_std),
    )


class ActorCritic(nn.Module):
    """One arm's actor-critic: a diagonal Gaussian policy and a state-value head.

    The two MLPs (3 hidden layers of 256, tanh) and the state-independent log-std (initially
    -0.5) match ManiSkill's `Agent`. Actions are sampled unclipped and clipped to the action
    bounds by the caller, so log-probs refer to the unclipped sample.
    """

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.critic = _mlp(obs_dim, 1, out_std=1.0)
        self.actor_mean = _mlp(obs_dim, action_dim, out_std=0.01 * float(np.sqrt(2)))
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """State values, shape `(batch, 1)`."""
        return self.critic(obs)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """A sampled action, or the distribution mean when `deterministic`."""
        mean = self.actor_mean(obs)
        if deterministic:
            return mean
        return Normal(mean, torch.exp(self.actor_logstd.expand_as(mean))).sample()

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(action, log_prob, entropy, value)`; samples an action unless one is given."""
        mean = self.actor_mean(obs)
        dist = Normal(mean, torch.exp(self.actor_logstd.expand_as(mean)))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(obs)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    final_values: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation that bootstraps across auto-resets.

    Args:
        rewards, values, dones, final_values: `(num_steps, num_envs)`. `dones[t]` is the done
            flag *before* step t (the env was reset at the end of step t-1). `final_values[t]`
            is the value of the true last observation of an episode that ended at step t (zero
            where nothing ended); it replaces the bootstrap from the already-reset observation.
        next_value: `(1, num_envs)` value of the observation after the last step.
        next_done: `(num_envs,)` done flag after the last step.

    Returns:
        `(advantages, returns)`, both `(num_steps, num_envs)`.
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_done)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_not_done = 1.0 - next_done
            next_values = next_value
        else:
            next_not_done = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        real_next_values = next_not_done * next_values + final_values[t]
        delta = rewards[t] + gamma * real_next_values - values[t]
        last_gae = delta + gamma * gae_lambda * next_not_done * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def ppo_update(
    agent: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    cfg: IPPOConfig,
) -> dict[str, float]:
    """Clipped-PPO update of one agent over `cfg.update_epochs` epochs of shuffled minibatches.

    Args:
        batch: flattened rollout of this agent with keys `obs`, `actions`, `logprobs`,
            `advantages`, `returns`, `values` (first dim = `num_envs * num_steps`).

    Returns:
        Scalar metrics of the last minibatch (plus the mean clip fraction): `policy_loss`,
        `value_loss`, `entropy`, `old_approx_kl`, `approx_kl`, `clipfrac`, `explained_variance`.
    """
    size = batch["obs"].shape[0]
    device = batch["obs"].device
    clipfracs: list[float] = []
    zero = torch.zeros((), device=device)
    pg_loss = v_loss = entropy_loss = old_approx_kl = approx_kl = zero
    stop = False
    for _ in range(cfg.update_epochs):
        permutation = torch.randperm(size, device=device)
        for start in range(0, size, cfg.minibatch_size):
            idx = permutation[start : start + cfg.minibatch_size]
            _, new_logprob, entropy, new_value = agent.get_action_and_value(
                batch["obs"][idx], batch["actions"][idx]
            )
            log_ratio = new_logprob - batch["logprobs"][idx]
            ratio = log_ratio.exp()
            with torch.no_grad():
                # http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-log_ratio).mean()
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())
            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                stop = True
                break

            advantages = batch["advantages"][idx]
            if cfg.norm_adv and advantages.numel() > 1:  # std of one sample is NaN
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            pg_loss = torch.max(
                -advantages * ratio,
                -advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef),
            ).mean()

            new_value = new_value.view(-1)
            returns = batch["returns"][idx]
            if cfg.clip_vloss:
                old_value = batch["values"][idx]
                clipped = old_value + torch.clamp(
                    new_value - old_value, -cfg.clip_coef, cfg.clip_coef
                )
                v_loss = (
                    0.5 * torch.max((new_value - returns) ** 2, (clipped - returns) ** 2).mean()
                )
            else:
                v_loss = 0.5 * ((new_value - returns) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()
        if stop:
            break

    y_pred, y_true = batch["values"].cpu().numpy(), batch["returns"].cpu().numpy()
    var_y = np.var(y_true)
    explained_variance = float("nan") if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)
    return {
        "policy_loss": pg_loss.item(),
        "value_loss": v_loss.item(),
        "entropy": entropy_loss.item(),
        "old_approx_kl": old_approx_kl.item(),
        "approx_kl": approx_kl.item(),
        "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        "explained_variance": explained_variance,
    }
