"""Pure-PyTorch PPO building blocks (step 2.1): actor-critic network, GAE,
and the clipped PPO update.

No mani_skill/gymnasium dependency, so this module is unit-testable on
macOS/CI, unlike callosum.training.ippo (which needs the real ManiSkill env
and so cannot even be imported there) -- see docs/implementation-plan.md
section 0. Given this step is "written blind" (no local GPU to run it on),
this split is what lets the RL algorithm itself be verified before ever
touching the paid server session -- only the ManiSkill-specific env glue in
ippo.py remains genuinely unverified until then.

The network architecture, GAE recursion, and clipped-surrogate PPO loss
below are copied from mani-skill's own PPO baseline
(examples/baselines/ppo/ppo.py @ v3.0.1), not invented; only the config
object they read from (IPPOConfig, one per two independent agents instead
of a single shared one) differs.
"""

import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal

from callosum.configs.ippo import IPPOConfig


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """One actor-critic for one arm: continuous, diagonal-Gaussian action."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        return Normal(action_mean, action_std).sample()

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    final_values: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    args: IPPOConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation, bootstrapping correctly across
    partial-reset auto-resets via `final_values`.

    rewards/values/dones/final_values: (num_steps, num_envs).
    next_value: (1, num_envs). next_done: (num_envs,).

    `final_values[t]` should be nonzero only at (t, env) pairs where that env
    terminated at step t and was auto-reset -- see callosum.training.ippo for
    how it's populated from infos["final_observation"]/infos["_final_info"].

    Returns (advantages, returns), both (num_steps, num_envs).
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(next_done)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_not_done = 1.0 - next_done
            nextvalues = next_value
        else:
            next_not_done = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        real_next_values = next_not_done * nextvalues + final_values[t]
        delta = rewards[t] + args.gamma * real_next_values - values[t]
        lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns


def ppo_update(
    agent: Agent,
    optimizer: torch.optim.Optimizer,
    b_obs: torch.Tensor,
    b_actions: torch.Tensor,
    b_logprobs: torch.Tensor,
    b_advantages: torch.Tensor,
    b_returns: torch.Tensor,
    b_values: torch.Tensor,
    args: IPPOConfig,
) -> dict:
    """One agent's clipped-PPO update over `args.update_epochs` epochs of
    shuffled minibatches (size `args.minibatch_size`). Returns scalar
    metrics for logging."""
    batch_size = b_obs.shape[0]
    b_inds = np.arange(batch_size)
    clipfracs = []
    old_approx_kl = approx_kl = torch.tensor(0.0)
    pg_loss = v_loss = entropy_loss = torch.tensor(0.0)
    for _epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, args.minibatch_size):
            mb_inds = b_inds[start : start + args.minibatch_size]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                b_obs[mb_inds], b_actions[mb_inds]
            )
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_variance = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    return {
        "policy_loss": pg_loss.item(),
        "value_loss": v_loss.item(),
        "entropy": entropy_loss.item(),
        "old_approx_kl": old_approx_kl.item(),
        "approx_kl": approx_kl.item(),
        "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        "explained_variance": explained_variance,
    }
