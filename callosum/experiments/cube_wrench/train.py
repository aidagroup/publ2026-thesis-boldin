"""IPPO trainer of the reduced two-agent suite: CubeWrench-v0 and the coop tasks.

The `--task` flag selects the environment: the default `"cube"` keeps the thesis main line
(`CubeWrench-v0`, `holder`/`rotator`) bit-for-bit on its original CPU/warp paths, while
`"cartpole"`/`"pendulum"` train the two-agent coop suite (`left`/`right`) on the same
trainer, policies, GAE, checkpoints and Bi-JEPA path -- the env supplies its observation
width, action width and agent keys (`WrenchPPOConfig.env_obs_size`/`action_size`/
`agent_keys`), and the coop vectorisation is `callosum.experiments.coop_tasks.vec_env.VecCoopEnv`.

Two independent PPO learners -- one Gaussian actor-critic per agent -- share one
environment step and one **team** reward, so the coordination has to be learned from the
decentralised observations alone (the `none` / `oracle` / `predicted` ablation of the
plan). The default env backend (`--backend cpu`) is the single-instance MuJoCo CPU env
(`callosum.experiments.cube_wrench.WrenchCubeEnv`), so its vectorisation is a Python loop:
`callosum.experiments.cube_wrench.vec_env.VecWrenchCube` drives `num_envs` copies with auto-reset and
the finished-episode statistics. The optional GPU backend (`--backend warp --device cuda`)
replaces the loop with `callosum.experiments.cube_wrench.env_gpu.WrenchCubeBatchGPU`, which steps
all `num_envs` worlds together through `mujoco_warp` (same env contract, same policies and
checkpoints; see the config module for the manual install note). The trainer's rollout code
is backend-agnostic: it only asks the vec env for its `torch_io` marker, so a GPU step
consumes CUDA tensors instead of the numpy copies of the CPU loop.

Reused from `callosum.common.ppo_core` (the ManiSkill PPO baseline's blocks, shared with
the arms IPPO trainer): `ActorCritic` (3 x 256 tanh MLPs, state-independent log-std),
`compute_gae` and the clipped-surrogate `ppo_update`. What this trainer adds is the
wrench-specific contract around them:

* the policy input is the env's flat observation, width `obs_size(partner_mode)` (28 for
  `none`, 45 for `oracle`), and the action is the normalized 6D wrench of the role;
* under `predicted` (`use_bijepa=True`) the env stays at `partner_mode="none"` and the policy
  input becomes `[own observation (28), ẑ_partner (latent_dim, 32)]` = 60: the agent predicts
  its partner's latent from its OWN observation history (`LatentInputBuilder` ->
  `BiJEPA.encode` + `BiJEPA.predict_partner`, no partner observation), so the policy is still
  trained on a decentralised input. The Bi-JEPA encoder/predictor are trained by an auxiliary
  latent MSE + VICReg terms (`compute_aux_loss`, EMA target, stop-gradient) with their own
  Adam optimizer, `aux_coef * loss` per update, then `update_target()`; the trainer logs
  `aux/*`, `latent/std`, `latent/effective_rank` and a held-out linear-probe quality of `ẑ_j`
  (`callosum.training._bijepa_probe.ridge_probe`);
* GAE bootstraps only on time-limit truncations: success and failure are true terminals,
  so the shared success bonus is not fed back through `V(final observation)`
  (`compute_advantages`);
* one update = `num_envs * rollout_steps` transitions of the loop vector env, then a PPO
  update of each policy on its own rollout batch (the reward and GAE inputs are shared,
  the value/advantage streams are per agent);
* logs `train/return`, `train/success_rate`, `train/episode_len` over the episodes that
  finished in the rollout, greedy `eval/*` every `eval_every` updates, and writes full
  `latest.pt` checkpoints (weights + Adam moments + config + counters + torch RNG + the
  Bi-JEPA module and its optimizer under `predicted`) that `--resume` restores.

The module deliberately imports torch but not mujoco at module level: the vec env is
imported inside `run()`, so `tests/test_wrench_ppo.py` and `tests/test_wrench_ppo_bijepa.py`
(policy shapes, aux loss/gradients, the probe, checkpoint roundtrip, a toy-vec training
smoke) run on macOS/CI without the `sandbox` extra.

Typical use (RunPod GPU pod, MuJoCo on the CPU; CPU simulation also works on a laptop but
the plan forbids training there):

    python -m callosum.experiments.cube_wrench.train --partner-mode oracle --seed 0 \
        --exp-name wrench_oracle_s0 --total-updates 300
    python -m callosum.experiments.cube_wrench.train --partner-mode none --use-bijepa --seed 0 \
        --exp-name wrench_pred_s0 --total-updates 300

The optional GPU backend and the harder-randomness preset (install note in the config module;
`uv pip install --python <venv>/bin/python mujoco-warp==4.0.1`, because a plain `uv sync --frozen`
removes it):

    python -m callosum.experiments.cube_wrench.train --backend warp --device cuda --num-envs 1024 \
        --rollout-steps 64 --randomization harsh --partner-mode oracle --total-updates 300

`--resume runs/<exp>` continues a killed run from its `latest.pt` (the saved config wins,
so the hyperparameters cannot drift between restarts). SIGTERM saves a checkpoint after the
current update, so a detached run is never lost.
"""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from callosum.common.bijepa import BiJEPA, latent_stats
from callosum.common.logging import MetricLogger
from callosum.common.ppo_core import ActorCritic, compute_gae, ppo_update
from callosum.experiments.cube_wrench.logic import AGENT_KEYS, obs_size
from callosum.experiments.cube_wrench.ppo_config import (
    INPUT_MODES,
    WrenchPPOConfig,
    checkpoint_task,
    parse_args_with_explicit,
    task_obs_size,
)
from callosum.training._bijepa_probe import ridge_probe

CHECKPOINT_FORMAT = 1
"""Format of the trainer's checkpoints: weights, optimizer states, config and counters."""

EVAL_SEED_OFFSET = 10_000_019
"""Offset of the held-out evaluation seeds from the training seed.

Training episodes start at `cfg.seed` and the auto-reset stream continues from
`cfg.seed + num_envs`, so evaluating at the raw seed would reuse the training episode
randomness. Every greedy evaluation therefore uses `base_seed + EVAL_SEED_OFFSET + episode`
(one episode per fixed seed, see :func:`evaluate_greedy`), which is a seed stream the run
never trains on.
"""

RUNTIME_RESOURCE_FIELDS = ("backend", "device")
"""Config fields `--resume` takes from the CLI instead of the checkpoint.

They describe WHERE the run executes (CPU loop vs the warp GPU env, torch device), not the
learning setup, so a CPU checkpoint can be continued on the GPU backend and vice versa; the
saved hyperparameters still win for everything else.
"""

_WRENCH_SIZE = 6
"""Width of one agent's action (a normalized 6D wrench)."""


# --- policies -----------------------------------------------------------------------------


def make_policies(
    partner_mode: str,
    device: torch.device | str = "cpu",
    *,
    latent_dim: int = 0,
    input_size: int | None = None,
    action_dim: int = _WRENCH_SIZE,
) -> list[ActorCritic]:
    """One independent Gaussian actor-critic per role, in the env's agent-key order.

    The default input width is `callosum.experiments.cube_wrench.logic.obs_size(partner_mode) +
    latent_dim` (the cube: 28 for `none`, 45 for `oracle`, plus the Bi-JEPA latent under
    `predicted`); the coop tasks pass their env observation width explicitly
    (`WrenchPPOConfig.env_obs_size`), so the same policy code serves both suites. The
    output width is the env's action width (`6` for the cube's 6D wrench, `1` for the coop
    tasks' one-sided force/torque). Sharing no parameters is what makes the pair an
    IPPO-style decentralised team rather than a single joint policy.

    Args:
        partner_mode: Env partner-visibility mode; used to derive the default input width.
        device: Torch device of both policies.
        latent_dim: Width of the appended Bi-JEPA predicted latent (`0` for the `none` /
            `oracle` baselines).
        input_size: Explicit policy input width (env observation width) when the caller
            knows the task, otherwise derived from `partner_mode`.
        action_dim: Width of one action (the env's action-space width).

    Returns:
        Two policies, one per agent (for the cube `[holder, rotator]`, for the coop suite
        `[left, right]`).
    """
    policy_input = (obs_size(partner_mode) if input_size is None else input_size) + latent_dim
    return [ActorCritic(policy_input, action_dim).to(device) for _ in AGENT_KEYS]


def make_bijepa(cfg: WrenchPPOConfig, device: torch.device | str = "cpu") -> BiJEPA | None:
    """Build the Bi-JEPA module of a `predicted` run (`None` for the `none`/`oracle` modes).

    Both roles observe the same env width under `none`, so the module gets the same
    `obs_dim` twice; its per-agent encoders/predictors stay separate
    (`cfg.bijepa_config()` keeps `BiJEPAConfig`'s defaults: separate encoders, EMA targets,
    VICReg-style variance/covariance terms).
    """
    if not cfg.use_bijepa:
        return None
    return BiJEPA(cfg.bijepa_config(), cfg.env_obs_size, cfg.env_obs_size).to(device)


def make_bijepa_optimizer(
    cfg: WrenchPPOConfig, bijepa: BiJEPA | None
) -> torch.optim.Optimizer | None:
    """Adam over the online encoders/predictors of `bijepa` (`None` without Bi-JEPA).

    The EMA target encoders have `requires_grad=False` and are not passed to the optimizer;
    `BiJEPA.trainable_parameters()` already excludes them.
    """
    if bijepa is None:
        return None
    return torch.optim.Adam(bijepa.trainable_parameters(), lr=cfg.learning_rate, eps=1e-5)


class LatentInputBuilder:
    """Builds the `predicted` policy input `[own env observation, ẑ_partner]`.

    The partner latent is predicted from the agent's OWN observation history only
    (`BiJEPA.encode` of the own window -> `BiJEPA.predict_partner`), never from the partner's
    observation -- so the policy cannot see the ground-truth partner block of the `oracle`
    mode even when the environment knows it. The policy's own part is the last row of the
    history, i.e. the current env observation.

    Every method runs under `torch.no_grad()`: the produced latent is a fixed feature of the
    observation for the PPO update (like the env observation itself), and the encoder /
    predictor are trained by the separate Bi-JEPA auxiliary loss (`aux_update`).

    The history is a `(num_envs, history_len, obs_dim)` rolling window per role. An episode
    boundary (the done flag of the step that just ended) resets the affected envs' windows to
    repetitions of the fresh observation, so a predictor with `history_len > 1` never mixes
    latents of two episodes.
    """

    def __init__(
        self,
        bijepa: BiJEPA,
        device: torch.device | str = "cpu",
        *,
        agent_keys: tuple[str, ...] | None = None,
    ) -> None:
        """Wrap a trained-in-training Bi-JEPA module; `history_len`/`latent_dim` come from it.

        Args:
            bijepa: The Bi-JEPA module (its observation width fixes the own-observation size).
            device: Torch device of the history buffers and the module.
            agent_keys: The env's two agent keys, in observation-dict order; defaults to the
                cube's `("holder", "rotator")` so the old call sites keep working.
        """
        self.bijepa = bijepa
        self.device = torch.device(device)
        self.agent_keys = AGENT_KEYS if agent_keys is None else tuple(agent_keys)
        self.history_len = bijepa.cfg.history_len
        self.obs_dim = bijepa.obs_dims[0]
        self.latent_dim = bijepa.cfg.latent_dim
        self.history: dict[str, torch.Tensor] = {}

    @property
    def input_size(self) -> int:
        """Width of one built policy input (`obs_dim + latent_dim`, 58 by default)."""
        return self.obs_dim + self.latent_dim

    @torch.no_grad()
    def reset(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Start a new rollout: every history window is `history_len` copies of `obs`."""
        self.history = {
            name: obs[name].unsqueeze(1).repeat(1, self.history_len, 1) for name in self.agent_keys
        }
        return self._augment(self.history)

    @torch.no_grad()
    def step(self, obs: dict[str, torch.Tensor], done: torch.Tensor) -> dict[str, torch.Tensor]:
        """Advance the history by `obs` (resetting the `done` envs) and return new inputs.

        Args:
            obs: The next observation per role, `(num_envs, obs_dim)`; for `done` envs it is
                the fresh first observation of the auto-reset episode.
            done: `(num_envs,)` 0/1 flags of the step that just finished.
        """
        self.history = self._rolled(obs, done)
        return self._augment(self.history)

    @torch.no_grad()
    def history_inputs(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Inputs as if `obs` came next, WITHOUT committing it to the history.

        Used for the GAE truncation bootstrap: `result.final_obs` is the true last observation
        of the episode, so its value must be computed from the history window ending at it.
        """
        return self._augment(self._rolled(obs, None))

    def _rolled(
        self, obs: dict[str, torch.Tensor], done: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        """Shift every history window by one observation (episode-resetting `done` envs)."""
        rolled: dict[str, torch.Tensor] = {}
        for name in self.agent_keys:
            window = torch.cat([self.history[name][:, 1:, :], obs[name].unsqueeze(1)], dim=1)
            if done is not None:
                mask = done.bool().view(-1, 1, 1)
                fresh = obs[name].unsqueeze(1).expand(-1, self.history_len, -1)
                window = torch.where(mask, fresh, window)
            rolled[name] = window
        return rolled

    def _augment(self, history: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """`[last own observation, predicted partner latent]` for every role."""
        inputs: dict[str, torch.Tensor] = {}
        for index, name in enumerate(self.agent_keys):
            z_hist = self.bijepa.encode(history[name], index)
            z_hat = self.bijepa.predict_partner(z_hist, index)
            inputs[name] = torch.cat([history[name][:, -1, :], z_hat], dim=-1)
        return inputs


# --- GAE / PPO ----------------------------------------------------------------------------


def compute_advantages(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    truncated: torch.Tensor,
    final_values: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE whose bootstrap is taken only on time-limit truncations.

    Success and failure are true terminals (the plan: the success bonus is the terminal
    reward, not a potential), so their `final_values` must be zero and the advantage chain
    stops there. A truncation is not a terminal -- the episode budget simply ran out -- so
    it bootstraps from the value of its true last observation. The masking is done here
    once, so the training loop cannot get it wrong per agent.

    Args:
        rewards, values, dones: `(rollout_steps, num_envs)` of the shared rollout;
            `dones[t]` is the done flag before step t (the env was reset at the end of
            step t-1), as in `_ppo_core.compute_gae`.
        truncated: `(rollout_steps, num_envs)` 0/1 truncation masks of the same rollout.
        final_values: `(rollout_steps, num_envs)` value of the true last observation where
            the episode ended (zero elsewhere).
        next_value: `(1, num_envs)` value of the observation after the rollout.
        next_done: `(num_envs,)` done flag after the rollout.
        gamma: Discount.
        gae_lambda: GAE lambda.

    Returns:
        `(advantages, returns)`, both `(rollout_steps, num_envs)`.
    """
    bootstrap = torch.where(truncated.bool(), final_values, torch.zeros_like(final_values))
    return compute_gae(
        rewards,
        values,
        dones,
        bootstrap,
        next_value,
        next_done,
        gamma,
        gae_lambda,
    )


def ppo_step(
    agent: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    cfg: WrenchPPOConfig,
) -> dict[str, float]:
    """One clipped-PPO update of one policy, rejecting a non-finite loss.

    Thin wrapper around `_ppo_core.ppo_update` (its `cfg` duck-type: `update_epochs`,
    `minibatch_size`, `clip_coef`, ... all exist on `WrenchPPOConfig`), adding the trainer's
    guard: a non-finite policy/value/entropy loss is raised instead of being written to
    TensorBoard and silently poisoning the run.

    Args:
        agent: The policy to update.
        optimizer: Its Adam optimizer (learning rate is set by the caller).
        batch: the flattened rollout with keys `obs`, `actions`, `logprobs`, `advantages`,
            `returns`, `values`.
        cfg: The trainer config.

    Returns:
        The `ppo_update` metrics of the last minibatch.
    """
    metrics = ppo_update(agent, optimizer, batch, cfg)
    bad = [
        name for name in ("policy_loss", "value_loss", "entropy") if not np.isfinite(metrics[name])
    ]
    if bad:
        raise FloatingPointError(f"non-finite {bad} at this update: {metrics}")
    return metrics


@torch.no_grad()
def evaluate_greedy(
    policies: list[ActorCritic],
    vec: object,
    episodes: int,
    *,
    device: torch.device | str = "cpu",
    seed: int = 0,
    bijepa: BiJEPA | None = None,
    agent_keys: tuple[str, ...] | None = None,
) -> EvalSummary:
    """Greedy evaluation on a FIXED set of held-out seeds, one episode per seed.

    The old protocol collected the first `episodes` finished episodes of an auto-resetting
    batch env; fast successes then dominated (each world could contribute several episodes)
    and time-limit truncations were almost never collected, so the number was biased towards
    easy episodes and the seed-to-episode mapping was implicit. Here world `i` of a batch is
    reset with exactly one seed, `seed + <batch offset> + i`, and only that world's FIRST
    episode is recorded; the evaluation steps until the first episode of every requested
    world has finished (success, failure or the 350-step truncation). If `episodes` exceeds
    the vec env's `num_envs`, the seeds are covered in consecutive batches of worlds.
    Callers pass `cfg.seed + EVAL_SEED_OFFSET + ...` so the seeds are outside the training
    stream.

    With `bijepa` (the `predicted` input mode) the policies are fed the exact same
    `[own observation, ẑ_partner]` construction as in training (`LatentInputBuilder` from
    the agent's own observation history, no ground-truth partner data).

    Args:
        policies: `[holder_policy, rotator_policy]`.
        vec: A `callosum.experiments.cube_wrench.vec_env.VecWrenchCube`-like object (`reset`, `step`,
            `num_envs`, `config.max_control_steps`); typed loosely so this function needs
            no simulator import.
        episodes: Number of held-out episodes to evaluate (one episode per seed).
        device: Torch device of the policies.
        seed: Base episode seed; episode `k` uses `seed + k` (the caller applies
            `EVAL_SEED_OFFSET`).
        bijepa: The `predicted` run's Bi-JEPA module, or `None` for `none`/`oracle`.

    Returns:
        The aggregated `EvalSummary` over exactly `episodes` first episodes.
    """
    keys = AGENT_KEYS if agent_keys is None else tuple(agent_keys)
    torch_io = bool(getattr(vec, "torch_io", False))
    builder = LatentInputBuilder(bijepa, device, agent_keys=keys) if bijepa is not None else None
    num_envs = int(vec.num_envs)
    max_control_steps = int(vec.config.max_control_steps)
    successes: list[bool] = []
    returns: list[float] = []
    lengths: list[int] = []
    angles: list[float] = []
    while len(successes) < episodes:
        batch = min(num_envs, episodes - len(successes))
        obs = _to_torch(vec.reset(seed=seed + len(successes)), device)
        inputs = builder.reset(obs) if builder is not None else obs
        recorded = [False] * num_envs
        for _step in range(max_control_steps + 1):
            actions = {}
            for name, policy in zip(keys, policies):
                action = policy.get_action(inputs[name], deterministic=True)
                clipped = torch.clamp(action, -1.0, 1.0)
                actions[name] = clipped if torch_io else clipped.cpu().numpy()
            result = vec.step(actions)
            obs = _to_torch(result.obs, device)
            terminated = _flags(result.terminated)
            truncated = _flags(result.truncated)
            done = terminated | truncated
            if builder is not None:
                done_tensor = torch.as_tensor(terminated, device=device, dtype=torch.bool)
                done_tensor = done_tensor | torch.as_tensor(
                    truncated, device=device, dtype=torch.bool
                )
                inputs = builder.step(obs, done_tensor.float())
            else:
                inputs = obs
            done_indices = np.flatnonzero(done)
            if len(done_indices) != len(result.finished_returns):
                raise RuntimeError(
                    "vec env contract broken: "
                    f"{len(done_indices)} worlds finished but "
                    f"{len(result.finished_returns)} finished statistics reported"
                )
            for position, world in enumerate(done_indices):
                if world >= batch or recorded[world]:
                    continue
                recorded[world] = True
                successes.append(bool(result.finished_successes[position]))
                returns.append(float(result.finished_returns[position]))
                lengths.append(int(result.finished_lengths[position]))
                angles.append(float(result.finished_angles[position]))
            if all(recorded[:batch]):
                break
        else:
            raise RuntimeError(
                "evaluation did not finish the first episode of every world within "
                f"{max_control_steps + 1} steps; the env's episode budget is broken "
                f"({sum(recorded[:batch])}/{batch} recorded)"
            )
    return EvalSummary(
        episodes=episodes,
        success_rate=float(np.mean(successes)),
        mean_return=float(np.mean(returns)),
        mean_angle_deg=float(np.mean(angles)),
        mean_length=float(np.mean(lengths)),
    )


def _flags(value: object) -> np.ndarray:
    """A vec env's terminated/truncated flags as a numpy bool array (tensor or array input)."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(bool)
    return np.asarray(value, dtype=bool)


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate of one greedy evaluation."""

    episodes: int
    success_rate: float
    mean_return: float
    mean_angle_deg: float
    mean_length: float


# --- Bi-JEPA auxiliary loss (`predicted`) -------------------------------------------------


def compute_aux_loss(
    bijepa: BiJEPA,
    cfg: WrenchPPOConfig,
    obs_hist_a: torch.Tensor,
    obs_hist_b: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Weighted Bi-JEPA auxiliary loss over one rollout batch, both directions.

    "Bi" is `BiJEPA.forward_both`: the holder predicts the rotator's latent and the rotator
    predicts the holder's, each from its own observation history; the target is the partner's
    latent through its EMA target encoder (stop-gradient), so no gradient reaches the partner
    observation or the target encoders.

    Args:
        bijepa: The online Bi-JEPA module.
        cfg: The run config (only `aux_coef` is read here).
        obs_hist_a: Holder own-observation history `(B, K, obs_dim)` of the rollout.
        obs_hist_b: Rotator own-observation history `(B, K, obs_dim)` of the rollout.

    Returns:
        A flat tensor dict: `loss` = `aux_coef * (mse + var + cov)` (the tensor to
        `backward()`), `mse` / `var` / `cov` = the mean scalar terms, `mse_a` / `mse_b` = the
        per-direction latents MSE, and the per-direction online self latents, predictions and
        stop-grad targets (`z_self_*`, `z_pred_*`, `z_target_*`) for the logger/probe.
    """
    out = bijepa.forward_both(obs_hist_a, obs_hist_b)
    a, b = out["a"], out["b"]
    return {
        "loss": cfg.aux_coef * out["loss"],
        "mse": 0.5 * (a["loss_pred"] + b["loss_pred"]),
        "var": 0.5 * (a["loss_var"] + b["loss_var"]),
        "cov": 0.5 * (a["loss_cov"] + b["loss_cov"]),
        "mse_a": a["loss_pred"],
        "mse_b": b["loss_pred"],
        "z_self_a": a["z_self"],
        "z_self_b": b["z_self"],
        "z_pred_a": a["z_partner_pred"],
        "z_pred_b": b["z_partner_pred"],
        "z_target_a": a["z_partner_target"],
        "z_target_b": b["z_partner_target"],
    }


@torch.no_grad()
def probe_predictions(
    terms: dict[str, torch.Tensor],
    obs_hist_a: torch.Tensor,
    obs_hist_b: torch.Tensor,
    action_size: int = _WRENCH_SIZE,
) -> dict[str, float]:
    """Held-out linear-probe quality of the predicted partner latents (`aux/probe/*` scalars).

    Two probes per direction (`_bijepa_probe.ridge_probe`, train/test split inside):

    * `wrench_*`: `ẑ_j` -> the partner's current commanded 6D wrench (the first six fields of
      the partner's own observation) -- the interpretable "does the prediction carry the
      partner's action" number for the report;
    * `latent_*`: `ẑ_j` -> the partner's stop-grad target latent -- the auxiliary objective
      itself, in the same units as the training curves.

    Args:
        terms: Output of :func:`compute_aux_loss` (detached use only).
        obs_hist_a: Holder own-observation history `(B, K, obs_dim)`.
        obs_hist_b: Rotator own-observation history `(B, K, obs_dim)`.

    Returns:
        Mean over the two directions of `mse`, `r2` and `baseline_mse` per probe family.
        The rows of one rollout are temporally correlated, so the held-out split is not a
        fully independent test: treat `r2` as an optimistic but consistently comparable
        quality signal, not as an unbiased generalisation estimate.
    """
    wrench = {"a": obs_hist_b[:, -1, :action_size], "b": obs_hist_a[:, -1, :action_size]}
    wrench_rows = [
        ridge_probe(terms[f"z_pred_{direction}"], wrench[direction]) for direction in ("a", "b")
    ]
    latent_rows = [
        ridge_probe(terms[f"z_pred_{direction}"], terms[f"z_target_{direction}"])
        for direction in ("a", "b")
    ]
    metrics: dict[str, float] = {}
    for prefix, rows in (("wrench", wrench_rows), ("latent", latent_rows)):
        for key in ("mse", "r2", "baseline_mse"):
            metrics[f"aux/probe/{prefix}_{key}"] = float(np.mean([row[key] for row in rows]))
    return metrics


def aux_update(
    bijepa: BiJEPA,
    optimizer: torch.optim.Optimizer,
    cfg: WrenchPPOConfig,
    obs_hist_a: torch.Tensor,
    obs_hist_b: torch.Tensor,
) -> dict[str, float]:
    """One Bi-JEPA auxiliary optimiser step + EMA target update; returns logged scalars.

    The policy optimizers never see these parameters: the encoders/predictors are trained only
    by this loss, and the produced `ẑ_j` is a fixed policy input (see `LatentInputBuilder`).

    Args:
        bijepa: The online Bi-JEPA module.
        optimizer: Its Adam (built from `bijepa.trainable_parameters()`).
        cfg: The run config (`aux_coef`, `max_grad_norm`).
        obs_hist_a: Holder own-observation history `(B, K, obs_dim)` of the rollout.
        obs_hist_b: Rotator own-observation history `(B, K, obs_dim)` of the rollout.

    Returns:
        TensorBoard-ready floats: `aux/loss`, `aux/mse`, `aux/var`, `aux/cov`, the
        per-direction `aux/mse_a`/`aux/mse_b`, `latent/std`, `latent/std_min`,
        `latent/effective_rank` and the probe block (`aux/probe/*`).

    Raises:
        FloatingPointError: if a loss/metric is not finite (same guard as the PPO step).
    """
    terms = compute_aux_loss(bijepa, cfg, obs_hist_a, obs_hist_b)
    optimizer.zero_grad(set_to_none=True)
    terms["loss"].backward()
    nn.utils.clip_grad_norm_(bijepa.trainable_parameters(), cfg.max_grad_norm)
    optimizer.step()
    bijepa.update_target()

    metrics = {
        "aux/loss": float(terms["loss"].detach()),
        "aux/mse": float(terms["mse"].detach()),
        "aux/var": float(terms["var"].detach()),
        "aux/cov": float(terms["cov"].detach()),
        "aux/mse_a": float(terms["mse_a"].detach()),
        "aux/mse_b": float(terms["mse_b"].detach()),
    }
    bad = [tag for tag, value in metrics.items() if not np.isfinite(value)]
    if bad:
        raise FloatingPointError(f"non-finite Bi-JEPA loss at this update: {bad} ({metrics})")
    with torch.no_grad():
        z_self = torch.cat(
            [
                terms["z_self_a"].reshape(-1, bijepa.cfg.latent_dim),
                terms["z_self_b"].reshape(-1, bijepa.cfg.latent_dim),
            ]
        )
    stats = latent_stats(z_self)
    metrics["latent/std"] = stats["std_mean"]
    metrics["latent/std_min"] = stats["std_min"]
    metrics["latent/effective_rank"] = stats["eff_rank"]
    metrics.update(probe_predictions(terms, obs_hist_a, obs_hist_b, cfg.action_size))
    return metrics


# --- checkpoints --------------------------------------------------------------------------


def _rng_payload() -> dict:
    """The torch RNG state a checkpoint stores (CPU state + all CUDA devices when present).

    The CUDA generator state matters for a warp run: the policy sampling happens on the GPU,
    so a resume without it replays different actions. The list is serialised by `torch.save`
    and accepted by `torch.load(weights_only=True)` (it is a list of byte tensors).
    """
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(rng: dict) -> None:
    """Restore a checkpoint's torch RNG state (CUDA too when the field/device is available)."""
    torch.set_rng_state(rng["torch"].cpu())
    cuda_state = rng.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(
    path: Path,
    policies: list[ActorCritic],
    optimizers: list[torch.optim.Optimizer],
    cfg: WrenchPPOConfig,
    update: int,
    global_step: int,
    eval_metrics: dict[str, float] | None = None,
    *,
    bijepa: BiJEPA | None = None,
    bijepa_optimizer: torch.optim.Optimizer | None = None,
    agent_keys: tuple[str, ...] | None = None,
) -> None:
    """Write the full training state to `path` atomically.

    The payload holds both policies' weights, both Adam states, the config (so `--resume`
    cannot run with drifted hyperparameters), the update / environment-step counters and the
    torch RNG state (CPU + all CUDA device generators, so a warp run's GPU policy sampling is
    reproducible after `--resume`). A `predicted` run additionally stores the Bi-JEPA module
    (encoders,
    predictors and the EMA target encoders) and its optimizer under `bijepa` /
    `bijepa_optimizer`, plus the policy input metadata (`input_mode`, `policy_input_size`,
    `latent_dim`) the evaluator needs. A kill mid-write cannot corrupt the file: it is written
    to `<path>.tmp` and renamed.

    Args:
        path: Destination file (normally `<run dir>/latest.pt`).
        policies: `[holder_policy, rotator_policy]`.
        optimizers: Their optimizers, in the same order.
        cfg: The run config.
        update: Number of completed updates (1-based).
        global_step: Environment steps consumed so far.
        eval_metrics: The latest greedy evaluation, for the checkpoint's `eval` block.
        bijepa: The run's Bi-JEPA module (only for `use_bijepa=True` runs).
        bijepa_optimizer: Its Adam optimizer, restored by `--resume`.
    """
    keys = AGENT_KEYS if agent_keys is None else tuple(agent_keys)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "agents": {name: policy.state_dict() for name, policy in zip(keys, policies)},
        "optimizers": {name: optimizer.state_dict() for name, optimizer in zip(keys, optimizers)},
        "config": dataclasses.asdict(cfg),
        "obs_size": cfg.obs_size,
        "action_size": cfg.action_size,
        "partner_mode": cfg.partner_mode,
        "input_mode": cfg.input_mode,
        "policy_input_size": cfg.policy_input_size,
        "use_bijepa": cfg.use_bijepa,
        "latent_dim": cfg.latent_dim,
        "update": int(update),
        "global_step": int(global_step),
        "eval": {key: float(value) for key, value in (eval_metrics or {}).items()},
        "rng": _rng_payload(),
    }
    if bijepa is not None:
        payload["bijepa"] = bijepa.state_dict()
    if bijepa_optimizer is not None:
        payload["bijepa_optimizer"] = bijepa_optimizer.state_dict()
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path) -> dict:
    """Read a trainer checkpoint (tensors and plain containers only)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format {payload.get('format')!r} in {path}")
    return payload


def load_weights(
    payload: dict,
    policies: list[ActorCritic],
    *,
    agent_keys: tuple[str, ...] | None = None,
) -> None:
    """Load both policies' weights from a checkpoint payload (in the env's agent-key order)."""
    keys = AGENT_KEYS if agent_keys is None else tuple(agent_keys)
    for name, policy in zip(keys, policies):
        policy.load_state_dict(payload["agents"][name])


def load_optimizers(
    payload: dict,
    optimizers: list[torch.optim.Optimizer],
    *,
    agent_keys: tuple[str, ...] | None = None,
) -> None:
    """Load both optimizer states from a checkpoint payload (in the env's agent-key order)."""
    keys = AGENT_KEYS if agent_keys is None else tuple(agent_keys)
    for name, optimizer in zip(keys, optimizers):
        optimizer.load_state_dict(payload["optimizers"][name])


def load_bijepa_weights(payload: dict, bijepa: BiJEPA) -> None:
    """Load a `predicted` checkpoint's Bi-JEPA module (online + EMA target encoders)."""
    if payload.get("bijepa") is None:
        raise ValueError(
            "checkpoint has no Bi-JEPA weights; it was not trained with use_bijepa=True"
        )
    bijepa.load_state_dict(payload["bijepa"])


def load_bijepa_optimizer(payload: dict, optimizer: torch.optim.Optimizer) -> None:
    """Load the Bi-JEPA optimizer state of a `predicted` checkpoint."""
    if payload.get("bijepa_optimizer") is None:
        raise ValueError("checkpoint has no Bi-JEPA optimizer state")
    optimizer.load_state_dict(payload["bijepa_optimizer"])


# --- checkpoint / input-mode compatibility -------------------------------------------------


def env_partner_mode(input_mode: str) -> str:
    """The env's partner-visibility mode a policy input mode runs on.

    `predicted` is a *policy input* mode: the env itself stays unprivileged
    (`partner_mode="none"`), so the agent cannot see the partner block it is asked to predict.
    `none` and `oracle` are env modes and map onto themselves.

    Raises:
        ValueError: if `input_mode` is not one of `callosum.experiments.cube_wrench.ppo_config.INPUT_MODES`.
    """
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}")
    return "none" if input_mode == "predicted" else input_mode


def checkpoint_latent_dim(payload: dict) -> int:
    """The Bi-JEPA latent width recorded in a checkpoint (config fallback: 32)."""
    if payload.get("latent_dim") is not None:
        return int(payload["latent_dim"])
    config = payload.get("config") or {}
    if config.get("latent_dim") is not None:
        return int(config["latent_dim"])
    return 32


def check_checkpoint_compat(payload: dict, input_mode: str) -> tuple[str, int]:
    """Validate a checkpoint against a requested policy input mode.

    The evaluator calls this before building anything, so a wrong `--input-mode` fails with a
    readable error instead of a shape mismatch deep in the policy forward pass. Old
    (`format 1`, pre-Bi-JEPA) checkpoints are accepted: they record `partner_mode` and
    `obs_size` but no `input_mode`, and their policy width is the env observation width.

    Args:
        payload: A `load_checkpoint` payload.
        input_mode: One of `INPUT_MODES`.

    Returns:
        `(env partner mode, latent width)` -- the env mode to construct and the extra policy
        input width (`0` outside `predicted`).

    Raises:
        ValueError: on any mismatch between the checkpoint and the requested mode.
    """
    env_mode = env_partner_mode(input_mode)
    saved_input = payload.get("input_mode")
    if saved_input is not None and saved_input != input_mode:
        raise ValueError(
            f"checkpoint was made with input_mode={saved_input!r}, this run uses "
            f"{input_mode!r}: the policy input differs"
        )
    saved_partner = payload.get("partner_mode")
    if saved_partner is not None and saved_partner != env_mode:
        raise ValueError(
            f"checkpoint was made with partner_mode={saved_partner!r}, input_mode "
            f"{input_mode!r} needs partner_mode={env_mode!r}"
        )
    task = checkpoint_task(payload)
    expected_obs = task_obs_size(task, env_mode)
    saved_task = (payload.get("config") or {}).get("task")
    if saved_task is not None and saved_task != task:  # pragma: no cover - defensive
        raise ValueError(f"checkpoint records task={saved_task!r} but its config says {task!r}")
    saved_obs = payload.get("obs_size")
    if saved_obs is not None and int(saved_obs) != expected_obs:
        raise ValueError(
            f"checkpoint policies take observations of width {saved_obs} but partner_mode "
            f"{env_mode!r} observations are {expected_obs} wide"
        )
    has_bijepa = payload.get("bijepa") is not None
    if input_mode == "predicted":
        if not has_bijepa:
            raise ValueError(
                "checkpoint has no Bi-JEPA weights: it was not trained with use_bijepa=True; "
                "evaluate it with --input-mode none/oracle"
            )
        latent_dim = checkpoint_latent_dim(payload)
    else:
        if has_bijepa:
            raise ValueError(
                "checkpoint contains Bi-JEPA weights: evaluate it with --input-mode predicted"
            )
        latent_dim = 0
    expected_input = expected_obs + latent_dim
    expected_action = 6 if task == "cube" else 1
    saved_action = payload.get("action_size")
    if saved_action is not None and int(saved_action) != expected_action:
        raise ValueError(
            f"checkpoint policies output {saved_action} action values but task {task!r} has "
            f"{expected_action} action dimensions"
        )
    saved_input_size = payload.get("policy_input_size")
    if saved_input_size is not None and int(saved_input_size) != expected_input:
        raise ValueError(
            f"checkpoint policies take inputs of width {saved_input_size} but input_mode "
            f"{input_mode!r} gives {expected_input} ({expected_obs} env obs + "
            f"{latent_dim} latent)"
        )
    return env_mode, latent_dim


def config_from_payload(payload: dict) -> WrenchPPOConfig:
    """The run config saved in a checkpoint payload.

    New (`use_bijepa` etc.) fields fall back to their dataclass defaults when an older
    checkpoint's config dict predates them, which keeps the `none`/`oracle` checkpoints of the
    step-2 trainer loadable.
    """
    field_names = {field.name for field in dataclasses.fields(WrenchPPOConfig)}
    values = {key: value for key, value in payload["config"].items() if key in field_names}
    return WrenchPPOConfig(**values)


def _check_resumed_state(payload: dict, cfg: WrenchPPOConfig) -> None:
    """Refuse to resume a checkpoint into an incompatible input mode / policy width."""
    saved_use_bijepa = bool(payload.get("use_bijepa", False))
    if saved_use_bijepa != cfg.use_bijepa:
        raise ValueError(
            f"checkpoint use_bijepa={saved_use_bijepa} but the run config has "
            f"use_bijepa={cfg.use_bijepa}: the policy input width differs "
            f"({payload.get('policy_input_size')} vs {cfg.policy_input_size})"
        )
    saved_input = payload.get("input_mode")
    if saved_input is not None and saved_input != cfg.input_mode:
        raise ValueError(
            f"checkpoint input_mode={saved_input!r} but the run config uses {cfg.input_mode!r}"
        )


# --- training -----------------------------------------------------------------------------


def apply_runtime_overrides(
    saved: WrenchPPOConfig,
    requested: WrenchPPOConfig,
    explicit: frozenset[str] | None = None,
) -> WrenchPPOConfig:
    """The resumed config: saved hyperparameters + the CLI's runtime resources.

    `--resume` continues with the hyperparameters stored in the checkpoint, but `backend` and
    `device` describe WHERE the run executes and must come from the CLI, otherwise a CPU
    checkpoint could never be continued on the GPU backend (and vice versa). The merged config
    is re-validated (the `backend="warp"` -> CUDA `device` rule must hold after the override).

    Args:
        saved: The config read from the checkpoint.
        requested: The config parsed from the CLI (defaults for flags not given).
        explicit: Names of the fields the user actually passed. `None` infers them by
            comparing `requested` with `WrenchPPOConfig()` defaults.

    Returns:
        The merged, validated config.
    """
    if explicit is None:
        defaults = WrenchPPOConfig()
        explicit = frozenset(
            field
            for field in RUNTIME_RESOURCE_FIELDS
            if getattr(requested, field) != getattr(defaults, field)
        )
    overrides = {
        field: getattr(requested, field) for field in RUNTIME_RESOURCE_FIELDS if field in explicit
    }
    if not overrides:
        return saved
    merged = dataclasses.replace(saved, **overrides)
    # Re-run __post_init__: the saved config may be "cpu" while the CLI asks for "warp".
    return WrenchPPOConfig(**dataclasses.asdict(merged))


def run(cfg: WrenchPPOConfig, *, explicit: frozenset[str] | None = None) -> Path:
    """Train (or continue with `cfg.resume`) and return the run directory.

    Args:
        cfg: The run config.
        explicit: Field names the caller passed explicitly (from the CLI parser); used by
            `--resume` for the runtime resources (`backend`, `device`).
    """
    requested = cfg

    # The policies are tiny MLPs; on a many-core server the default thread pool
    # oversubscribes and makes the env loop 30x slower (measured on 96-core RunPod).
    # One thread per training process; run seeds as separate processes.
    torch.set_num_threads(int(os.environ.get("WRENCH_TORCH_THREADS", "1")))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    resumed: dict | None = None
    if cfg.resume:
        run_dir, resumed = _resolve_resume(cfg.resume)
        cfg = apply_runtime_overrides(_config_from_payload(resumed, run_dir), requested, explicit)
        if int(resumed["update"]) >= cfg.total_updates:
            print(
                f"run {run_dir} is already complete (update {resumed['update']}/"
                f"{cfg.total_updates}, step {resumed['global_step']}); nothing to do"
            )
            return run_dir
        print(
            f"resuming {run_dir} from update {resumed['update']}/{cfg.total_updates} "
            f"(step {resumed['global_step']}); using the config saved in the checkpoint "
            "(other flags are ignored)",
            flush=True,
        )
    else:
        task_prefix = "wrench" if cfg.task == "cube" else f"coop_{cfg.task}"
        run_name = cfg.exp_name or f"{task_prefix}_{cfg.input_mode}_s{cfg.seed}_{int(time.time())}"
        run_dir = Path(cfg.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Backend selection is task-based: the cube keeps its original CPU/warp paths (the
    # environment variant follows the run's randomisation preset), while the coop suite has
    # only the CPU MuJoCo loop (`callosum.experiments.coop_tasks.vec_env`, the same contract).
    if cfg.task == "cube":
        env_config = cfg.env_config()
        if cfg.backend == "warp":
            # Lazy import: the GPU stack (mujoco_warp/warp) is only needed for this branch.
            from callosum.experiments.cube_wrench.env_gpu import WrenchCubeBatchGPU

            train_vec = WrenchCubeBatchGPU(
                cfg.num_envs,
                config=env_config,
                partner_mode=cfg.partner_mode,
                seed=cfg.seed,
                device=cfg.device,
            )
            eval_vec = WrenchCubeBatchGPU(
                cfg.eval_envs,
                config=env_config,
                partner_mode=cfg.partner_mode,
                seed=cfg.seed + 1_000_003,
                device=cfg.device,
            )
        else:
            from callosum.experiments.cube_wrench.vec_env import VecWrenchCube

            train_vec = VecWrenchCube(
                cfg.num_envs, cfg.partner_mode, config=env_config, seed=cfg.seed
            )
            eval_vec = VecWrenchCube(
                cfg.eval_envs, cfg.partner_mode, config=env_config, seed=cfg.seed + 1_000_003
            )
    else:
        from callosum.experiments.coop_tasks.vec_env import VecCoopEnv

        env_config = cfg.coop_env_config()
        train_vec = VecCoopEnv(
            cfg.task, cfg.num_envs, cfg.partner_mode, config=env_config, seed=cfg.seed
        )
        eval_vec = VecCoopEnv(
            cfg.task,
            cfg.eval_envs,
            cfg.partner_mode,
            config=env_config,
            seed=cfg.seed + 1_000_003,
        )
    try:
        return train(cfg, run_dir, train_vec, eval_vec, resumed)
    finally:
        train_vec.close()
        eval_vec.close()


def _resolve_resume(resume: str) -> tuple[Path, dict]:
    """`(run directory, checkpoint payload)` of a `--resume` argument (dir or `latest.pt`)."""
    path = Path(resume)
    if path.is_dir():
        run_dir = path
        checkpoint_path = run_dir / "latest.pt"
    elif path.is_file():
        checkpoint_path = path
        run_dir = path.parent
    else:
        raise FileNotFoundError(f"cannot resume {resume}: no such file or directory")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"cannot resume {run_dir}: {checkpoint_path} not found")
    return run_dir, load_checkpoint(checkpoint_path)


def _config_from_payload(payload: dict, run_dir: Path) -> WrenchPPOConfig:
    """The run config saved in a checkpoint, located in `run_dir` for `--resume`."""
    cfg = config_from_payload(payload)
    return dataclasses.replace(
        cfg, exp_name=run_dir.name, runs_dir=str(run_dir.parent), resume=str(run_dir)
    )


def train(
    cfg: WrenchPPOConfig,
    run_dir: Path,
    train_vec: object,
    eval_vec: object,
    resumed: dict | None = None,
) -> Path:
    """The update loop: rollout on `train_vec`, GAE, per-agent PPO, eval, checkpoint.

    `train_vec` / `eval_vec` are `callosum.experiments.cube_wrench.vec_env.VecWrenchCube` objects; they
    are typed loosely here so the training function itself does not import the simulator
    (only `run()` does). Under `predicted` every rollout step also builds the policy input
    `[own observation, ẑ_partner]` from the own-observation history, and every update takes
    one Bi-JEPA auxiliary step; the checkpoints then carry the module and its optimizer too.

    Args:
        cfg: The run config.
        run_dir: The run directory (`latest.pt`, `config.json`, TensorBoard events).
        train_vec: The training vec env (`cfg.num_envs` envs).
        eval_vec: The smaller vec env of the greedy evaluation (`cfg.eval_envs` envs).
        resumed: The checkpoint payload to continue, or `None` for a fresh run.

    Returns:
        `run_dir`.
    """
    from torch.utils.tensorboard import SummaryWriter

    device = torch.device(cfg.device)
    latent_dim = cfg.latent_dim if cfg.use_bijepa else 0
    agent_keys = cfg.agent_keys
    action_size = cfg.action_size
    policies = make_policies(
        cfg.partner_mode,
        device,
        latent_dim=latent_dim,
        input_size=cfg.env_obs_size,
        action_dim=action_size,
    )
    optimizers = [
        torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate, eps=1e-5) for policy in policies
    ]
    bijepa = make_bijepa(cfg, device)
    bijepa_optimizer = make_bijepa_optimizer(cfg, bijepa)
    start_update = 0
    eval_metrics: dict[str, float] = {}
    if resumed is not None:
        _check_resumed_state(resumed, cfg)
        load_weights(resumed, policies, agent_keys=agent_keys)
        load_optimizers(resumed, optimizers, agent_keys=agent_keys)
        if bijepa is not None and bijepa_optimizer is not None:
            load_bijepa_weights(resumed, bijepa)
            load_bijepa_optimizer(resumed, bijepa_optimizer)
        start_update = int(resumed["update"])
        eval_metrics = {key: float(value) for key, value in (resumed.get("eval") or {}).items()}

    if resumed is None:
        global_step = 0
        writer = SummaryWriter(str(run_dir))
        config_dict = dataclasses.asdict(cfg)
        config_dict.update(
            batch_size=cfg.batch_size,
            minibatch_size=cfg.minibatch_size,
            total_steps=cfg.total_steps,
            obs_size=cfg.obs_size,
            env_obs_size=cfg.env_obs_size,
            policy_input_size=cfg.policy_input_size,
            input_mode=cfg.input_mode,
        )
        (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2))
    else:
        global_step = int(resumed["global_step"])
        # Discard the events the killed process logged after the checkpoint (they have steps
        # >= purge_step), so the resumed curves continue cleanly.
        writer = SummaryWriter(str(run_dir), purge_step=global_step + 1)
        _restore_rng(resumed["rng"])
    logger = MetricLogger(writer)

    if hasattr(train_vec, "software_versions"):
        print(f"env software: {train_vec.software_versions()}", flush=True)
    env_label = "CubeWrench-v0" if cfg.task == "cube" else f"coop:{cfg.task}"
    print(
        f"run {run_dir} | {env_label} | input_mode {cfg.input_mode} (env partner_mode "
        f"{cfg.partner_mode}) | backend {cfg.backend} | randomization {cfg.randomization} | "
        f"device {device} | torch {torch.__version__} | obs "
        f"{cfg.env_obs_size} + latent {latent_dim} = policy input "
        f"{cfg.policy_input_size} -> act {action_size}",
        flush=True,
    )
    if bijepa is not None:
        print(
            f"Bi-JEPA: latent_dim {cfg.latent_dim} | history_len {cfg.history_len} | "
            f"aux_coef {cfg.aux_coef} | ema_momentum {cfg.ema_momentum} | "
            f"var_weight {cfg.var_weight} | cov_weight {cfg.cov_weight} | "
            f"predictor {cfg.bijepa_config().history_aggregator}",
            flush=True,
        )
    print(
        f"envs {cfg.num_envs} x steps {cfg.rollout_steps} = batch {cfg.batch_size} "
        f"(minibatch {cfg.minibatch_size}, {cfg.update_epochs} epochs) | "
        f"{cfg.total_updates} updates = {cfg.total_steps} steps | "
        f"episode {train_vec.config.max_control_steps} steps, eval {cfg.eval_episodes} "
        f"episodes in {cfg.eval_envs} envs every {cfg.eval_every} updates",
        flush=True,
    )

    steps = cfg.rollout_steps
    num_envs = cfg.num_envs
    torch_io = bool(getattr(train_vec, "torch_io", False))
    obs_dim = cfg.policy_input_size
    obs_buf = [torch.zeros((steps, num_envs, obs_dim), device=device) for _ in policies]
    act_buf = [torch.zeros((steps, num_envs, action_size), device=device) for _ in policies]
    logprob_buf = [torch.zeros((steps, num_envs), device=device) for _ in policies]
    value_buf = [torch.zeros((steps, num_envs), device=device) for _ in policies]
    rewards = torch.zeros((steps, num_envs), device=device)
    dones = torch.zeros((steps, num_envs), device=device)
    truncated_buf = torch.zeros((steps, num_envs), device=device)
    # Own-observation history of each role, `(rollout_steps, num_envs, history_len, obs_dim)`;
    # the aux loss reconstructs the predictor windows (with the episode resets of the rollout).
    obs_hist_buf = (
        [
            torch.zeros((steps, num_envs, cfg.history_len, cfg.env_obs_size), device=device)
            for _ in agent_keys
        ]
        if bijepa is not None
        else None
    )
    builder = (
        LatentInputBuilder(bijepa, device, agent_keys=agent_keys) if bijepa is not None else None
    )

    stop_requested = False

    def request_stop(signum: int, frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"signal {signum}: saving a checkpoint and stopping after this update", flush=True)

    signal.signal(signal.SIGTERM, request_stop)

    # The env state cannot be saved, so a resumed run re-seeds its resets from the resumed
    # update: the initial states are not a replay of the original start (as in the IPPO
    # trainer).
    # The Bi-JEPA history window is part of the (unsaved) episode state too, so a resumed run
    # seeds it from its reset observation; as with the env itself, the first rollout after a
    # resume is not a replay of the original start.
    reset_seed = cfg.seed + 1_000_003 * start_update
    obs_t = _to_torch(train_vec.reset(seed=reset_seed), device)
    inputs_t = builder.reset(obs_t) if builder is not None else obs_t
    next_done = torch.zeros(num_envs, device=device)
    start_time = time.time()
    start_global_step = global_step
    update = start_update
    for update in range(start_update + 1, cfg.total_updates + 1):
        update_start = time.time()
        final_values = [torch.zeros((steps, num_envs), device=device) for _ in policies]
        finished_returns: list[float] = []
        finished_lengths: list[int] = []
        finished_successes: list[bool] = []

        # --- Rollout -------------------------------------------------------------------
        # TODO(review): the loop is covered by the toy-vec test only; the first server run must
        # confirm the throughput and that the truncation bootstrap sees the real env's
        # `final_obs` (the success/failure terminals must not bootstrap).
        for step in range(steps):
            dones[step] = next_done
            env_actions: dict[str, np.ndarray] = {}
            with torch.no_grad():
                for index, (name, policy) in enumerate(zip(agent_keys, policies)):
                    obs_buf[index][step] = inputs_t[name]
                    if obs_hist_buf is not None and builder is not None:
                        obs_hist_buf[index][step] = builder.history[name]
                    action, logprob, _, value = policy.get_action_and_value(inputs_t[name])
                    act_buf[index][step] = action
                    logprob_buf[index][step] = logprob
                    value_buf[index][step] = value.flatten()
                    # The stored action is the unclipped sample the log-prob refers to; the
                    # env receives the clipped wrench (its action box is [-1, 1]). The GPU
                    # backend consumes the CUDA tensor directly (no host round trip).
                    clipped = torch.clamp(action, -1.0, 1.0)
                    env_actions[name] = clipped if torch_io else clipped.cpu().numpy()
            result = train_vec.step(env_actions)
            obs_t = _to_torch(result.obs, device)
            rewards[step] = torch.as_tensor(result.reward, device=device, dtype=torch.float32)
            truncated = torch.as_tensor(result.truncated, device=device)
            terminated = torch.as_tensor(result.terminated, device=device)
            truncated_buf[step] = truncated.float()
            if bool(truncated.any()):
                # Truncations are the only episodes that bootstrap: their true last
                # observation is `result.final_obs` (the auto-reset already replaced it in
                # `result.obs`). Under `predicted` its policy input rolls the own-observation
                # history one more step with that final observation.
                final_obs_t = _to_torch(result.final_obs, device)
                with torch.no_grad():
                    final_inputs = (
                        builder.history_inputs(final_obs_t) if builder is not None else final_obs_t
                    )
                    for index, (name, policy) in enumerate(zip(agent_keys, policies)):
                        final_values[index][step, truncated] = policy.get_value(
                            final_inputs[name][truncated]
                        ).view(-1)
            next_done = (terminated | truncated).float()
            if builder is not None:
                inputs_t = builder.step(obs_t, next_done)
            else:
                inputs_t = obs_t
            finished_returns.extend(result.finished_returns)
            finished_lengths.extend(result.finished_lengths)
            finished_successes.extend(result.finished_successes)
        global_step += cfg.batch_size

        # --- Advantages and per-policy PPO updates -------------------------------------
        losses: list[dict[str, float]] = []
        for index, (name, policy) in enumerate(zip(agent_keys, policies)):
            with torch.no_grad():
                next_value = policy.get_value(inputs_t[name]).reshape(1, -1)
                advantages, returns = compute_advantages(
                    rewards=rewards,
                    values=value_buf[index],
                    dones=dones,
                    truncated=truncated_buf,
                    final_values=final_values[index],
                    next_value=next_value,
                    next_done=next_done,
                    gamma=cfg.gamma,
                    gae_lambda=cfg.gae_lambda,
                )
            batch = {
                "obs": obs_buf[index].reshape(-1, obs_dim),
                "actions": act_buf[index].reshape(-1, action_size),
                "logprobs": logprob_buf[index].reshape(-1),
                "advantages": advantages.reshape(-1),
                "returns": returns.reshape(-1),
                "values": value_buf[index].reshape(-1),
            }
            policy.train()
            metrics = ppo_step(policy, optimizers[index], batch, cfg)
            logger.log_many(metrics, global_step, prefix=f"losses/{name}/")
            logger.log(
                f"policy/{name}/action_std",
                policy.actor_logstd.exp().mean().item(),
                global_step,
            )
            losses.append(metrics)

        # --- Bi-JEPA auxiliary update (`predicted`) ------------------------------------
        aux_metrics: dict[str, float] = {}
        if bijepa is not None and bijepa_optimizer is not None and obs_hist_buf is not None:
            hist_a = obs_hist_buf[0].reshape(-1, cfg.history_len, cfg.env_obs_size)
            hist_b = obs_hist_buf[1].reshape(-1, cfg.history_len, cfg.env_obs_size)
            aux_metrics = aux_update(bijepa, bijepa_optimizer, cfg, hist_a, hist_b)
            for tag, value in aux_metrics.items():
                logger.log(tag, value, global_step)

        # --- Logging, evaluation, checkpoint -------------------------------------------
        if finished_returns:
            logger.log("train/return", float(np.mean(finished_returns)), global_step)
            logger.log("train/success_rate", float(np.mean(finished_successes)), global_step)
            logger.log("train/episode_len", float(np.mean(finished_lengths)), global_step)
            logger.log("train/episodes", len(finished_returns), global_step)
        sps = (global_step - start_global_step) / max(time.time() - start_time, 1e-9)
        logger.log("charts/SPS", sps, global_step)
        last = update == cfg.total_updates
        if update % cfg.eval_every == 0 or last:
            summary = evaluate_greedy(
                policies,
                eval_vec,
                cfg.eval_episodes,
                device=device,
                seed=cfg.seed + EVAL_SEED_OFFSET + 1_000_003 * update,
                bijepa=bijepa,
                agent_keys=agent_keys,
            )
            eval_metrics = {
                "success_rate": summary.success_rate,
                "return": summary.mean_return,
                "angle_deg": summary.mean_angle_deg,
                "episode_len": summary.mean_length,
            }
            logger.log_many(eval_metrics, global_step, prefix="eval/")
            print(
                f"eval @ update {update}: {summary.episodes} episodes | "
                f"success_rate={summary.success_rate:.3f} return={summary.mean_return:.3f} "
                f"angle_deg={summary.mean_angle_deg:.1f} "
                f"len={summary.mean_length:.1f}",
                flush=True,
            )
        if update % cfg.checkpoint_every == 0 or last or stop_requested:
            save_checkpoint(
                run_dir / "latest.pt",
                policies,
                optimizers,
                cfg,
                update,
                global_step,
                eval_metrics,
                bijepa=bijepa,
                bijepa_optimizer=bijepa_optimizer,
                agent_keys=agent_keys,
            )
            print(f"saved {run_dir / 'latest.pt'} (update {update})", flush=True)

        if finished_returns:
            shown = (
                f"return {np.mean(finished_returns):.3f} "
                f"success {np.mean(finished_successes):.3f} "
                f"len {np.mean(finished_lengths):.1f}"
            )
        else:
            shown = "no finished episode yet"
        aux_shown = ""
        if aux_metrics:
            aux_shown = (
                f" | aux {aux_metrics['aux/loss']:.3f} mse {aux_metrics['aux/mse']:.3f} "
                f"probe_r2 {aux_metrics['aux/probe/wrench_r2']:.2f}"
            )
        print(
            f"update {update}/{cfg.total_updates} | step {global_step} | {sps:.0f} sps | "
            f"{len(finished_returns)} eps | {shown} | "
            f"policy_loss {losses[0]['policy_loss']:.3f}/{losses[1]['policy_loss']:.3f}"
            f"{aux_shown} | {time.time() - update_start:.1f}s",
            flush=True,
        )
        writer.flush()
        if stop_requested:
            break

    total_seconds = time.time() - start_time
    print(
        f"done: {update} updates ({global_step - start_global_step} steps of this process) in "
        f"{total_seconds:.0f}s | last eval {eval_metrics or 'none'}",
        flush=True,
    )
    writer.close()
    return run_dir


def _to_torch(
    obs: dict[str, np.ndarray | torch.Tensor], device: torch.device | str
) -> dict[str, torch.Tensor]:
    """The vec env's observation dict as float32 tensors on `device`.

    The CPU `VecWrenchCube` returns numpy arrays, the GPU `WrenchCubeBatchGPU` already CUDA
    tensors; `Tensor.to` is a no-op when the device/dtype already match, so this is a copy
    only for the CPU backend.
    """
    return {
        key: (
            value.to(device=device, dtype=torch.float32)
            if torch.is_tensor(value)
            else torch.as_tensor(value, device=device, dtype=torch.float32)
        )
        for key, value in obs.items()
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: `python -m callosum.experiments.cube_wrench.train [flags]`.

    A detached run has its stdout redirected to a file, so flush every line.
    """
    sys.stdout.reconfigure(line_buffering=True)
    cfg, explicit = parse_args_with_explicit(argv)
    run(cfg, explicit=explicit)


if __name__ == "__main__":
    main()
