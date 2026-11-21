"""Continuous-wrench IPPO trainer for CubeWrench-v0 (PLAN.md step 2, `wrench_ppo`).

Two independent PPO learners -- one Gaussian actor-critic per role (`holder`, `rotator`) --
share one environment step and one **team** reward, so the coordination has to be learned
from the decentralised observations alone (the `none` / `oracle` ablation of the plan).
The env (`callosum.envs.wrench_cube.WrenchCubeEnv`) is single-instance MuJoCo on the CPU,
so the vectorisation is a Python loop: `callosum.training._vec_wrench.VecWrenchCube`
drives `num_envs` copies with auto-reset and the finished-episode statistics.

Reused from `callosum.training._ppo_core` (the ManiSkill PPO baseline's blocks, shared with
the arms IPPO trainer): `ActorCritic` (3 x 256 tanh MLPs, state-independent log-std),
`compute_gae` and the clipped-surrogate `ppo_update`. What this trainer adds is the
wrench-specific contract around them:

* the policy input is the env's flat observation, width `obs_size(partner_mode)` (26 for
  `none`, 35 for `oracle`), and the action is the normalized 6D wrench of the role;
* GAE bootstraps only on time-limit truncations: success and failure are true terminals,
  so the shared success bonus is not fed back through `V(final observation)`
  (`compute_advantages`);
* one update = `num_envs * rollout_steps` transitions of the loop vector env, then a PPO
  update of each policy on its own rollout batch (the reward and GAE inputs are shared,
  the value/advantage streams are per agent);
* logs `train/return`, `train/success_rate`, `train/episode_len` over the episodes that
  finished in the rollout, greedy `eval/*` every `eval_every` updates, and writes full
  `latest.pt` checkpoints (weights + Adam moments + config + counters + torch RNG) that
  `--resume` restores.

The module deliberately imports torch but not mujoco at module level: the vec env is
imported inside `run()`, so `tests/test_wrench_ppo.py` (config validation, GAE, policy
shapes, one PPO step, checkpoint roundtrip) runs on macOS/CI without the `sandbox` extra.

Typical use (RunPod GPU pod, MuJoCo on the CPU; CPU simulation also works on a laptop but
the plan forbids training there):

    python -m callosum.training.wrench_ppo --partner-mode oracle --seed 0 \
        --exp-name wrench_oracle_s0 --total-updates 300

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

from callosum.configs.wrench_ppo import WrenchPPOConfig, parse_args
from callosum.envs._wrench_logic import AGENT_KEYS, obs_size
from callosum.training._metrics import MetricLogger
from callosum.training._ppo_core import ActorCritic, compute_gae, ppo_update

CHECKPOINT_FORMAT = 1
"""Format of the trainer's checkpoints: weights, optimizer states, config and counters."""

_WRENCH_SIZE = 6
"""Width of one agent's action (a normalized 6D wrench)."""


# --- policies -----------------------------------------------------------------------------


def make_policies(partner_mode: str, device: torch.device | str = "cpu") -> list[ActorCritic]:
    """One independent Gaussian actor-critic per role, in `AGENT_KEYS` order.

    The input width is `callosum.envs._wrench_logic.obs_size(partner_mode)` -- the env's
    flat observation for that mode (26 for `none`, 35 for `oracle`) -- and the output is
    the role's normalized 6D wrench. Sharing no parameters is what makes the pair an
    IPPO-style decentralised team rather than a single joint policy.

    Args:
        partner_mode: Env partner-visibility mode fixing the observation width.
        device: Torch device of both policies.

    Returns:
        `[holder_policy, rotator_policy]`.
    """
    input_size = obs_size(partner_mode)
    return [ActorCritic(input_size, _WRENCH_SIZE).to(device) for _ in AGENT_KEYS]


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
) -> EvalSummary:
    """Run the deterministic policies in `vec` until `episodes` episodes finish.

    The actions are the distribution means (clipped to the `[-1, 1]` wrench box) -- the
    greedy evaluation the plan asks for -- and the env is the same `CubeWrench-v0` with the
    same randomisation as training, so the number is comparable with `train/*`.

    Args:
        policies: `[holder_policy, rotator_policy]`.
        vec: A `callosum.training._vec_wrench.VecWrenchCube`-like object (`reset`, `step`,
            `num_envs`, `config.max_control_steps`); typed loosely so this function needs
            no simulator import.
        episodes: Number of finished episodes to collect.
        device: Torch device of the policies.
        seed: Base episode seed of the evaluation.

    Returns:
        The aggregated `EvalSummary`.
    """
    obs = _to_torch(vec.reset(seed=seed), device)
    max_steps = (episodes // int(vec.num_envs) + 2) * int(vec.config.max_control_steps)
    successes: list[bool] = []
    returns: list[float] = []
    lengths: list[int] = []
    angles: list[float] = []
    steps = 0
    while len(successes) < episodes:
        if steps >= max_steps:
            raise RuntimeError(
                f"evaluation did not finish {episodes} episodes in {max_steps} steps "
                f"({len(successes)} finished); the env's episode budget is broken"
            )
        actions = {}
        for name, policy in zip(AGENT_KEYS, policies):
            action = policy.get_action(obs[name], deterministic=True)
            actions[name] = torch.clamp(action, -1.0, 1.0).cpu().numpy()
        result = vec.step(actions)
        obs = _to_torch(result.obs, device)
        successes.extend(result.finished_successes)
        returns.extend(result.finished_returns)
        lengths.extend(result.finished_lengths)
        angles.extend(result.finished_angles)
        steps += 1
    count = episodes
    return EvalSummary(
        episodes=count,
        success_rate=float(np.mean(successes[:count])),
        mean_return=float(np.mean(returns[:count])),
        mean_angle_deg=float(np.mean(angles[:count])),
        mean_length=float(np.mean(lengths[:count])),
    )


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate of one greedy evaluation."""

    episodes: int
    success_rate: float
    mean_return: float
    mean_angle_deg: float
    mean_length: float


# --- checkpoints --------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    policies: list[ActorCritic],
    optimizers: list[torch.optim.Optimizer],
    cfg: WrenchPPOConfig,
    update: int,
    global_step: int,
    eval_metrics: dict[str, float] | None = None,
) -> None:
    """Write the full training state to `path` atomically.

    The payload holds both policies' weights, both Adam states, the config (so `--resume`
    cannot run with drifted hyperparameters), the update / environment-step counters and the
    torch RNG state. A kill mid-write cannot corrupt the file: it is written to
    `<path>.tmp` and renamed.

    Args:
        path: Destination file (normally `<run dir>/latest.pt`).
        policies: `[holder_policy, rotator_policy]`.
        optimizers: Their optimizers, in the same order.
        cfg: The run config.
        update: Number of completed updates (1-based).
        global_step: Environment steps consumed so far.
        eval_metrics: The latest greedy evaluation, for the checkpoint's `eval` block.
    """
    payload = {
        "format": CHECKPOINT_FORMAT,
        "agents": {name: policy.state_dict() for name, policy in zip(AGENT_KEYS, policies)},
        "optimizers": {
            name: optimizer.state_dict() for name, optimizer in zip(AGENT_KEYS, optimizers)
        },
        "config": dataclasses.asdict(cfg),
        "obs_size": cfg.obs_size,
        "partner_mode": cfg.partner_mode,
        "update": int(update),
        "global_step": int(global_step),
        "eval": {key: float(value) for key, value in (eval_metrics or {}).items()},
        "rng": {"torch": torch.get_rng_state()},
    }
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


def load_weights(payload: dict, policies: list[ActorCritic]) -> None:
    """Load both policies' weights from a checkpoint payload (in `AGENT_KEYS` order)."""
    for name, policy in zip(AGENT_KEYS, policies):
        policy.load_state_dict(payload["agents"][name])


def load_optimizers(payload: dict, optimizers: list[torch.optim.Optimizer]) -> None:
    """Load both optimizer states from a checkpoint payload (in `AGENT_KEYS` order)."""
    for name, optimizer in zip(AGENT_KEYS, optimizers):
        optimizer.load_state_dict(payload["optimizers"][name])


# --- training -----------------------------------------------------------------------------


def run(cfg: WrenchPPOConfig) -> Path:
    """Train (or continue with `cfg.resume`) and return the run directory."""
    from callosum.training._vec_wrench import VecWrenchCube

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    resumed: dict | None = None
    if cfg.resume:
        run_dir, resumed = _resolve_resume(cfg.resume)
        cfg = _config_from_payload(resumed, run_dir)
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
        run_name = cfg.exp_name or f"wrench_{cfg.partner_mode}_s{cfg.seed}_{int(time.time())}"
        run_dir = Path(cfg.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_vec = VecWrenchCube(cfg.num_envs, cfg.partner_mode, seed=cfg.seed)
    eval_vec = VecWrenchCube(cfg.eval_envs, cfg.partner_mode, seed=cfg.seed + 1_000_003)
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
    field_names = {field.name for field in dataclasses.fields(WrenchPPOConfig)}
    values = {key: value for key, value in payload["config"].items() if key in field_names}
    values.update(exp_name=run_dir.name, runs_dir=str(run_dir.parent), resume=str(run_dir))
    return WrenchPPOConfig(**values)


def train(
    cfg: WrenchPPOConfig,
    run_dir: Path,
    train_vec: object,
    eval_vec: object,
    resumed: dict | None = None,
) -> Path:
    """The update loop: rollout on `train_vec`, GAE, per-agent PPO, eval, checkpoint.

    `train_vec` / `eval_vec` are `callosum.training._vec_wrench.VecWrenchCube` objects; they
    are typed loosely here so the training function itself does not import the simulator
    (only `run()` does).

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
    policies = make_policies(cfg.partner_mode, device)
    optimizers = [
        torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate, eps=1e-5) for policy in policies
    ]
    start_update = 0
    eval_metrics: dict[str, float] = {}
    if resumed is not None:
        load_weights(resumed, policies)
        load_optimizers(resumed, optimizers)
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
        )
        (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2))
    else:
        global_step = int(resumed["global_step"])
        # Discard the events the killed process logged after the checkpoint (they have steps
        # >= purge_step), so the resumed curves continue cleanly.
        writer = SummaryWriter(str(run_dir), purge_step=global_step + 1)
        torch.set_rng_state(resumed["rng"]["torch"].cpu())
    logger = MetricLogger(writer)

    print(
        f"run {run_dir} | CubeWrench-v0 | partner_mode {cfg.partner_mode} | device {device} | "
        f"obs {cfg.obs_size} -> act {_WRENCH_SIZE}",
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
    obs_dim = cfg.obs_size
    obs_buf = [torch.zeros((steps, num_envs, obs_dim), device=device) for _ in policies]
    act_buf = [torch.zeros((steps, num_envs, _WRENCH_SIZE), device=device) for _ in policies]
    logprob_buf = [torch.zeros((steps, num_envs), device=device) for _ in policies]
    value_buf = [torch.zeros((steps, num_envs), device=device) for _ in policies]
    rewards = torch.zeros((steps, num_envs), device=device)
    dones = torch.zeros((steps, num_envs), device=device)
    truncated_buf = torch.zeros((steps, num_envs), device=device)

    stop_requested = False

    def request_stop(signum: int, frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"signal {signum}: saving a checkpoint and stopping after this update", flush=True)

    signal.signal(signal.SIGTERM, request_stop)

    # The env state cannot be saved, so a resumed run re-seeds its resets from the resumed
    # update: the initial states are not a replay of the original start (as in the IPPO
    # trainer).
    reset_seed = cfg.seed + 1_000_003 * start_update
    obs_t = _to_torch(train_vec.reset(seed=reset_seed), device)
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
                for index, (name, policy) in enumerate(zip(AGENT_KEYS, policies)):
                    obs_buf[index][step] = obs_t[name]
                    action, logprob, _, value = policy.get_action_and_value(obs_t[name])
                    act_buf[index][step] = action
                    logprob_buf[index][step] = logprob
                    value_buf[index][step] = value.flatten()
                    # The stored action is the unclipped sample the log-prob refers to; the
                    # env receives the clipped wrench (its action box is [-1, 1]).
                    env_actions[name] = torch.clamp(action, -1.0, 1.0).cpu().numpy()
            result = train_vec.step(env_actions)
            obs_t = _to_torch(result.obs, device)
            rewards[step] = torch.as_tensor(result.reward, device=device, dtype=torch.float32)
            truncated = torch.as_tensor(result.truncated, device=device)
            terminated = torch.as_tensor(result.terminated, device=device)
            truncated_buf[step] = truncated.float()
            if bool(truncated.any()):
                # Truncations are the only episodes that bootstrap: their true last
                # observation is `result.final_obs` (the auto-reset already replaced it in
                # `result.obs`).
                final_obs_t = _to_torch(result.final_obs, device)
                with torch.no_grad():
                    for index, (name, policy) in enumerate(zip(AGENT_KEYS, policies)):
                        final_values[index][step, truncated] = policy.get_value(
                            final_obs_t[name][truncated]
                        ).view(-1)
            next_done = (terminated | truncated).float()
            finished_returns.extend(result.finished_returns)
            finished_lengths.extend(result.finished_lengths)
            finished_successes.extend(result.finished_successes)
        global_step += cfg.batch_size

        # --- Advantages and per-policy PPO updates -------------------------------------
        losses: list[dict[str, float]] = []
        for index, (name, policy) in enumerate(zip(AGENT_KEYS, policies)):
            with torch.no_grad():
                next_value = policy.get_value(obs_t[name]).reshape(1, -1)
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
                "actions": act_buf[index].reshape(-1, _WRENCH_SIZE),
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
                seed=cfg.seed + 1_000_003 * update,
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
        print(
            f"update {update}/{cfg.total_updates} | step {global_step} | {sps:.0f} sps | "
            f"{len(finished_returns)} eps | {shown} | "
            f"policy_loss {losses[0]['policy_loss']:.3f}/{losses[1]['policy_loss']:.3f} "
            f"| {time.time() - update_start:.1f}s",
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


def _to_torch(obs: dict[str, np.ndarray], device: torch.device | str) -> dict[str, torch.Tensor]:
    """Convert the vec env's numpy observation dict to float32 tensors on `device`."""
    return {
        key: torch.as_tensor(value, device=device, dtype=torch.float32)
        for key, value in obs.items()
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: `python -m callosum.training.wrench_ppo [flags]`.

    A detached run has its stdout redirected to a file, so flush every line.
    """
    sys.stdout.reconfigure(line_buffering=True)
    run(parse_args(argv))


if __name__ == "__main__":
    main()
