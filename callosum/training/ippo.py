"""IPPO for the two-arm envs: two independent PPO learners on one shared GPU simulation.

Run as `python -m callosum.training.ippo --env-id FaceTurn-v0 ...` (flags: `--help`, defaults:
`callosum.configs.ippo.IPPOConfig`).

Based on ManiSkill's single-file GPU PPO (`examples/baselines/ppo/ppo.py` @ v3.0.1): same
rollout/GAE/update structure, `ManiSkillVectorEnv` with partial resets and `record_metrics`,
`final_observation` bootstrapping, and the same network and hyperparameter style. What differs:

* Two actor-critics, one per arm (`so101_pg-0` = agent_a = holder, `so101_pg-1` = agent_b =
  rotator), each with its own optimizer. They only share the environment step.
* The env's actions are a dict `{uid: (num_envs, 6)}` (5 arm joint deltas + 1 absolute gripper
  target), so every agent samples its own 6-D action.
* `obs_mode="state"` yields one flat tensor for both agents; `_agent_obs.AgentObsBuilder` cuts
  out each agent's own policy input (own proprioception, task state, TCP poses as allowed by
  `partner_obs`). Each critic sees the same input as its own actor (decentralised, no CTDE).
* Rewards: both agents are trained on the env's one shared reward, the *normalised dense* reward
  (`reward_mode="normalized_dense"`, as in the ManiSkill baseline). Per-agent credit assignment
  is a later concern; the roles are encoded in the env's reward terms, not split per learner.
* Success is a true terminal: only time-limit truncations bootstrap from the final observation
  (the baseline bootstraps terminations too), see FaceTurnRewardConfig.success_bonus.
* The learning rate decays linearly to 0 over the run by default (`anneal_lr`).
* Logging to TensorBoard (`runs/<run_name>/`) and one compact line per iteration on stdout;
  checkpoints `latest.pt` and `best.pt` (by evaluation success) in the same directory. Both carry
  the full training state (weights, Adam moments, RNG states, counters, best-eval score), see
  `_checkpoint`.
* Per-env diagnostics: every `diag_*` key of the env's `info` is averaged over the rollout
  (`diag/<name>`) and each evaluation (`eval_diag/<name>`), see `_diag`; envs whose GPU state goes
  non-finite are reset individually and made terminal with zero reward, see `_nonfinite`.
* `--resume <run dir>` continues a killed run exactly where its `latest.pt` stopped (same
  directory, LR schedule position, optimizer state, counters, best-eval tracking); the simulator
  state cannot be saved, so the envs are reset with seeds derived from the resumed iteration.
  `--checkpoint <file>` is the weights-only warm start of a new run.
* Fine-tuning from demonstrations (IPPO from scratch never finds a grasp on FaceTurn-v0, see
  `callosum.training.bc`): `--checkpoint <bc.pt>` warm-starts from a BC checkpoint (input widths,
  field names and `partner_obs` are checked), `--critic-warmup-iters K` updates only the critics
  for the first K iterations (actors frozen), and `--demos <file> --bc-coef C --bc-decay-iters N`
  add a DAPG-style auxiliary loss `C * mse(actor_mean(demo_obs), demo_action)` to every PPO
  minibatch step, decaying linearly to 0 over N iterations (`bc_loss` / `charts/bc_coef` are
  logged). Both schedules depend only on the iteration, so `--resume` continues them.
  `--eval-only --checkpoint <file> [--eval-repeats N]` evaluates a checkpoint's deterministic
  policies without training and writes `eval.json` (how a BC policy is judged).

Envs are always created with `obs_mode="state", render_backend="none"`: the lab server only has a
software Vulkan device, and the default render device cannot be created there.
"""

import dataclasses
import json
import signal
import sys
import time
from pathlib import Path
from typing import NamedTuple

import gymnasium as gym
import mani_skill.envs  # noqa: F401  (registers the stock ManiSkill envs)
import numpy as np
import torch
from mani_skill.utils import gym_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.utils.tensorboard import SummaryWriter

import callosum.envs.face_turn  # noqa: F401  (registers TwoSO101-v0 and FaceTurn-v0)
from callosum.configs.ippo import (
    IPPOConfig,
    bc_coef_at,
    critic_warmup_active,
    env_reset_seeds,
    learning_rate_at,
    parse_args,
    resolve_episode_length,
)
from callosum.envs._sim_compat import prepare_sim_backend
from callosum.training._agent_obs import AgentObsBuilder
from callosum.training._checkpoint import (
    TrainingState,
    check_warm_start_compat,
    load_agent_weights,
    load_checkpoint,
    load_training_state,
    save_checkpoint,
)
from callosum.training._demos import agent_demo_tensors, check_layout_matches, load_demos
from callosum.training._diag import DiagStats
from callosum.training._metrics import (
    EpisodeStats,
    MetricLogger,
    eval_diag_summary,
    progress_line,
)
from callosum.training._nonfinite import NonFiniteTracker, guard_step
from callosum.training._ppo_core import ActorCritic, compute_gae, ppo_update

AGENT_NAMES = ("agent_a", "agent_b")


def make_envs(cfg: IPPOConfig, num_envs: int, ignore_terminations: bool) -> ManiSkillVectorEnv:
    """The state-observation vector env: `gym.make` plus `ManiSkillVectorEnv(record_metrics)`.

    `ignore_terminations=False` lets envs reset on success (training); `True` makes every episode
    last the full `max_episode_steps` (evaluation, which then also reports `success_at_end`).
    """
    # TODO(review): verified on the Mac CPU sim (1 env) only; the GPU backend (many envs, two
    # vector envs in one process as in the ManiSkill baseline) is unverified until the server run.
    prepare_sim_backend(cfg.sim_backend)
    kwargs = {}
    if cfg.control_mode is not None:
        kwargs["control_mode"] = cfg.control_mode
    if cfg.max_episode_steps is not None:
        kwargs["max_episode_steps"] = cfg.max_episode_steps
    env = gym.make(
        cfg.env_id,
        num_envs=num_envs,
        obs_mode="state",
        reward_mode=cfg.reward_mode,
        sim_backend=cfg.sim_backend,
        render_backend="none",
        render_mode=None,
        partner_obs=cfg.partner_obs,
        # No per-episode geometry randomisation here, so never rebuild the scene on reset.
        reconfiguration_freq=0,
        **kwargs,
    )
    return ManiSkillVectorEnv(
        env, num_envs, ignore_terminations=ignore_terminations, record_metrics=True
    )


def make_obs_builder(envs: ManiSkillVectorEnv, flat_obs: torch.Tensor, cfg: IPPOConfig):
    """The per-agent observation builder, verified against the env's structured observation."""
    structured = envs.unwrapped.get_obs(unflattened=True)
    return AgentObsBuilder.from_env_obs(structured, flat_obs, cfg.partner_obs, agent_uids_of(envs))


def agent_uids_of(envs: ManiSkillVectorEnv) -> tuple[str, str]:
    """The two agent uids (agent_a first) from the env's dict action space."""
    uids = tuple(envs.single_action_space.spaces)
    if len(uids) != 2:
        raise ValueError(f"IPPO expects a two-agent env, got action space keys {uids}")
    return uids


class EvalResult(NamedTuple):
    """Outcome of `evaluate`.

    Attributes:
        metrics: mean of each episode metric over the finished episodes.
        episodes: how many episodes finished (envs that went non-finite are not counted).
        diag_means / diag_maxes: `DiagStats.take()` of the env's `diag_*` info keys over the
            whole evaluation (empty for envs without diagnostics).
        nonfinite: envs that had a non-finite state and were dropped from the evaluation.
    """

    metrics: dict[str, float]
    episodes: int
    diag_means: dict[str, float]
    diag_maxes: dict[str, float]
    nonfinite: int


@torch.no_grad()
def evaluate(
    agents: list[ActorCritic],
    builder: AgentObsBuilder,
    eval_envs: ManiSkillVectorEnv,
    action_bounds: list[tuple[torch.Tensor, torch.Tensor]],
    num_steps: int,
    tracker: NonFiniteTracker | None = None,
    seed: int | None = None,
) -> EvalResult:
    """Run `num_steps` deterministic steps in the eval envs; metric means over finished episodes.

    An env whose observation or reward goes non-finite is reset (so the policy never sees NaN)
    and excluded from the statistics for the rest of the evaluation. `seed` seeds the reset of the
    eval envs (`None`: continue the envs' own random stream).
    """
    uids = builder.agent_uids
    tracker = tracker or NonFiniteTracker()
    obs, _ = eval_envs.reset(seed=seed)
    stats = EpisodeStats()
    diag = DiagStats(eval_envs.num_envs, eval_envs.device)
    dead: torch.Tensor | None = None  # envs dropped from the statistics
    for step in range(num_steps):
        inputs = builder(obs)
        actions = {
            uid: torch.clamp(agent.get_action(x, deterministic=True), low, high)
            for uid, agent, x, (low, high) in zip(uids, agents, inputs, action_bounds, strict=True)
        }
        obs, reward, _, _, infos = eval_envs.step(actions)
        guard = guard_step(
            eval_envs,
            obs,
            reward,
            infos,
            layout=builder.layout,
            actions=actions,
            describe=not tracker.reported,
        )
        if guard is not None:
            obs = guard.obs
            dead = guard.poisoned if dead is None else dead | guard.poisoned
            tracker.record("eval", guard, f"evaluation step {step}")
        stats.add(infos, exclude=dead)
        diag.add(infos, None if dead is None else ~dead)
    diag_means, diag_maxes = diag.take()
    nonfinite = 0 if dead is None else int(dead.sum())
    return EvalResult(stats.means(), stats.num_episodes, diag_means, diag_maxes, nonfinite)


def _without_reward_terms(maxes: dict[str, float]) -> dict[str, float]:
    """`maxes` without the per-reward-term entries (`r_*`): their maxima are TensorBoard clutter."""
    return {k: v for k, v in maxes.items() if not k.startswith("r_")}


def run(cfg: IPPOConfig) -> Path:
    """Train (or, with `cfg.resume`, continue a run); returns the run directory."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.eval_only:
        return run_eval_only(cfg)

    resume_payload = None
    if cfg.resume:
        run_dir = Path(cfg.resume)
        latest = run_dir / "latest.pt"
        if not latest.is_file():
            raise FileNotFoundError(f"cannot resume {run_dir}: {latest} not found")
        resume_payload = load_checkpoint(latest)
        if resume_payload.get("format") != 2:
            # Same error as load_training_state, but before the (slow) env creation.
            load_training_state(resume_payload, [], [])
        if resume_payload["iteration"] >= cfg.num_iterations:
            print(
                f"run {run_dir} is already complete (iter {resume_payload['iteration']}/"
                f"{cfg.num_iterations}, step {resume_payload['global_step']}); nothing to do"
            )
            return run_dir
    else:
        run_name = cfg.exp_name or f"{cfg.env_id}__ippo__{cfg.seed}__{int(time.time())}"
        run_dir = Path(cfg.runs_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

    envs = make_envs(cfg, cfg.num_envs, ignore_terminations=not cfg.partial_reset)
    eval_envs = make_envs(cfg, cfg.num_eval_envs, ignore_terminations=True)
    try:
        return _train(cfg, run_dir, envs, eval_envs, resume_payload)
    finally:
        envs.close()
        eval_envs.close()


def _action_info(
    envs: ManiSkillVectorEnv,
) -> tuple[tuple[str, str], list[tuple[torch.Tensor, torch.Tensor]], list[int]]:
    """`(agent uids, per-agent action bounds on the env's device, per-agent action widths)`."""
    device = envs.device
    uids = agent_uids_of(envs)
    for uid in uids:
        space = envs.single_action_space[uid]
        assert isinstance(space, gym.spaces.Box), "only continuous actions are supported"
    action_bounds = [
        (
            torch.as_tensor(envs.single_action_space[uid].low, device=device),
            torch.as_tensor(envs.single_action_space[uid].high, device=device),
        )
        for uid in uids
    ]
    action_dims = [int(envs.single_action_space[uid].shape[0]) for uid in uids]
    return uids, action_bounds, action_dims


def run_eval_only(cfg: IPPOConfig) -> Path:
    """Evaluate the deterministic policies of `cfg.checkpoint` (no training); returns the run dir.

    Runs `cfg.eval_repeats` evaluations of the eval envs (reset seeds `seed + 1 + k`), prints the
    metrics averaged over all finished episodes together with the grasp diagnostics, and writes
    them to `<run dir>/eval.json`. This is how a BC checkpoint is judged before any PPO.
    """
    run_name = cfg.exp_name or f"{cfg.env_id}__eval__{cfg.seed}__{int(time.time())}"
    run_dir = Path(cfg.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_envs = make_envs(cfg, cfg.num_eval_envs, ignore_terminations=True)
    try:
        device = eval_envs.device
        _, action_bounds, action_dims = _action_info(eval_envs)
        env_episode_steps = int(gym_utils.find_max_episode_steps_value(eval_envs._env))
        num_eval_steps = resolve_episode_length(cfg, env_episode_steps)
        obs, _ = eval_envs.reset(seed=cfg.seed + 1)
        builder = make_obs_builder(eval_envs, obs, cfg)
        agents = [ActorCritic(d, a).to(device) for d, a in zip(builder.obs_dims, action_dims)]
        payload = load_checkpoint(cfg.checkpoint, map_location=device)
        check_warm_start_compat(payload, builder.obs_dims, builder.fields, cfg.partner_obs)
        load_agent_weights(payload, agents)
        print(
            f"eval only: {cfg.checkpoint} | env {cfg.env_id} | sim {cfg.sim_backend} on {device} "
            f"| partner_obs {cfg.partner_obs} | {cfg.eval_repeats} x {cfg.num_eval_envs} episodes "
            f"of {num_eval_steps} steps"
        )
        tracker = NonFiniteTracker()
        sums: dict[str, float] = {}
        episodes = 0
        diag_sums: dict[str, float] = {}
        diag_maxes: dict[str, float] = {}
        nonfinite = 0
        for k in range(cfg.eval_repeats):
            result = evaluate(
                agents, builder, eval_envs, action_bounds, num_eval_steps, tracker, cfg.seed + 1 + k
            )
            for key, value in result.metrics.items():
                sums[key] = sums.get(key, 0.0) + value * result.episodes
            episodes += result.episodes
            for key, value in result.diag_means.items():
                diag_sums[key] = diag_sums.get(key, 0.0) + value / cfg.eval_repeats
            for key, value in result.diag_maxes.items():
                diag_maxes[key] = max(diag_maxes.get(key, value), value)
            nonfinite += result.nonfinite
            print(
                f"  round {k + 1}/{cfg.eval_repeats}: {result.episodes} episodes | "
                + " ".join(f"{m}={v:.3f}" for m, v in sorted(result.metrics.items())),
                flush=True,
            )
        metrics = {key: total / episodes for key, total in sums.items()} if episodes else {}
        shown = " ".join(f"{k}={v:.3f}" for k, v in sorted(metrics.items()))
        print(f"eval only total: {episodes} episodes | {shown}")
        print(eval_diag_summary(diag_sums, diag_maxes))
        if nonfinite:
            print(f"NONFINITE {nonfinite} env(s) dropped")
        (run_dir / "eval.json").write_text(
            json.dumps(
                {
                    "checkpoint": cfg.checkpoint,
                    "episodes": episodes,
                    "metrics": metrics,
                    "diag_means": diag_sums,
                    "diag_maxes": diag_maxes,
                    "nonfinite_envs": nonfinite,
                    "seeds": [cfg.seed + 1 + k for k in range(cfg.eval_repeats)],
                },
                indent=2,
            )
        )
    finally:
        eval_envs.close()
    return run_dir


def _best_score_of(run_dir: Path) -> tuple[float, float] | None:
    """The best-eval score stored in `best.pt` (it can be newer than the one in `latest.pt`)."""
    path = run_dir / "best.pt"
    if not path.is_file():
        return None
    best = load_checkpoint(path).get("best_score")
    return None if best is None else (float(best[0]), float(best[1]))


def _train(
    cfg: IPPOConfig,
    run_dir: Path,
    envs: ManiSkillVectorEnv,
    eval_envs: ManiSkillVectorEnv,
    resume_payload: dict | None = None,
) -> Path:
    device = envs.device
    uids, action_bounds, action_dims = _action_info(envs)

    env_episode_steps = int(gym_utils.find_max_episode_steps_value(envs._env))
    num_eval_steps = resolve_episode_length(cfg, env_episode_steps)

    # Start the envs and build/verify the per-agent observation slicing on both of them. The env
    # state cannot be restored on resume, so the resets are seeded from the resumed iteration
    # (a fresh run: `seed` / `seed + 1`) and are not a replay of the start of the run.
    resumed_iteration = 0 if resume_payload is None else int(resume_payload["iteration"])
    train_seed, eval_seed = env_reset_seeds(cfg, resumed_iteration)
    obs, _ = envs.reset(seed=train_seed)
    builder = make_obs_builder(envs, obs, cfg)
    eval_obs, _ = eval_envs.reset(seed=eval_seed)
    eval_builder = make_obs_builder(eval_envs, eval_obs, cfg)
    assert eval_builder.fields == builder.fields, "train and eval observations differ"

    # TODO(review): observations are fed raw, without normalisation (as in the ManiSkill baseline).
    # qvel and the unbounded poses are on different scales; add a running normaliser if learning
    # stalls on the server.
    agents = [ActorCritic(d, a).to(device) for d, a in zip(builder.obs_dims, action_dims)]
    optimizers = [torch.optim.Adam(a.parameters(), lr=cfg.learning_rate, eps=1e-5) for a in agents]
    resumed: TrainingState | None = None
    if resume_payload is not None:
        # After the env resets and the network construction (the last consumers of torch random
        # numbers during setup), so the restored RNG state is the one at the checkpoint.
        resumed = load_training_state(resume_payload, agents, optimizers)
        best_on_disk = _best_score_of(run_dir)
        if best_on_disk is not None and (
            resumed.best_score is None or best_on_disk > resumed.best_score
        ):
            resumed.best_score = best_on_disk
    elif cfg.checkpoint:
        payload = load_checkpoint(cfg.checkpoint, map_location=device)
        check_warm_start_compat(payload, builder.obs_dims, builder.fields, cfg.partner_obs)
        load_agent_weights(payload, agents)
        print(f"warm start from {cfg.checkpoint}")

    # Demonstrations for the auxiliary BC loss (cut with the same per-agent input rule as the
    # rollouts; the file's layout must be the env's). On resume the file is read again: it is
    # named in the run's config.json and does not change.
    demo_batches: list[dict[str, torch.Tensor] | None] = [None] * len(agents)
    if cfg.demos and cfg.bc_coef > 0:
        demo_file = load_demos(cfg.demos)
        check_layout_matches(demo_file["meta"], builder.layout)
        demo_obs, demo_actions = agent_demo_tensors(demo_file, builder)
        demo_batches = [
            {"obs": o.to(device), "actions": a.to(device)}
            for o, a in zip(demo_obs, demo_actions, strict=True)
        ]
        print(
            f"auxiliary BC loss: {demo_batches[0]['obs'].shape[0]} demo transitions from "
            f"{cfg.demos}, coef {cfg.bc_coef} decaying to 0 over {cfg.bc_decay_iters} iterations"
        )
    if cfg.critic_warmup_iters > 0:
        print(f"critic warm-up: actors frozen for the first {cfg.critic_warmup_iters} iterations")

    config_dict = dataclasses.asdict(cfg)
    config_dict.update(
        env_max_episode_steps=env_episode_steps,
        num_eval_steps=num_eval_steps,
        batch_size=cfg.batch_size,
        minibatch_size=cfg.minibatch_size,
        num_iterations=cfg.num_iterations,
        obs_dims=builder.obs_dims,
    )
    if resumed is None:
        writer = SummaryWriter(str(run_dir))
        (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in config_dict.items()),
        )
    else:
        # config.json and the hyperparameters text are already there. Events the killed process
        # logged after the checkpoint (steps >= purge_step) are discarded by TensorBoard, so the
        # curves continue cleanly from the checkpointed iteration.
        writer = SummaryWriter(str(run_dir), purge_step=resumed.global_step + 1)
    logger = MetricLogger(writer)

    print(
        f"run {run_dir} | env {cfg.env_id} | sim {cfg.sim_backend} on {device} | "
        f"partner_obs {cfg.partner_obs}"
    )
    print(
        f"envs {cfg.num_envs} x steps {cfg.num_steps} = batch {cfg.batch_size} "
        f"(minibatch {cfg.minibatch_size}, {cfg.update_epochs} epochs) | "
        f"{cfg.num_iterations} iterations = {cfg.num_iterations * cfg.batch_size} steps | "
        f"episode {env_episode_steps} steps, eval {num_eval_steps} steps x {cfg.num_eval_envs} envs"
    )
    for i, name in enumerate(AGENT_NAMES):
        print(
            f"{name} ({uids[i]}): obs {builder.obs_dims[i]} -> act {action_dims[i]} | "
            + ", ".join(builder.fields[i])
        )
    print(f"reward: {cfg.reward_mode} (shared by both agents), gamma {cfg.gamma}")

    stop_requested = False

    def request_stop(signum, frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"signal {signum}: stopping after this iteration", flush=True)

    signal.signal(signal.SIGTERM, request_stop)

    # Rollout storage, per agent where it differs.
    steps, n = cfg.num_steps, cfg.num_envs
    obs_buf = [torch.zeros((steps, n, d), device=device) for d in builder.obs_dims]
    act_buf = [torch.zeros((steps, n, a), device=device) for a in action_dims]
    logprob_buf = [torch.zeros((steps, n), device=device) for _ in agents]
    value_buf = [torch.zeros((steps, n), device=device) for _ in agents]
    rewards = torch.zeros((steps, n), device=device)  # shared by both agents
    dones = torch.zeros((steps, n), device=device)

    state = {"best_score": None, "last_eval": {}}
    if resumed is not None:
        state["best_score"] = resumed.best_score
        state["last_eval"] = resumed.last_eval
        # So the stdout line shows the last evaluation until the next one runs.
        for key, value in resumed.last_eval.items():
            logger.latest["eval/" + key] = value

    tracker = NonFiniteTracker()

    def run_eval(iteration: int, global_step: int) -> None:
        eval_start = time.time()
        result = evaluate(agents, eval_builder, eval_envs, action_bounds, num_eval_steps, tracker)
        metrics, episodes = result.metrics, result.episodes
        logger.log_many(metrics, global_step, prefix="eval/")
        logger.log("eval/episodes", episodes, global_step)
        logger.log_many(result.diag_means, global_step, prefix="eval_diag/")
        logger.log_many(
            _without_reward_terms(result.diag_maxes), global_step, prefix="eval_diag_max/"
        )
        logger.log("eval/nonfinite_envs", result.nonfinite, global_step)
        state["last_eval"] = metrics
        shown = " ".join(f"{k}={v:.3f}" for k, v in sorted(metrics.items()))
        summary = eval_diag_summary(result.diag_means, result.diag_maxes)
        extras = ([summary] if summary else []) + (
            [f"NONFINITE {result.nonfinite} env(s) dropped"] if result.nonfinite else []
        )
        print(
            f"eval @ iter {iteration}: {episodes} episodes | {shown} | "
            + "".join(f"{e} | " for e in extras)
            + f"{time.time() - eval_start:.1f}s",
            flush=True,
        )
        score = (metrics.get("success_once", 0.0), metrics.get("return", float("-inf")))
        if state["best_score"] is None or score > state["best_score"]:
            state["best_score"] = score
            checkpoint("best.pt", iteration, global_step)

    def checkpoint(name: str, iteration: int, global_step: int) -> None:
        save_checkpoint(
            run_dir / name,
            agents,
            cfg,
            builder.obs_dims,
            list(uids),
            iteration,
            global_step,
            state["last_eval"],
            optimizers=optimizers,
            best_score=state["best_score"],
            train_seconds=base_seconds + time.time() - start_time,
        )

    global_step = 0
    base_seconds = 0.0  # wall time of the earlier processes of a resumed run
    first_iteration = 1
    start_time = time.time()
    agent_obs = builder(obs)
    next_done = torch.zeros(n, device=device)
    diag = DiagStats(n, device)  # its per-episode "ever" flags span the iterations
    if resumed is None:
        run_eval(0, 0)
        iteration = 0
    else:
        iteration = resumed.iteration
        global_step = resumed.global_step
        base_seconds = resumed.train_seconds
        first_iteration = iteration + 1
        print(
            f"resumed {run_dir} from iter {iteration}/{cfg.num_iterations}, step {global_step}",
            flush=True,
        )
    start_step = global_step  # SPS counts only the steps of this process

    for iteration in range(first_iteration, cfg.num_iterations + 1):
        lr = learning_rate_at(cfg, iteration)
        for opt in optimizers:
            opt.param_groups[0]["lr"] = lr
        warmup = critic_warmup_active(cfg, iteration)
        bc_coef = bc_coef_at(cfg, iteration)

        # --- Rollout ---------------------------------------------------------------------
        rollout_start = time.time()
        final_values = [torch.zeros((steps, n), device=device) for _ in agents]
        train_stats = EpisodeStats()
        iteration_nonfinite = 0
        for step in range(steps):
            global_step += n
            dones[step] = next_done
            actions = {}
            with torch.no_grad():
                for i, agent in enumerate(agents):
                    obs_buf[i][step] = agent_obs[i]
                    action, logprob, _, value = agent.get_action_and_value(agent_obs[i])
                    act_buf[i][step] = action
                    logprob_buf[i][step] = logprob
                    value_buf[i][step] = value.flatten()
                    low, high = action_bounds[i]
                    actions[uids[i]] = torch.clamp(action, low, high)

            obs, reward, terminations, truncations, infos = envs.step(actions)
            # Envs whose GPU state blew up (non-finite obs/reward) are reset on their own, made
            # terminal with zero reward and kept out of the statistics; see _nonfinite.
            poisoned = None
            guard = guard_step(
                envs,
                obs,
                reward,
                infos,
                layout=builder.layout,
                actions=actions,
                describe=not tracker.reported,
            )
            if guard is not None:
                obs, reward, poisoned = guard.obs, guard.reward, guard.poisoned
                iteration_nonfinite += guard.count
                tracker.record("train", guard, f"iteration {iteration}, rollout step {step}")
            agent_obs = builder(obs)
            next_done = torch.logical_or(terminations, truncations).float()
            if poisoned is not None:
                next_done = torch.where(poisoned, torch.ones_like(next_done), next_done)
            rewards[step] = reward.view(-1) * cfg.reward_scale
            diag.add(infos, None if poisoned is None else ~poisoned)

            if "final_info" in infos:
                train_stats.add(infos, exclude=poisoned)
                # Episodes that just ended were auto-reset. Time-limit truncations bootstrap
                # from their true last obs; terminations (success) are true terminals and keep
                # final_values = 0. (The ManiSkill baseline bootstraps both. With the FaceTurn
                # success bonus that would feed the bonus back through V(final obs) of the
                # success state and inflate the value function.) Poisoned envs never bootstrap.
                done_mask = infos["_final_info"] & ~terminations
                if poisoned is not None:
                    done_mask = done_mask & ~poisoned
                if done_mask.any():
                    with torch.no_grad():
                        final_obs = builder(infos["final_observation"][done_mask])
                        for i, agent in enumerate(agents):
                            final_values[i][step, done_mask] = agent.get_value(final_obs[i]).view(
                                -1
                            )
        rollout_time = time.time() - rollout_start

        # --- Advantages and per-agent PPO updates ----------------------------------------
        update_start = time.time()
        for i, agent in enumerate(agents):
            with torch.no_grad():
                next_value = agent.get_value(agent_obs[i]).reshape(1, -1)
                advantages, returns = compute_gae(
                    rewards,
                    value_buf[i],
                    dones,
                    final_values[i],
                    next_value,
                    next_done,
                    cfg.gamma,
                    cfg.gae_lambda,
                )
            batch = {
                "obs": obs_buf[i].reshape(-1, builder.obs_dims[i]),
                "actions": act_buf[i].reshape(-1, action_dims[i]),
                "logprobs": logprob_buf[i].reshape(-1),
                "advantages": advantages.reshape(-1),
                "returns": returns.reshape(-1),
                "values": value_buf[i].reshape(-1),
            }
            agent.train()
            metrics = ppo_update(
                agent,
                optimizers[i],
                batch,
                cfg,
                update_actor=not warmup,
                demo_batch=demo_batches[i],
                bc_coef=bc_coef,
            )
            if not all(np.isfinite(metrics[k]) for k in ("policy_loss", "value_loss", "entropy")):
                raise FloatingPointError(
                    f"{AGENT_NAMES[i]}: non-finite loss at iteration {iteration}: {metrics}"
                )
            logger.log_many(metrics, global_step, prefix=f"losses/{AGENT_NAMES[i]}/")
            logger.log(
                f"policy/{AGENT_NAMES[i]}/action_std",
                agent.actor_logstd.exp().mean().item(),
                global_step,
            )
        update_time = time.time() - update_start

        # --- Logging, evaluation, checkpoints ---------------------------------------------
        sps = (global_step - start_step) / (time.time() - start_time)
        logger.log_many(train_stats.means(), global_step, prefix="train/")
        logger.log("train/episodes", train_stats.num_episodes, global_step)
        diag_means, diag_maxes = diag.take()
        logger.log_many(diag_means, global_step, prefix="diag/")
        logger.log_many(_without_reward_terms(diag_maxes), global_step, prefix="diag_max/")
        logger.log("train/nonfinite_envs", iteration_nonfinite, global_step)
        logger.log("train/nonfinite_envs_total", tracker.totals["train"], global_step)
        logger.log("rollout/step_reward", rewards.mean(), global_step)
        logger.log("charts/learning_rate", lr, global_step)
        logger.log("charts/bc_coef", bc_coef, global_step)
        logger.log("charts/critic_warmup", float(warmup), global_step)
        logger.log("charts/SPS", sps, global_step)
        logger.log("time/rollout_time", rollout_time, global_step)
        logger.log("time/update_time", update_time, global_step)
        line = progress_line(
            logger,
            iteration,
            cfg.num_iterations,
            global_step,
            sps,
            train_stats.num_episodes,
            nonfinite=(iteration_nonfinite, tracker.totals["train"]),
        )
        if warmup:
            line += " | critic warm-up"
        if bc_coef > 0 and not warmup:  # the BC loss is idle while the actors are frozen
            bc_losses = [logger.get(f"losses/{name}/bc_loss") for name in AGENT_NAMES]
            line += f" | bc coef {bc_coef:.3f} loss a/b {bc_losses[0]:.4f}/{bc_losses[1]:.4f}"
        print(line, flush=True)

        last = iteration == cfg.num_iterations or stop_requested
        if iteration % cfg.eval_freq == 0 or last:
            run_eval(iteration, global_step)
        if iteration % cfg.checkpoint_freq == 0 or last:
            checkpoint("latest.pt", iteration, global_step)
            print(f"saved {run_dir / 'latest.pt'}", flush=True)
        writer.flush()
        if stop_requested:
            break

    process_seconds = time.time() - start_time
    print(
        f"done: {iteration} iterations, {global_step} steps in {base_seconds + process_seconds:.0f}s "
        f"({(global_step - start_step) / process_seconds:.0f} sps) | "
        f"best eval {state['best_score']}",
        flush=True,
    )
    writer.close()
    return run_dir


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: `python -m callosum.training.ippo [flags]`."""
    # The server runs this detached under nohup with stdout redirected to a file: flush per line.
    sys.stdout.reconfigure(line_buffering=True)
    run(parse_args(argv))


if __name__ == "__main__":
    main()
