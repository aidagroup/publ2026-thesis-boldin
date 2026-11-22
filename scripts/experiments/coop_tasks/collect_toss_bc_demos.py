"""Collect scripted-oracle `TossCatch-v0` demonstrations on the batched GPU physics env.

The learned IPPO policies of the toss-catch task start from a behaviour-cloning (BC)
baseline: this script rolls out the scripted centralised oracle
(`callosum.experiments.coop_tasks.oracle.TossOracle`, the 0.89-success scripted upper bound)
on the GPU-first environment and stores the per-step policy inputs and actions, so
`scripts/experiments/coop_tasks/train_toss_bc.py` can clone the actor means without any
simulator knowledge.

The rollout runs on the batched `mujoco_warp` backend (`CoopBatchGPU`), `--num-envs` worlds
per batch. One batch resets its worlds with the consecutive episode seeds
`base + world_index` and records only each world's FIRST episode, exactly like the held-out
evaluation (`callosum/experiments/coop_tasks/eval.py`), so the dataset has a clean
seed-to-episode mapping and no auto-reset bookkeeping. `--max-steps` caps a batch: the
oracle's own successes complete in ~30 control steps, so a rare long failure cannot stall the
collection; an episode that has not finished at the cap is stored as
`episode_complete=False` and the trainer ignores its transitions.

The scripted oracle is the CPU numpy controller (`make_oracle`), so each world gets the CPU
env's oracle surface through a tiny per-world adapter:

* `state()` -- the ball state from the shared observation block, the paddle servo targets
  (`CoopBatchGPU._pad_target`) and the raw ball/paddle contact flags
  (`CoopBatchGPU._contact_side`), which are not part of the observation;
* `pending_commands()` -- the CPU latency queue (`logic.delayed_command`) replayed on the host
  from the commands the oracle itself issued, exactly like the renderer's `GpuDemoEnv`;
* the task geometry queries (`contact_offset`, `paddle_receive_x`, ...) -- from a
  never-stepped CPU env instance, i.e. model constants, not physics (the GPU backend builds
  its model from the same MJCF).

Only the observation (`obs`, 26 values per agent under `partner_mode="none"`,
`occlusion="none"`), the applied normalized action (3 per agent), the episode ids and the
per-episode success flags are copied to the host; all physics stays on the GPU. The demo file
is an `.npz` plus a `.json` metadata sidecar; the trainer (`train_toss_bc.py`) reads both.

Run on the pod from the repo root:

    MUJOCO_GL=egl /root/.callosum/venv/bin/python \\
        scripts/experiments/coop_tasks/collect_toss_bc_demos.py \\
        --num-envs 128 --batches 8 --max-steps 120 --out demos/toss_bc.npz

The `--controller reactive` variant collects the same demonstration format from a Markov
teacher (see `ReactiveTossOracle`), for the case where the stock oracle's hidden-queue
compensation is not observable to a memoryless policy.

Exit status: 1 if the collected (complete) episodes' teacher success rate is below
`--min-success` (0.8 by default), so a broken GPU backend cannot silently poison the dataset.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from callosum.experiments.coop_tasks import make_coop_env, make_oracle
from callosum.experiments.coop_tasks.config import TossConfig, coop_config
from callosum.experiments.coop_tasks.env_gpu import CoopBatchGPU, observation_blocks
from callosum.experiments.coop_tasks.logic import (
    AGENT_KEYS,
    TOSS_SPEC,
    sample_toss_randomization,
)
from callosum.experiments.coop_tasks.oracle import TossOracle

TASK = "toss"
PARTNER_MODE = "none"
OCCLUSION = "none"
_OBS_WIDTHS = {
    "action": TOSS_SPEC.action_dim,
    "command": TOSS_SPEC.action_dim,
    "params": 3,
    "privileged": 1,
    "shared": len(TOSS_SPEC.shared_fields),
    "goal": len(TOSS_SPEC.goal_fields),
    "time": 1,
}
"""Widths of the GPU env's observation blocks (the renderer's `_observation_slices`)."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the collection CLI."""
    parser = argparse.ArgumentParser(
        description="Collect TossCatch scripted-oracle demos on the batched GPU env.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=128, help="GPU worlds per batch")
    parser.add_argument("--batches", type=int, default=8, help="reset batches to roll out")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=120,
        help="per-batch control-step cap (unfinished episodes are stored as incomplete)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=5_000_000,
        help="episode seed of batch 0, world 0 (kept clear of the trainer's and evaluator's "
        "seed streams)",
    )
    parser.add_argument("--out", type=Path, default=Path("demos/toss_bc.npz"), help=".npz path")
    parser.add_argument("--device", default="cuda", help="CUDA device of the GPU env")
    parser.add_argument(
        "--controller",
        choices=("oracle", "reactive", "nopending"),
        default="oracle",
        help=(
            "oracle: the stock scripted controller (its deadbeat command compensates the "
            "hidden latency queue); reactive: the same planner, but the command is a "
            "one-step measured-pose prediction, i.e. a function of the OBSERVABLE state "
            "only -- the variant a memoryless policy can actually clone"
        ),
    )
    parser.add_argument(
        "--min-success",
        type=float,
        default=0.8,
        help="exit non-zero below this oracle success rate over the complete episodes",
    )
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.batches < 1 or args.max_steps < 1:
        parser.error("--num-envs, --batches and --max-steps must be >= 1")
    if args.base_seed < 0:
        parser.error(f"--base-seed must be >= 0, got {args.base_seed}")
    return args


def observation_slices(spec: Any, partner_mode: str) -> dict[str, slice]:
    """Column slices of the `shared` / `goal` / `time` observation blocks (of any role).

    Args:
        spec: The task's observation/action layout (`coop_tasks.logic.task_spec`).
        partner_mode: The env's partner visibility (`none` here).

    Returns:
        `{block_key: slice}` for the shared-state, goal and time blocks.

    Raises:
        RuntimeError: if the reconstructed layout has no shared/goal/time block.
    """
    slices: dict[str, slice] = {}
    offset = 0
    for key, owner in observation_blocks(spec, partner_mode)[AGENT_KEYS[0]]:
        width = _OBS_WIDTHS[key]
        if owner is None:
            slices[key] = slice(offset, offset + width)
        offset += width
    missing = {"shared", "goal", "time"} - set(slices)
    if missing:  # pragma: no cover - guarded by the env's own layout check
        raise RuntimeError(f"the GPU observation layout has no {sorted(missing)} block")
    return slices


class ReactiveTossOracle(TossOracle):
    """The toss-catch planner with an OBSERVABLE-STATE-ONLY command (a Markov teacher).

    The stock oracle converts its planned intercept into a servo-target-space deadbeat command
    that subtracts the latency queue's in-flight commands. Those commands are NOT part of the
    26-d policy observation (only the last applied action is), so cloning the stock oracle with
    a memoryless policy is an ill-posed regression: the same observation maps to many
    different commands and their conditional mean is close to zero exactly where precision
    matters.

    This subclass keeps the whole oracle unchanged (the same ballistic planning, the same
    anticipation phase and aiming tilt) and only replaces `_deadbeat`: the command is the
    correction from a one-step prediction of the MEASURED paddle pose (position + velocity *
    control_dt) to the planned target pose. That is a function of the observation only (the
    measured pose/velocity are the shared-block fields), so the learned policy can reproduce
    it from the 26 inputs.
    """

    def _deadbeat(
        self, env: Any, side: str, target: tuple[float, float, float], max_step: float
    ) -> np.ndarray:
        """Paddle command from the measured pose and its one-step prediction."""
        cfg = self.config
        state = env.state()
        measured = np.array([state[f"{side}_pad_x"], state[f"{side}_pad_z"]])
        velocity = np.array([state[f"{side}_pad_vx"], state[f"{side}_pad_vz"]])
        desired = np.array([target[0], target[1]]) - measured - velocity * cfg.control_dt
        norm = float(np.linalg.norm(desired))
        if norm > max_step:
            desired *= max_step / norm
        randomization = env.randomization
        if randomization is None:
            raise RuntimeError("call env.reset() before the controller acts")
        agent = randomization.agents[AGENT_KEYS.index(side)]
        authority = agent.gain * agent.limit_scale
        max_dtilt = cfg.pad_tilt_rate * cfg.control_dt * authority
        d_tilt = float(np.clip(target[2] - state[f"{side}_pad_tilt"], -max_dtilt, max_dtilt))
        return np.array([desired[0] / max_step, desired[1] / max_step, d_tilt / max_dtilt])


class NoPendingTossOracle(TossOracle):
    """The stock oracle's target-space deadbeat WITHOUT the hidden in-flight compensation.

    `_deadbeat` normally subtracts the latency queue's pending commands, which a memoryless
    policy cannot observe. This variant simply drops that term (the plan, the anticipation
    phase and the aiming tilt are unchanged), so the command is `planned pose - servo target`
    over the max step: the servo target is (approximately) the measured pose, which the
    observation carries. It is the closest Markov controller to the stock oracle and is used
    to test whether the hidden queue is what breaks behaviour cloning.
    """

    def _pending_offset(
        self, env: Any, side: str, max_step: float, step: int
    ) -> tuple[np.ndarray, float]:
        """No in-flight compensation (the queue is invisible to a memoryless policy)."""
        del env, side, max_step, step
        return np.zeros(2), 0.0


def make_controller(controller: str, cfg: TossConfig) -> TossOracle:
    """The teaching controller of a collection run (see `ReactiveTossOracle`)."""
    if controller == "reactive":
        return ReactiveTossOracle(cfg)
    if controller == "nopending":
        return NoPendingTossOracle(cfg)
    return make_oracle(TASK, cfg)


class WorldView:
    """One world of the batched GPU env behind the CPU `TossCatchEnv` oracle surface.

    The scripted `TossOracle` is a CPU numpy controller that reads `state()`,
    `pending_commands()`, the episode randomisation and a handful of static model queries from
    the env. This adapter holds those values for one world, so the same oracle drives a
    `CoopBatchGPU` world without touching the simulator: the dynamics, the randomisation and
    the success rule stay the GPU env's.
    """

    __slots__ = ("_config", "_geometry", "_queues", "_randomization", "_state")

    def __init__(
        self,
        *,
        geometry: Any,
        config: TossConfig,
        randomization: Any,
        state: dict[str, float],
        queues: dict[str, deque],
    ) -> None:
        self._geometry = geometry
        self._config = config
        self._randomization = randomization
        self._state = state
        self._queues = queues

    @property
    def randomization(self) -> Any:
        """The episode's sampled randomisation record (the CPU env's `randomization`)."""
        return self._randomization

    def state(self) -> dict[str, float]:
        """The task's kinematic state, servo targets and contact flags (CPU state keys)."""
        return self._state

    def pending_commands(self) -> dict[str, tuple[np.ndarray, ...]]:
        """The agents' in-flight commands, oldest first (the CPU env's latency queues)."""
        return {role: tuple(queue) for role, queue in self._queues.items()}

    def contact_offset(self) -> float:
        """The ball/paddle contact offset along the paddle normal (model geometry)."""
        return float(self._geometry.contact_offset())

    def paddle_receive_x(self) -> float:
        """The receiving paddle's pre-throw anticipation x (config geometry)."""
        return float(self._geometry.paddle_receive_x())

    def paddle_ready_x(self) -> float:
        """The ready-pose paddle centre |x| (config geometry)."""
        return float(self._geometry.paddle_ready_x())

    def paddle_ready_tilt(self, side: str) -> float:
        """The ready-pose tilt of one paddle (config geometry)."""
        return float(self._geometry.paddle_ready_tilt(side))


def read_gpu_step(
    env: CoopBatchGPU,
    observations: dict[str, torch.Tensor],
    shared_slice: slice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Copy one step's observations, paddle targets and contact flags to the host (one sync).

    The paddle servo targets and the raw ball/paddle contact flags are task state of the
    batched env that its observation does not carry; they are read through the env's own
    attributes (the demo renderer's `GpuDemoEnv` does the same). Everything is packed into one
    host copy so a control step costs exactly one GPU->CPU synchronisation.
    """
    num_envs = env.num_envs
    obs_width = observations[AGENT_KEYS[0]].shape[1]
    # Task state the observation does not carry: the servo targets the plates chase and
    # the raw ball/paddle contact flags of the current control step.
    targets = env._pad_target.reshape(num_envs, 6)
    contacts = env._contact_side
    packed = torch.cat(
        (
            observations[AGENT_KEYS[0]],
            observations[AGENT_KEYS[1]],
            targets,
            contacts.to(torch.float32),
        ),
        dim=1,
    )
    rows = packed.detach().cpu().numpy()
    obs_left = rows[:, :obs_width]
    obs_right = rows[:, obs_width : 2 * obs_width]
    offset = 2 * obs_width
    pad_targets = rows[:, offset : offset + 6]
    contact_flags = rows[:, offset + 6 : offset + 8]
    shared = obs_left[:, shared_slice]
    return obs_left, obs_right, shared, pad_targets, contact_flags


def world_state(
    step: int,
    shared_names: tuple[str, ...],
    shared_row: np.ndarray,
    pad_targets_row: np.ndarray,
    contact_row: np.ndarray,
) -> dict[str, float]:
    """The CPU `TossCatchEnv.state()` dict of one world from the copied GPU arrays."""
    state = {"step": float(step)}
    # The shared block carries the ball and BOTH paddles' measured pose/velocity; the stock
    # oracle only reads the ball fields, while the reactive variant (and the restitution
    # estimate) also uses the measured paddle state.
    for name, value in zip(shared_names, shared_row):
        state[name] = float(value)
    for index, side in enumerate(AGENT_KEYS):
        x, z, tilt = (float(value) for value in pad_targets_row[3 * index : 3 * index + 3])
        state[f"{side}_pad_target_x"] = x
        state[f"{side}_pad_target_z"] = z
        state[f"{side}_pad_target_tilt"] = tilt
        state[f"{side}_contact"] = float(contact_row[index])
    return state


def new_latency_queues(records: list[Any]) -> list[dict[str, deque]]:
    """The host-side latency queues of every world, initialised like `CoopTaskEnv.reset`."""
    queues: list[dict[str, deque]] = []
    for record in records:
        queues.append(
            {
                role: deque(
                    np.zeros(TOSS_SPEC.action_dim, dtype=np.float64)
                    for _ in range(record.agents[index].latency_steps)
                )
                for index, role in enumerate(AGENT_KEYS)
            }
        )
    return queues


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """Roll out the scripted oracle and return the demo arrays plus the run metadata."""
    cfg = coop_config(TASK)
    if not isinstance(cfg, TossConfig):  # pragma: no cover - the task is fixed
        raise TypeError(f"toss needs a TossConfig, got {type(cfg)}")
    torch.set_num_threads(1)
    # The static geometry of the oracle (contact offset, ready pose) is a model constant: the
    # CPU env is built but never reset/stepped, exactly like the demo renderer's `GpuDemoEnv`.
    geometry = make_coop_env(TASK, config=cfg, partner_mode=PARTNER_MODE)
    env = CoopBatchGPU(
        TASK,
        args.num_envs,
        partner_mode=PARTNER_MODE,
        config=cfg,
        seed=args.base_seed,
        device=args.device,
    )
    slices = observation_slices(TOSS_SPEC, PARTNER_MODE)
    num_episodes = args.num_envs * args.batches
    episode_success = np.zeros(num_episodes, dtype=bool)
    episode_complete = np.zeros(num_episodes, dtype=bool)
    episode_seed = np.zeros(num_episodes, dtype=np.int64)
    episode_length = np.zeros(num_episodes, dtype=np.int32)
    obs_left_rows: list[np.ndarray] = []
    obs_right_rows: list[np.ndarray] = []
    action_left_rows: list[np.ndarray] = []
    action_right_rows: list[np.ndarray] = []
    episode_rows: list[np.ndarray] = []
    start = time.time()
    try:
        for batch in range(args.batches):
            base = args.base_seed + batch * args.num_envs
            observations = env.reset(seed=base)
            records = [
                sample_toss_randomization(
                    cfg.randomization,
                    control_dt=cfg.control_dt,
                    max_control_steps=cfg.max_control_steps,
                    seed=base + world,
                )
                for world in range(args.num_envs)
            ]
            oracles = [make_controller(args.controller, cfg) for _ in range(args.num_envs)]
            queues = new_latency_queues(records)
            episode_ids = np.arange(
                batch * args.num_envs, (batch + 1) * args.num_envs, dtype=np.int64
            )
            episode_seed[episode_ids] = base + np.arange(args.num_envs)
            active = np.ones(args.num_envs, dtype=bool)
            obs_left, obs_right, shared, pad_targets, contact_flags = read_gpu_step(
                env, observations, slices["shared"]
            )
            batch_success = 0
            batch_complete = 0
            steps_used = 0
            for step in range(args.max_steps):
                steps_used = step + 1
                active_rows = np.flatnonzero(active)
                if active_rows.size == 0:
                    break
                actions_left = np.zeros((args.num_envs, TOSS_SPEC.action_dim), dtype=np.float32)
                actions_right = np.zeros_like(actions_left)
                for world in active_rows.tolist():
                    view = WorldView(
                        geometry=geometry,
                        config=cfg,
                        randomization=records[world],
                        state=world_state(
                            step,
                            TOSS_SPEC.shared_fields,
                            shared[world],
                            pad_targets[world],
                            contact_flags[world],
                        ),
                        queues=queues[world],
                    )
                    action = oracles[world].act(view)
                    actions_left[world] = np.clip(action["left"], -1.0, 1.0)
                    actions_right[world] = np.clip(action["right"], -1.0, 1.0)
                obs_left_rows.append(obs_left[active_rows])
                obs_right_rows.append(obs_right[active_rows])
                action_left_rows.append(actions_left[active_rows])
                action_right_rows.append(actions_right[active_rows])
                episode_rows.append(episode_ids[active_rows])
                for world in active_rows.tolist():
                    for role, action in zip(
                        AGENT_KEYS, (actions_left[world], actions_right[world])
                    ):
                        queue = queues[world][role]
                        queue.append(action.astype(np.float64))
                        queue.popleft()
                result = env.step(
                    {
                        "left": torch.as_tensor(
                            actions_left, dtype=torch.float32, device=env.device
                        ),
                        "right": torch.as_tensor(
                            actions_right, dtype=torch.float32, device=env.device
                        ),
                    }
                )
                done = (result.terminated | result.truncated).detach().cpu().numpy()
                done_rows = np.flatnonzero(done)
                if len(done_rows) != len(result.finished_successes):  # pragma: no cover
                    raise RuntimeError(
                        "GPU env contract broken: "
                        f"{len(done_rows)} worlds finished but "
                        f"{len(result.finished_successes)} finished statistics reported"
                    )
                for position, world in enumerate(done_rows.tolist()):
                    # Worlds that finished in an earlier step keep auto-resetting in the
                    # background; their later episodes are not part of this dataset.
                    if not active[world]:
                        continue
                    episode = int(episode_ids[world])
                    episode_success[episode] = bool(result.finished_successes[position])
                    episode_length[episode] = int(result.finished_lengths[position])
                    episode_complete[episode] = True
                    batch_success += int(episode_success[episode])
                    batch_complete += 1
                    active[world] = False
                obs_left, obs_right, shared, pad_targets, contact_flags = read_gpu_step(
                    env, result.obs, slices["shared"]
                )
            print(
                f"batch {batch + 1}/{args.batches} | seeds {base}..{base + args.num_envs - 1} | "
                f"steps {steps_used} | complete {batch_complete}/{args.num_envs} | "
                f"success {batch_success}/{batch_complete}",
                flush=True,
            )
    finally:
        env.close()
        geometry.close()

    obs = np.stack((np.concatenate(obs_left_rows), np.concatenate(obs_right_rows)), axis=1)
    actions = np.stack(
        (np.concatenate(action_left_rows), np.concatenate(action_right_rows)), axis=1
    )
    episodes = np.concatenate(episode_rows)
    total = int(episodes.size)
    successes = int(episode_success.sum())
    complete = int(episode_complete.sum())
    success_rate = successes / complete if complete else 0.0
    successful_episodes = np.flatnonzero(episode_success)
    metadata = {
        "task": TASK,
        "partner_mode": PARTNER_MODE,
        "occlusion": OCCLUSION,
        "controller": args.controller,
        "policy_input_size": int(obs.shape[2]),
        "action_size": int(actions.shape[2]),
        "agent_keys": list(AGENT_KEYS),
        "shared_fields": list(TOSS_SPEC.shared_fields),
        "num_envs": int(args.num_envs),
        "batches": int(args.batches),
        "max_steps": int(args.max_steps),
        "base_seed": int(args.base_seed),
        "episode_seed_range": [int(args.base_seed), int(args.base_seed) + num_episodes - 1],
        "transitions": total,
        "successful_episodes": successes,
        "complete_episodes": complete,
        "incomplete_episodes": num_episodes - complete,
        "oracle_success_rate": success_rate,
        "successful_transitions": int(np.isin(episodes, successful_episodes).sum()),
        "mean_complete_episode_length": float(episode_length[episode_complete].mean())
        if complete
        else 0.0,
        "wall_seconds": time.time() - start,
        "device": str(env.device),
        "env_versions": env.software_versions(),
        "cuda_graph": bool(env.cuda_graph),
        "git_commit": _git_commit(),
    }
    return {
        "obs": obs.astype(np.float32),
        "actions": actions.astype(np.float32),
        "episode": episodes.astype(np.int32),
        "episode_success": episode_success,
        "episode_complete": episode_complete,
        "episode_seed": episode_seed,
        "episode_length": episode_length,
        "metadata": metadata,
    }


def _git_commit() -> str | None:
    """The checkout's HEAD commit (a provenance note; `None` outside a git checkout)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - not a git checkout
        return None


def main(argv: list[str] | None = None) -> int:
    """Collect the demos, write the `.npz` + `.json` files and report the oracle success."""
    args = parse_args(argv)
    data = collect(args)
    metadata = data.pop("metadata")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"wrote {out} | transitions {metadata['transitions']} "
        f"(successful {metadata['successful_transitions']}) | episodes "
        f"{metadata['complete_episodes']} complete / "
        f"{metadata['incomplete_episodes']} incomplete | oracle success "
        f"{metadata['oracle_success_rate']:.3f} | "
        f"{metadata['wall_seconds']:.1f}s | {metadata['env_versions']}"
    )
    print(f"wrote {meta_path}")
    if metadata["oracle_success_rate"] < args.min_success:
        print(
            f"error: oracle success {metadata['oracle_success_rate']:.3f} is below "
            f"--min-success {args.min_success}; the GPU env or the oracle is broken",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
