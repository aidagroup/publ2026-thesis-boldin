"""Demonstration files of the scripted expert: layout, assembly, loading, Monte-Carlo returns.

`scripts/collect_demos.py` records the expert (`callosum.experts.face_turn_expert`) driving the
training env (`pd_joint_delta_pos`, `obs_mode="state"`, `reward_mode="normalized_dense"`) and
writes a `.pt` file with the layout below; `callosum.training.bc` and the IPPO fine-tune
(`--demos`) read it. No mani_skill import, and torch is only imported inside the functions that
touch tensors (`discounted_returns` is plain numpy), so the logic is testable on macOS/CI.

File layout (a dict, loadable with `torch.load(..., weights_only=True)`), `S` = transitions of
all kept episodes, `E` = kept episodes, episodes stored back to back in order:

* `obs` `(S, D)` float32: the flat state observation the env returned *before* the step.
* `actions` `(S, 2, 6)` float32: the actions passed to `env.step` (index 1: agent_a = holder
  `so101_pg-0`, agent_b = rotator `so101_pg-1`; 5 normalised joint deltas + gripper, in [-1, 1]).
* `rewards` `(S,)`: the env's reward of the step (normalised dense, success bonus included).
* `terminated`, `truncated`, `success` `(S,)` bool: the step's flags. A successful episode is
  cut at its first success, which is where the training env terminates (`terminated` is True on
  that last step); an unsuccessful one (only with `--keep-all`) runs to the step budget.
* `episode_id` `(S,)` int64 and `step` `(S,)` int64: the episode index (0..E-1) and the step
  within the episode, per transition.
* `episodes`: dict of `(E,)` tensors: `length`, `success` (bool), `first_success_step`
  (-1 if none), `batch_seed` (reset seed of the batch it came from) and `env_index`.
* `meta`: plain-Python metadata: `format`, `env_id`, `control_mode`, `reward_mode`,
  `partner_obs` (the env's flag; it does not change the observation), `max_episode_steps`,
  `obs_layout` (`[[path_string, width], ...]` in flattening order, path like
  `agent/so101_pg-0/qpos`), `agent_uids`, `obs_dim`, `action_dims`, `num_episodes`,
  `num_attempted`, `num_successful`, `batch_seeds`, `num_envs`, `sim_backend`, `git_commit`,
  `expert` (the expert's control config).
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

DEMO_FORMAT = 1
ACTION_DIM = 6
NUM_AGENTS = 2
_STEP_KEYS = ("obs", "actions", "rewards", "terminated", "truncated", "success")
_EPISODE_KEYS = ("length", "success", "first_success_step", "batch_seed", "env_index")


def discounted_returns(
    rewards: np.ndarray, episode_id: np.ndarray, gamma: float, reward_scale: float = 1.0
) -> np.ndarray:
    """Discounted Monte-Carlo return of every transition, per episode.

    `G_t = scale * r_t + gamma * G_{t+1}` with `G = 0` after an episode's last step. Success is a
    true terminal in the IPPO trainer (the value after it is 0, only time-limit truncations
    bootstrap), so for the successful demos this is exactly the value the critic should have.
    For an episode that ran into the time limit the tail is under-estimated (no bootstrap).

    Args:
        rewards: `(S,)` rewards of the transitions.
        episode_id: `(S,)` episode index per transition; each episode must be contiguous.
        gamma: discount factor in (0, 1].
        reward_scale: multiplier applied to the rewards (the trainer's `reward_scale`).

    Returns:
        `(S,)` float64 returns.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    episode_id = np.asarray(episode_id)
    if rewards.shape != episode_id.shape or rewards.ndim != 1:
        raise ValueError(f"rewards {rewards.shape} and episode_id {episode_id.shape} must be 1-D")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    out = np.zeros_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        if t == len(rewards) - 1 or episode_id[t + 1] != episode_id[t]:
            running = 0.0
        running = reward_scale * rewards[t] + gamma * running
        out[t] = running
    return out


def split_rollout(
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    success: torch.Tensor,
    *,
    keep_all: bool,
    batch_seed: int,
) -> list[dict[str, Any]]:
    """Cut a time-major batched rollout into per-env episodes.

    Args:
        obs: `(T, n, D)`; actions `(T, n, 2, 6)`; rewards / terminated / truncated / success
            `(T, n)`. Index `[t, i]` is step `t` of env `i`.
        keep_all: keep unsuccessful envs too (default: only the successful ones).
        batch_seed: reset seed of this batch (stored per episode).

    Returns:
        One dict per kept env with the per-step tensors (keys `obs`, `actions`, `rewards`,
        `terminated`, `truncated`, `success`, each `(L, ...)`) and the ints `first_success_step`
        (-1 if none), `batch_seed`, `env_index`. A successful episode ends at its first success.
    """
    steps, num_envs = rewards.shape
    ever = success.any(dim=0)
    first = success.long().argmax(dim=0)  # first True along time (0 where never)
    episodes = []
    for i in range(num_envs):
        if not bool(ever[i]) and not keep_all:
            continue
        first_step = int(first[i]) if bool(ever[i]) else -1
        length = first_step + 1 if first_step >= 0 else steps
        episodes.append(
            {
                "obs": obs[:length, i],
                "actions": actions[:length, i],
                "rewards": rewards[:length, i],
                "terminated": terminated[:length, i],
                "truncated": truncated[:length, i],
                "success": success[:length, i],
                "first_success_step": first_step,
                "batch_seed": batch_seed,
                "env_index": i,
            }
        )
    return episodes


def assemble_demos(episodes: Sequence[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    """Concatenate `split_rollout` episodes into the demo-file dict (see the module doc)."""
    import torch

    if not episodes:
        raise ValueError("no episodes to assemble (no successful demonstration was collected)")
    data: dict[str, Any] = {
        key: torch.cat([e[key] for e in episodes]).cpu().contiguous() for key in _STEP_KEYS
    }
    data["obs"] = data["obs"].float()
    data["actions"] = data["actions"].float()
    data["rewards"] = data["rewards"].float()
    lengths = torch.tensor([len(e["rewards"]) for e in episodes], dtype=torch.long)
    data["episode_id"] = torch.repeat_interleave(torch.arange(len(episodes)), lengths)
    data["step"] = torch.cat([torch.arange(n) for n in lengths.tolist()])
    data["episodes"] = {
        "length": lengths,
        "success": torch.tensor([bool(e["success"].any()) for e in episodes]),
        "first_success_step": torch.tensor([e["first_success_step"] for e in episodes]),
        "batch_seed": torch.tensor([e["batch_seed"] for e in episodes]),
        "env_index": torch.tensor([e["env_index"] for e in episodes]),
    }
    data["meta"] = dict(meta, format=DEMO_FORMAT, num_episodes=len(episodes))
    validate_demos(data)
    return data


def validate_demos(demos: dict[str, Any]) -> None:
    """Raise `ValueError` if a demo dict does not follow the file layout."""
    missing = [k for k in (*_STEP_KEYS, "episode_id", "step", "episodes", "meta") if k not in demos]
    if missing:
        raise ValueError(f"demo file is missing keys {missing}")
    if demos["meta"].get("format") != DEMO_FORMAT:
        raise ValueError(f"unsupported demo format {demos['meta'].get('format')!r}")
    size = demos["obs"].shape[0]
    if demos["obs"].ndim != 2 or demos["actions"].shape != (size, NUM_AGENTS, ACTION_DIM):
        raise ValueError(
            f"obs {tuple(demos['obs'].shape)} / actions {tuple(demos['actions'].shape)} do not "
            f"match (S, D) / (S, {NUM_AGENTS}, {ACTION_DIM})"
        )
    for key in ("rewards", "terminated", "truncated", "success", "episode_id", "step"):
        if demos[key].shape != (size,):
            raise ValueError(f"{key} has shape {tuple(demos[key].shape)}, expected ({size},)")
    for key in _EPISODE_KEYS:
        if key not in demos["episodes"]:
            raise ValueError(f"demo file is missing episodes/{key}")
    if int(demos["episodes"]["length"].sum()) != size:
        raise ValueError("episode lengths do not add up to the number of transitions")
    layout_width = sum(width for _, width in demos["meta"]["obs_layout"])
    if layout_width != demos["obs"].shape[1]:
        raise ValueError(f"obs_layout is {layout_width} wide but obs has {demos['obs'].shape[1]}")


def save_demos(path: str | Path, demos: dict[str, Any]) -> None:
    """Validate and write a demo dict with `torch.save`."""
    import torch

    validate_demos(demos)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(demos, path)


def load_demos(path: str | Path) -> dict[str, Any]:
    """Read and validate a demo file (tensors and plain containers only, `weights_only=True`)."""
    import torch

    demos = torch.load(path, map_location="cpu", weights_only=True)
    validate_demos(demos)
    return demos


def layout_from_meta(meta: dict[str, Any]) -> list[tuple[tuple[str, ...], int]]:
    """The observation layout of a demo file in `_agent_obs.Layout` form (path tuples, widths)."""
    return [(tuple(str(path).split("/")), int(width)) for path, width in meta["obs_layout"]]


def layout_to_meta(layout: Sequence[tuple[Sequence[str], int]]) -> list[list[Any]]:
    """`_agent_obs.Layout` as plain lists for the demo file (`["agent/so101_pg-0/qpos", 7]`)."""
    return [["/".join(path), int(width)] for path, width in layout]


def check_layout_matches(meta: dict[str, Any], layout: Sequence[tuple[Sequence[str], int]]) -> None:
    """Raise `ValueError` if the demo file's observation layout differs from the env's `layout`.

    The demo observations are sliced with the env's `AgentObsBuilder`, so a different field order
    or width (an env change after the demos were collected) would feed wrong columns to the
    policy. Compared as `(path, width)` pairs in order.
    """
    saved = [(tuple(path), width) for path, width in layout_from_meta(meta)]
    current = [(tuple(path), int(width)) for path, width in layout]
    if saved != current:
        raise ValueError(
            "the demo file's observation layout differs from the env's (recollect the demos): "
            f"file {[(p, w) for p, w in saved]} vs env {[(p, w) for p, w in current]}"
        )


def episode_length_mismatch(
    demo_steps: int | None, env_steps: int, source: str = "the demo file"
) -> str | None:
    """A message if demos recorded with `demo_steps`-step episodes do not fit an env with
    `env_steps`, else `None` (also `None` when the file does not record its episode length).

    The demonstrations are time-limited trajectories: the expert's pace, the value targets (the
    returns of an episode cut by the limit are bootstrapped differently) and the policy's
    time-to-go cues all assume the episode length they were collected with, so BC and the IPPO
    fine-tune must run in an env with the same `max_episode_steps`.
    """
    if demo_steps is None or int(demo_steps) == int(env_steps):
        return None
    return (
        f"{source} was collected with max_episode_steps={int(demo_steps)} but the training env "
        f"runs {int(env_steps)}-step episodes: pass the same value (`--max-episode-steps "
        f"{int(demo_steps)}`) or recollect the demos"
    )


def success_mask(demos: dict[str, Any], only_success: bool = True) -> torch.Tensor:
    """`(S,)` bool mask of the transitions to use (those of successful episodes by default)."""
    import torch

    if not only_success:
        return torch.ones(demos["obs"].shape[0], dtype=torch.bool)
    return demos["episodes"]["success"][demos["episode_id"]]


def agent_demo_tensors(
    demos: dict[str, Any], builder: Any, only_success: bool = True
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-agent policy inputs and actions `([obs_a, obs_b], [act_a, act_b])` of the demos.

    The inputs are cut from the flat demo observations by `builder` (an `AgentObsBuilder`, i.e.
    with its `partner_obs` rule); the actions are the demonstrated 6-D actions of that agent.
    """
    mask = success_mask(demos, only_success)
    if not bool(mask.any()):
        raise ValueError("no transitions to use (no successful episode in the demo file)")
    obs = builder(demos["obs"][mask])
    actions = [demos["actions"][mask][:, i] for i in range(NUM_AGENTS)]
    return obs, actions


def git_commit() -> str | None:
    """Short hash of the current commit (with `+dirty` for local changes), or `None`."""
    try:
        run = subprocess.run
        root = Path(__file__).resolve().parent
        head = run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None
