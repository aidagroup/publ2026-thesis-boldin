"""Checkpoint files of the IPPO trainer: weights plus the full training state for exact resume.

Format 2 (written by `save_checkpoint`) holds both agents' weights, the config, both optimizers'
state (Adam moments and step counters), the best-evaluation score, the torch RNG states (CPU and
all CUDA devices) and the cumulative training wall time, next to the iteration / global step
counters. `ippo --resume <run dir>` restores all of it with `load_training_state`, so a run killed
at any point continues as if it had not been interrupted (the simulator state itself cannot be
saved; the trainer re-seeds the env resets instead). Format 1 (weights, config and counters only)
is still read: it is enough for a weights-only warm start (`--checkpoint`, `load_agent_weights`)
but not for a resume.

Files are a few MB (two small MLPs plus Adam's two moments) and are written atomically, so a kill
mid-write never leaves a corrupt `latest.pt`. Loading uses `weights_only=True`, so a checkpoint
can only contain tensors and plain Python containers and numbers (no numpy objects).
"""

import dataclasses
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from callosum.configs.ippo import IPPOConfig

CHECKPOINT_FORMAT = 2
SUPPORTED_FORMATS = (1, 2)
AGENT_NAMES = ("agent_a", "agent_b")


@dataclasses.dataclass
class TrainingState:
    """The counters restored by `load_training_state` (everything else is loaded in place)."""

    iteration: int
    global_step: int
    best_score: tuple[float, float] | None
    """`(success_once, return)` of the best evaluation so far, or `None` before the first."""
    last_eval: dict[str, float]
    train_seconds: float
    """Cumulative wall time of all processes that contributed to this run."""


def save_checkpoint(
    path: Path,
    agents: list[nn.Module],
    cfg: IPPOConfig,
    obs_dims: list[int],
    agent_uids: list[str],
    iteration: int,
    global_step: int,
    eval_metrics: dict[str, float] | None = None,
    *,
    optimizers: list[torch.optim.Optimizer],
    best_score: tuple[float, float] | None = None,
    train_seconds: float = 0.0,
) -> None:
    """Write the full training state (format 2) to `path` (atomically).

    `agents` and `optimizers` are in agent order (agent_a first). The RNG states are those of
    the moment of saving.
    """
    if len(optimizers) != len(agents):
        raise ValueError(f"got {len(agents)} agents but {len(optimizers)} optimizers")
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "agents": {name: agent.state_dict() for name, agent in zip(AGENT_NAMES, agents)},
        "optimizers": {name: opt.state_dict() for name, opt in zip(AGENT_NAMES, optimizers)},
        "config": dataclasses.asdict(cfg),
        "obs_dims": list(obs_dims),
        "agent_uids": list(agent_uids),
        "iteration": iteration,
        "global_step": global_step,
        "eval": dict(eval_metrics or {}),
        "best_score": None if best_score is None else [float(x) for x in best_score],
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "train_seconds": float(train_seconds),
    }
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: torch.device | str = "cpu") -> dict[str, Any]:
    """Read a checkpoint of format 1 or 2 (tensors and plain containers only)."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format") not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported checkpoint format {payload.get('format')!r} in {path}")
    return payload


def load_agent_weights(payload: dict[str, Any], agents: list[nn.Module]) -> None:
    """Load the weights of a checkpoint payload (any format) into `agents` (agent_a first)."""
    for name, agent in zip(AGENT_NAMES, agents):
        agent.load_state_dict(payload["agents"][name])


def load_training_state(
    payload: dict[str, Any],
    agents: list[nn.Module],
    optimizers: list[torch.optim.Optimizer],
) -> TrainingState:
    """Restore weights, optimizer state and RNG states from a format-2 payload.

    Returns the counters. Call it after everything that consumes torch random numbers during
    setup (env resets, network construction), right before training continues. Raises
    `ValueError` for a format-1 payload, which has no optimizer state.
    """
    if payload.get("format") != 2:
        raise ValueError(
            f"checkpoint format {payload.get('format')!r} holds weights only and cannot resume a "
            "run: use --checkpoint <file> for a warm start (fresh optimizer, schedule and "
            "counters) instead of --resume"
        )
    load_agent_weights(payload, agents)
    for name, opt in zip(AGENT_NAMES, optimizers):
        opt.load_state_dict(payload["optimizers"][name])
    torch.set_rng_state(payload["rng"]["torch"].cpu())
    cuda_states = payload["rng"]["cuda"]
    if cuda_states and torch.cuda.is_available():
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])
        else:
            print(
                f"warning: checkpoint has {len(cuda_states)} CUDA RNG states but this machine has "
                f"{torch.cuda.device_count()} devices; CUDA RNG not restored"
            )
    best = payload["best_score"]
    return TrainingState(
        iteration=int(payload["iteration"]),
        global_step=int(payload["global_step"]),
        best_score=None if best is None else (float(best[0]), float(best[1])),
        last_eval=dict(payload["eval"]),
        train_seconds=float(payload["train_seconds"]),
    )
