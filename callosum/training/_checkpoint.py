"""Checkpoint files of the IPPO trainer: weights plus the full training state for exact resume.

Format 2 (written by `save_checkpoint`) holds both agents' weights, the config, both optimizers'
state (Adam moments and step counters), the best-evaluation score, the torch RNG states (CPU and
all CUDA devices) and the cumulative training wall time, next to the iteration / global step
counters. `ippo --resume <run dir>` restores all of it with `load_training_state`, so a run killed
at any point continues as if it had not been interrupted (the simulator state itself cannot be
saved; the trainer re-seeds the env resets instead). Format 1 (weights, config and counters only)
is still read: it is enough for a weights-only warm start (`--checkpoint`, `load_agent_weights`)
but not for a resume. The BC pretraining (`callosum.training.bc`) writes format 1 too
(`save_weights_checkpoint`), with the per-agent input widths and field names added;
`check_warm_start_compat` compares those and `partner_obs` with the trainer's before the weights
are loaded, so a checkpoint made for other policy inputs fails loudly.

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


def save_weights_checkpoint(
    path: Path,
    agents: list[nn.Module],
    config: dict[str, Any],
    obs_dims: list[int],
    agent_uids: list[str],
    obs_fields: list[list[str]],
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a weights-only checkpoint (format 1) to `path` (atomically).

    This is what `callosum.training.bc` produces: both agents' weights plus the config it was made
    with (a plain dict, must hold `partner_obs`), the per-agent input widths and field names, and
    free-form `extra` (training statistics). `ippo --checkpoint` loads it as a warm start; it
    holds no optimizer state, so it cannot be resumed.
    """
    payload: dict[str, Any] = {
        "format": 1,
        "agents": {name: agent.state_dict() for name, agent in zip(AGENT_NAMES, agents)},
        "config": dict(config),
        "obs_dims": list(obs_dims),
        "obs_fields": [list(f) for f in obs_fields],
        "agent_uids": list(agent_uids),
        "iteration": 0,
        "global_step": 0,
        "eval": {},
        "extra": dict(extra or {}),
    }
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def check_warm_start_compat(
    payload: dict[str, Any],
    obs_dims: list[int],
    obs_fields: list[list[str]] | None = None,
    partner_obs: str | None = None,
) -> None:
    """Raise `ValueError` if a checkpoint's policies do not take the inputs the trainer builds.

    Compares the per-agent input widths (always present), the input field names (only if both
    sides have them: BC checkpoints store them) and `partner_obs` (if the checkpoint's config has
    it). A mismatch means the weights were trained on different inputs, e.g. another
    `--partner-obs` or a changed observation layout, and would silently feed garbage otherwise.
    """
    saved_dims = payload.get("obs_dims")
    if saved_dims is not None and list(saved_dims) != list(obs_dims):
        raise ValueError(
            f"checkpoint policies take inputs of width {list(saved_dims)} but the env gives "
            f"{list(obs_dims)} (different --partner-obs or observation layout?)"
        )
    saved_fields = payload.get("obs_fields")
    if (
        obs_fields is not None
        and saved_fields is not None
        and [list(f) for f in saved_fields] != [list(f) for f in obs_fields]
    ):
        raise ValueError("checkpoint input fields differ from the env's per-agent inputs")
    saved_partner = payload.get("config", {}).get("partner_obs")
    if partner_obs is not None and saved_partner is not None and saved_partner != partner_obs:
        raise ValueError(
            f"checkpoint was made with partner_obs={saved_partner!r}, this run uses {partner_obs!r}"
        )


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
