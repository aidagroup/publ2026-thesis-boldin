"""Checkpoint files of the IPPO trainer: both agents' weights plus the config, no optimizer.

Kept small on purpose (the server's `$HOME` quota is below 1 GB): a few MB each, written
atomically so a kill mid-write never leaves a corrupt `latest.pt`. Loading uses
`weights_only=True`, so a checkpoint can only contain tensors and plain Python containers.
"""

import dataclasses
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from callosum.configs.ippo import IPPOConfig

CHECKPOINT_FORMAT = 1
AGENT_NAMES = ("agent_a", "agent_b")


def save_checkpoint(
    path: Path,
    agents: list[nn.Module],
    cfg: IPPOConfig,
    obs_dims: list[int],
    agent_uids: list[str],
    iteration: int,
    global_step: int,
    eval_metrics: dict[str, float] | None = None,
) -> None:
    """Write both agents' weights and the run description to `path` (atomically)."""
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "agents": {name: agent.state_dict() for name, agent in zip(AGENT_NAMES, agents)},
        "config": dataclasses.asdict(cfg),
        "obs_dims": list(obs_dims),
        "agent_uids": list(agent_uids),
        "iteration": iteration,
        "global_step": global_step,
        "eval": dict(eval_metrics or {}),
    }
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: torch.device | str = "cpu") -> dict[str, Any]:
    """Read a checkpoint written by `save_checkpoint` (tensors and plain containers only)."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format {payload.get('format')!r} in {path}")
    return payload


def load_agent_weights(payload: dict[str, Any], agents: list[nn.Module]) -> None:
    """Load the weights of a checkpoint payload into `agents` (agent_a first)."""
    for name, agent in zip(AGENT_NAMES, agents):
        agent.load_state_dict(payload["agents"][name])
