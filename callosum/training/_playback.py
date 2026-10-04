"""Play back a trained IPPO checkpoint outside the trainer (no mani_skill import, torch only).

`scripts/render_episode.py --checkpoint` rolls a checkpoint out in an env that is *not* the one
it was trained in: it has the `so101_pg_wristcam` robots (other uids, plus camera sensors) and a
visual obs mode, while the policy was trained on plain `so101_pg` robots with
`obs_mode="state"`. This module turns the checkpoint back into actors and builds their inputs
from the env's state observation exactly as the trainer did:

* The env is created with `obs_mode="state+rgb"`. ManiSkill then emits the same privileged
  `extra` fields as in `"state"` mode (its task envs gate them on `"state" in obs_mode`) and
  flattens `{"agent", "extra"}` with the same `flatten_state_dict` into `obs["state"]`, next to
  the sensor data. That flat vector is what the trainer's policies were fed.
* The per-agent slicing is the trainer's own `AgentObsBuilder`. Its layout comes from the
  structured observation (`get_obs(unflattened=True)`) with the wrist-camera agent uids renamed
  to the training uids (`rename_agent_keys`), and `check_layout` verifies it against the real
  flat state. The camera robots have the same joints as the plain ones, so the layout is
  identical; `CheckpointPolicy.bind` asserts that the resulting input widths equal the
  checkpoint's `obs_dims`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from callosum.configs.ippo import IPPOConfig
from callosum.training._agent_obs import AgentObsBuilder
from callosum.training._checkpoint import load_agent_weights, load_checkpoint
from callosum.training._ppo_core import ActorCritic


def run_name_of(checkpoint_path: str | Path) -> str:
    """Run name of a checkpoint file: its directory's name (`runs/<run>/best.pt` -> `<run>`)."""
    path = Path(checkpoint_path).resolve()
    return path.parent.name or path.stem


def rename_agent_keys(
    structured: Mapping, env_uids: Sequence[str], train_uids: Sequence[str]
) -> dict:
    """The structured observation with the `agent/<uid>` keys renamed `env_uids[i]` -> `train_uids[i]`.

    Order is preserved (ManiSkill flattens in insertion order); everything else is untouched, so
    the layout of the result is directly comparable with the trainer's. Raises if an env uid is
    missing from `structured["agent"]`.
    """
    mapping = dict(zip(env_uids, train_uids, strict=True))
    agent = structured["agent"]
    missing = [uid for uid in mapping if uid not in agent]
    if missing or set(agent) != set(mapping):
        raise KeyError(f"agent observation keys {list(agent)} do not match env uids {env_uids}")
    renamed = {mapping[uid]: value for uid, value in agent.items()}
    # Rebuild the dict key by key so that `agent` keeps its position among the top-level keys.
    return {key: (renamed if key == "agent" else value) for key, value in structured.items()}


def state_of(obs: Any) -> torch.Tensor:
    """The flat state vector of an env observation: `obs["state"]` for `"state+rgb"`, else `obs`."""
    flat = obs["state"] if isinstance(obs, Mapping) else obs
    return torch.as_tensor(flat)


@dataclasses.dataclass
class EpisodeSummary:
    """Running totals of one rolled-out episode (one env)."""

    steps: int = 0
    total_return: float = 0.0
    success_once: bool = False
    success: bool = False
    face_angle_deg: float | None = None
    max_face_angle_deg: float | None = None

    def update(self, reward: float, info: Mapping) -> None:
        """Account for one env step: its reward and the `info` dict it returned."""
        self.steps += 1
        self.total_return += float(reward)
        self.observe(info)

    def observe(self, info: Mapping) -> None:
        """Take the success flag and face angle out of an `info` dict (also the reset's)."""
        if "success" in info:
            self.success = bool(torch.as_tensor(info["success"]).reshape(-1)[0])
            self.success_once |= self.success
        if "face_angle" in info:
            deg = float(torch.rad2deg(torch.as_tensor(info["face_angle"]).reshape(-1)[0]))
            self.face_angle_deg = deg
            if self.max_face_angle_deg is None or deg > self.max_face_angle_deg:
                self.max_face_angle_deg = deg

    def status(self) -> str:
        """Short per-step text for the video footer."""
        parts = []
        if self.face_angle_deg is not None:
            parts.append(f"face {self.face_angle_deg:5.1f} deg")
        parts.append(f"success {self.success}")
        return "   ".join(parts)

    def text(self) -> str:
        """One-line episode summary."""
        angle = "n/a" if self.face_angle_deg is None else f"{self.face_angle_deg:.1f}"
        peak = "n/a" if self.max_face_angle_deg is None else f"{self.max_face_angle_deg:.1f}"
        return (
            f"success={self.success} (once: {self.success_once}), face angle {angle} deg "
            f"(max {peak}), return {self.total_return:.3f}, {self.steps} steps"
        )


class CheckpointPolicy:
    """Both IPPO actors of a checkpoint, ready to act in a (possibly different) env.

    Args:
        path: a `latest.pt` / `best.pt` written by the trainer.
        device: where the actors run (the env's device).
        deterministic: act with the actor mean (default) instead of sampling.

    Attributes:
        cfg: the training `IPPOConfig` stored in the checkpoint.
        run_name: name of the run directory the checkpoint came from.
        agents: the two `ActorCritic`s (agent_a first), in eval mode.
        train_uids: the agent uids the policies were trained on (`so101_pg-0`, ...).
    """

    def __init__(
        self, path: str | Path, device: torch.device | str = "cpu", deterministic: bool = True
    ) -> None:
        self.path = Path(path)
        self.deterministic = deterministic
        self.device = torch.device(device)
        payload = load_checkpoint(self.path, map_location="cpu")
        self.payload = payload
        self.cfg = IPPOConfig(**payload["config"])
        self.run_name = run_name_of(self.path)
        self.train_uids: tuple[str, ...] = tuple(payload["agent_uids"])
        self.obs_dims: list[int] = list(payload["obs_dims"])
        # The action width is not stored; it is the width of the actor's output (= logstd).
        action_dims = [
            int(payload["agents"][name]["actor_logstd"].shape[1]) for name in ("agent_a", "agent_b")
        ]
        self.action_dims = action_dims
        self.agents = [ActorCritic(d, a) for d, a in zip(self.obs_dims, action_dims, strict=True)]
        load_agent_weights(payload, self.agents)
        for agent in self.agents:
            agent.to(self.device).eval()
        self.builder: AgentObsBuilder | None = None
        self.env_uids: tuple[str, ...] = ()
        self._bounds: list[tuple[torch.Tensor, torch.Tensor]] = []

    def bind(
        self,
        structured: Mapping,
        flat_state: torch.Tensor,
        env_uids: Sequence[str],
        bounds: Sequence[tuple[Any, Any]],
    ) -> AgentObsBuilder:
        """Attach the policy to an env: build and verify the per-agent input slicing.

        Args:
            structured: the env's `get_obs(unflattened=True)` (needs `agent` and `extra`).
            flat_state: the env's flat state observation from the same moment (`state_of(obs)`).
            env_uids: the env's agent uids, agent_a first (may differ from the training uids).
            bounds: per agent, the `(low, high)` of its action space.

        Raises:
            ValueError: if the layout does not describe `flat_state`, or the input widths differ
                from the ones the checkpoint was trained with.
        """
        canonical = rename_agent_keys(structured, env_uids, self.train_uids)
        builder = AgentObsBuilder.from_env_obs(
            canonical, flat_state, self.cfg.partner_obs, self.train_uids
        )
        if builder.obs_dims != self.obs_dims:
            raise ValueError(
                f"policy inputs built from the env have widths {builder.obs_dims}, the checkpoint "
                f"was trained with {self.obs_dims}: the env's state observation differs from the "
                f"training env's (fields: {builder.fields})"
            )
        self.builder = builder
        self.env_uids = tuple(env_uids)
        self._bounds = [
            (torch.as_tensor(low, device=self.device), torch.as_tensor(high, device=self.device))
            for low, high in bounds
        ]
        for (low, _), dim in zip(self._bounds, self.action_dims, strict=True):
            if low.numel() != dim:
                raise ValueError(f"action space has {low.numel()} dims, the actor outputs {dim}")
        return builder

    @torch.no_grad()
    def act(self, flat_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """`{env_uid: action}` for a flat state observation (clamped to the action bounds)."""
        if self.builder is None:
            raise RuntimeError("call bind() before act()")
        inputs = self.builder(flat_state.to(self.device))
        return {
            uid: torch.clamp(agent.get_action(x, deterministic=self.deterministic), low, high)
            for uid, agent, x, (low, high) in zip(
                self.env_uids, self.agents, inputs, self._bounds, strict=True
            )
        }
