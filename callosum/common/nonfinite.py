"""Guard against single envs whose GPU PhysX state blew up (non-finite observation or reward).

Seen in a 30M-step FaceTurn run: at iteration 725 a few of the 256 envs produced non-finite
observations, `Normal(loc=...)` raised inside the rollout and the whole run died. The network was
fine, only those envs' simulation state was. `guard_step` finds such envs after an `envs.step`,
resets just them through the vector env's partial reset, and tells the caller which envs to
treat as terminal with zero reward and no bootstrap (see `ippo.py`).

How the partial reset works (ManiSkill v3.0.1): `ManiSkillVectorEnv.reset(options={"env_idx":
idx})` calls `BaseEnv.reset`, which re-initialises only those envs (`_initialize_episode(idx)`,
`_elapsed_steps[idx] = 0`, velocities cleared through the reset mask) and returns the observation
of ALL envs; the wrapper then zeroes its `record_metrics` accumulators (`returns`, `success_once`,
`fail_once`) for those envs, which is what drops a NaN return from the episode statistics. This
is the same call the wrapper makes itself when an episode ends.

Only the cheap check (an `isfinite` reduction and one `.any()` host sync) runs when everything is
finite.
"""

import dataclasses
import math
from collections.abc import Mapping, Sequence

import torch


def nonfinite_mask(tensor: torch.Tensor) -> torch.Tensor:
    """`(n,)` bool: True where any value of env `i`'s row (or the scalar for a 1-D tensor) is
    non-finite."""
    bad = ~torch.isfinite(tensor)
    return bad if bad.ndim == 1 else bad.reshape(bad.shape[0], -1).any(dim=1)


def nonfinite_fields(layout: Sequence[tuple[tuple[str, ...], int]], row: torch.Tensor) -> list[str]:
    """Names (`agent/so101_pg-0/qpos`, ...) of the observation fields with a non-finite entry.

    Args:
        layout: `AgentObsBuilder.layout`, `[(path, width), ...]` in flat-observation order.
        row: one env's flat observation `(D,)`.
    """
    bad = (~torch.isfinite(row)).tolist()
    names, offset = [], 0
    for path, width in layout:
        if any(bad[offset : offset + width]):
            names.append("/".join(path))
        offset += width
    return names


def _field_values(
    layout: Sequence[tuple[tuple[str, ...], int]], row: torch.Tensor, wanted: Sequence[str]
) -> dict[str, list[float]]:
    """The values of the named observation fields of one flat row."""
    values = row.tolist()
    out, offset = {}, 0
    for path, width in layout:
        name = "/".join(path)
        if name in wanted:
            out[name] = [
                round(v, 4) if math.isfinite(v) else v for v in values[offset : offset + width]
            ]
        offset += width
    return out


@dataclasses.dataclass
class GuardResult:
    """What `guard_step` found and did (only returned when some env was non-finite).

    Attributes:
        obs: the flat observation with the poisoned envs replaced by their post-reset rows.
        reward: the reward with the poisoned envs set to 0.
        poisoned: `(n,)` bool, the envs to treat as terminal (done, zero reward, no bootstrap, not
            in the episode statistics). Includes envs that were auto-reset in this very step
            because their episode ended while their last step was non-finite.
        report: a multi-line description of the first poisoned envs (empty unless requested).
    """

    obs: torch.Tensor
    reward: torch.Tensor
    poisoned: torch.Tensor
    report: str

    @property
    def count(self) -> int:
        """Number of poisoned envs."""
        return int(self.poisoned.sum())


def guard_step(
    envs,
    obs: torch.Tensor,
    reward: torch.Tensor,
    infos: Mapping,
    *,
    layout: Sequence[tuple[tuple[str, ...], int]] = (),
    actions: Mapping[str, torch.Tensor] | None = None,
    describe: bool = False,
) -> GuardResult | None:
    """Detect non-finite envs after `envs.step` and reset them; `None` if all are finite.

    An env is poisoned if its observation row or reward is non-finite, or, when its episode just
    ended and the wrapper auto-reset it, if its true last observation (`infos["final_observation"]`)
    is. Poisoned envs that were not already auto-reset get a partial reset (see module docstring).
    Rows that are still non-finite after the reset are zeroed (the env is reset again on the next
    step); if every env is still non-finite the simulation is beyond saving and this raises.

    Args:
        envs: the `ManiSkillVectorEnv` that produced `obs`.
        layout: `AgentObsBuilder.layout`, only used for the report.
        actions: the actions just applied, only used for the report.
        describe: build `GuardResult.report` (host copies; pass True only for the first event).
    """
    bad_obs = nonfinite_mask(obs)
    bad_reward = nonfinite_mask(reward)
    autoreset = infos.get("_final_info")
    bad_final = None
    if autoreset is not None and "final_observation" in infos:
        bad_final = nonfinite_mask(infos["final_observation"]) & autoreset
    poisoned = bad_obs | bad_reward
    if bad_final is not None:
        poisoned = poisoned | bad_final
    if not bool(poisoned.any()):  # the only host sync of the all-finite path
        return None

    report = ""
    if describe:
        report = _report(
            poisoned, bad_obs, bad_reward, bad_final, obs, reward, infos, layout, actions
        )

    # Envs that were auto-reset in this step already have a fresh state (their obs row is the
    # post-reset one); every other poisoned env, and any whose fresh obs is itself bad, needs one.
    needs_reset = poisoned if autoreset is None else bad_obs | (poisoned & ~autoreset)
    if bool(needs_reset.any()):
        fresh, _ = envs.reset(options={"env_idx": torch.nonzero(needs_reset).flatten()})
        obs = torch.where(needs_reset.unsqueeze(1), fresh, obs)
    still_bad = nonfinite_mask(obs)
    if bool(still_bad.all()):
        raise FloatingPointError("every env is non-finite even after a partial reset")
    if bool(still_bad.any()):
        obs = torch.where(still_bad.unsqueeze(1), torch.zeros_like(obs), obs)
        report += f"\n  {int(still_bad.sum())} env(s) still non-finite after the reset: zeroed"
    reward = torch.where(poisoned, torch.zeros_like(reward), reward)
    return GuardResult(obs=obs, reward=reward, poisoned=poisoned, report=report)


_REPORT_FIELDS = ("extra/face_angle", "extra/cube_pose", "extra/face_pose")
_MAX_REPORTED_ENVS = 6


def _report(poisoned, bad_obs, bad_reward, bad_final, obs, reward, infos, layout, actions) -> str:
    """Describe the first poisoned envs: which signal, which obs fields, task-state values."""
    lines = []
    for i in torch.nonzero(poisoned).flatten().tolist()[:_MAX_REPORTED_ENVS]:
        sources = [
            name
            for name, mask in (("obs", bad_obs), ("reward", bad_reward), ("final_obs", bad_final))
            if mask is not None and bool(mask[i])
        ]
        # The row that carried the non-finite values: the last obs of an auto-reset episode
        # lives in `final_observation`, the current one otherwise.
        row = infos["final_observation"][i] if "final_obs" in sources else obs[i]
        lines.append(f"  env {i}: non-finite {'+'.join(sources)}, reward {float(reward[i])}")
        if layout:
            lines.append(f"    non-finite obs fields: {nonfinite_fields(layout, row) or '-'}")
            lines.append(f"    values: {_field_values(layout, row, _REPORT_FIELDS)}")
        if actions is not None:
            for uid, action in actions.items():
                lines.append(f"    action {uid}: {[round(v, 3) for v in action[i].tolist()]}")
    hidden = int(poisoned.sum()) - _MAX_REPORTED_ENVS
    if hidden > 0:
        lines.append(f"  ... and {hidden} more")
    return "\n".join(lines)


class NonFiniteTracker:
    """Counts poisoned envs (training and evaluation separately) and reports the first event.

    Attributes:
        totals: cumulative number of poisoned envs per source (`"train"`, `"eval"`).
        reported: whether the one-time diagnostic was already printed in this process.
    """

    def __init__(self) -> None:
        self.totals = {"train": 0, "eval": 0}
        self.reported = False

    def record(self, source: str, result: GuardResult, where: str) -> None:
        """Count `result.count` envs for `source`; print the report on the first event only.

        Args:
            source: `"train"` or `"eval"`.
            result: the `guard_step` result (its `report` is empty after the first event).
            where: context for the message, e.g. `"iteration 725, rollout step 31"`.
        """
        self.totals[source] += result.count
        if result.report and not self.reported:
            self.reported = True
            print(
                f"[nonfinite] first occurrence ({source}, {where}): {result.count} env(s) had a "
                "non-finite observation/reward; reset them, zeroed their reward and cut GAE "
                f"there. Details (for the root-cause hunt):\n{result.report}",
                flush=True,
            )
