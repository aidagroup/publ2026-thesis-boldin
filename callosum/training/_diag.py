"""Generic averaging of the env's `diag_*` info keys over a rollout or an evaluation.

An env publishes diagnostics as per-env tensors in `info` under the prefix `callosum.envs._diag.
DIAG_PREFIX` (FaceTurn-v0: grasps, drifts, distances, reward terms; TwoSO101-v0: none). `DiagStats`
averages every such key over all envs and steps, tracks the maximum of the float ones, and for the
boolean ones (grasps) also the per-episode "ever" rate: the fraction of finished episodes in which
the flag was True at least once.

All accumulation stays on the simulation device as tensor sums, with no host sync per step; only
`take` (once per iteration or evaluation) converts to Python floats.

`ManiSkillVectorEnv` replaces `infos` by the reset info of the new episodes in a step where an
episode ended, and keeps the true last-step info in `infos["final_info"]`; `add` reads the final
values for those envs, so a step is always counted with the values of the state it ended in.
"""

import torch

from callosum.envs._diag import diag_name


class DiagStats:
    """Running means / maxima / per-episode "ever" rates of the diagnostics of one vector env.

    Args:
        num_envs: number of parallel envs (size of the per-env "ever" state).
        device: device of the env's tensors.
    """

    def __init__(self, num_envs: int, device: torch.device | str) -> None:
        self._num_envs = num_envs
        self._device = device
        self._ever: dict[str, torch.Tensor] = {}  # per env, spans calls to `take`
        self._reset_sums()

    def _reset_sums(self) -> None:
        zero = torch.zeros((), device=self._device)
        self._sums: dict[str, torch.Tensor] = {}
        self._maxes: dict[str, torch.Tensor] = {}
        self._count = zero.clone()  # env-steps counted
        self._ever_sums: dict[str, torch.Tensor] = {}
        self._episodes = zero.clone()  # finished episodes counted for the "ever" rates

    def add(self, infos: dict, valid: torch.Tensor | None = None) -> None:
        """Accumulate one `envs.step` / `eval_envs.step` worth of diagnostics.

        Args:
            infos: the info dict returned by the vector env.
            valid: optional `(num_envs,)` bool; envs that are False (non-finite state, see
                `callosum.training._nonfinite`) contribute nothing and lose their running "ever"
                flags. `None` means all envs (the normal, sync-free path).
        """
        final_mask = infos.get("_final_info")
        final_info = infos.get("final_info")
        found = False
        for key, value in infos.items():
            name = diag_name(key)
            if name is None:
                continue
            found = True
            if final_mask is not None and final_info is not None and key in final_info:
                value = torch.where(final_mask, final_info[key], value)
            self._add_one(name, value, valid, final_mask)
        if found:
            self._count = self._count + (
                float(self._num_envs) if valid is None else valid.sum().float()
            )
            if final_mask is not None:
                finished = final_mask if valid is None else final_mask & valid
                self._episodes = self._episodes + finished.sum().float()

    def _add_one(
        self,
        name: str,
        value: torch.Tensor,
        valid: torch.Tensor | None,
        final_mask: torch.Tensor | None,
    ) -> None:
        is_flag = value.dtype == torch.bool
        x = value.float()
        zero = torch.zeros((), device=x.device)
        masked = x if valid is None else torch.where(valid, x, zero)
        self._sums[name] = self._sums.get(name, zero) + masked.sum()
        if not is_flag:
            top = (
                x if valid is None else torch.where(valid, x, torch.full_like(x, -torch.inf))
            ).max()
            self._maxes[name] = torch.maximum(self._maxes.get(name, top), top)
            return
        ever = self._ever.get(name)
        if ever is None:
            ever = torch.zeros(self._num_envs, dtype=torch.bool, device=value.device)
        ever = ever | (value if valid is None else value & valid)
        if valid is not None:
            ever = ever & valid
        if final_mask is not None:
            finished = final_mask if valid is None else final_mask & valid
            self._ever_sums[name] = (
                self._ever_sums.get(name, zero) + (ever & finished).sum().float()
            )
            ever = ever & ~final_mask
        self._ever[name] = ever

    def take(self) -> tuple[dict[str, float], dict[str, float]]:
        """`(means, maxes)` since the last `take`, then clear the sums (not the "ever" flags).

        `means` maps a diagnostic's short name to its mean over all counted env-steps, plus
        `ever_<name>` for the boolean ones (over the finished episodes, absent if none finished).
        `maxes` holds the maximum of the non-boolean ones. Both are empty if nothing was added.
        """
        means: dict[str, float] = {}
        maxes: dict[str, float] = {}
        count = float(self._count)
        if count > 0:
            means = {name: float(total) / count for name, total in self._sums.items()}
            maxes = {
                name: float(top) for name, top in self._maxes.items() if float(top) > -float("inf")
            }
        episodes = float(self._episodes)
        if episodes > 0:
            for name, total in self._ever_sums.items():
                means["ever_" + name] = float(total) / episodes
        self._reset_sums()
        return means, maxes
