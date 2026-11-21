"""Vectorised coop suite: N single-instance MuJoCo envs stepped in a Python loop.

`callosum.experiments.coop_tasks.base.CoopTaskEnv` is single-instance and CPU-only (the vectorisation is
the trainer's job, exactly as for `CubeWrench-v0`), and the plan keeps the coop suite on the
CPU loop until the GPU-server throughput says otherwise. This module is that loop: it drives
`num_envs` copies of one coop task, so `callosum.experiments.cube_wrench.train` can train the new
tasks with the same PPO code and the same vec-env contract as `VecWrenchCube`.

It is the only place in the training package that imports the coop simulator package:
`callosum.experiments.cube_wrench.train` imports this module lazily inside `run()`, so the trainer's
pure-torch parts (policies, GAE, the PPO step, checkpoints) stay importable on macOS/CI
without the `sandbox` extra.

The wrapper owns the same auto-reset contract as `_vec_wrench.VecWrenchCube`:

* `step` returns the observation for the NEXT step (a done env is reset immediately), so
  the trainer never sees a terminal transition;
* `final_obs` keeps the episode's true last observation (the truncation bootstrap of GAE);
* the finished episodes' return / length / success / final diagnostic are collected for the
  trainer's `train/*` logs, and `infos` keeps every env's last step info (`finished_angles`
  is the task's primary diagnostic: the pole angle, the pendulum amplitude or the pass
  count).

Episode seeds are explicit: `reset(seed)` seeds env `i` with `seed + i`, and every
auto-reset continues one global counter, so the envs do not replay the same episode
randomisation and a run is reproducible from its base seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from callosum.experiments.coop_tasks import make_coop_env
from callosum.experiments.coop_tasks.config import coop_config
from callosum.experiments.coop_tasks.logic import (
    AGENT_KEYS,
    task_diagnostic,
    validate_partner_mode,
)


@dataclass
class VecCoopStep:
    """One control step of all coop envs (the auto-reset contract is in the module docstring).

    The field names mirror `callosum.experiments.cube_wrench.vec_env.VecStep`, so
    `wrench_ppo.evaluate_greedy`/`train` consume both backends through the same duck-typed
    contract (a coop env defines no `torch_io`, so the trainer treats it as numpy, like the
    cube CPU backend).

    Attributes:
        obs: next-step observation per agent, `(num_envs, obs_dim)`; done envs already
            restarted, so `obs` is always the input of the next policy call.
        reward: shared team reward, `(num_envs,)`.
        terminated: success/failure flags, `(num_envs,)` bool.
        truncated: time-limit flags, `(num_envs,)` bool.
        final_obs: per-agent observation at the end of the episode, `(num_envs, obs_dim)`;
            only the rows of done envs are meaningful (the truncation bootstrap reads the
            `truncated` ones).
        infos: every env's `step` info dict (success plus the task diagnostics).
        finished_returns: undiscounted returns of the episodes that ended on this step.
        finished_lengths: lengths (control steps) of those episodes.
        finished_successes: success flags of those episodes.
        finished_angles: final primary diagnostic of those episodes (`angle_deg` for the
            cart-pole, `amplitude_deg` for the pendulum, `passes` for the toss-catch); the
            field name is kept for the shared trainer contract.
    """

    obs: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    final_obs: dict[str, np.ndarray]
    infos: list[dict] = field(default_factory=list)
    finished_returns: list[float] = field(default_factory=list)
    finished_lengths: list[int] = field(default_factory=list)
    finished_successes: list[bool] = field(default_factory=list)
    finished_angles: list[float] = field(default_factory=list)

    @property
    def done(self) -> np.ndarray:
        """`terminated | truncated`, `(num_envs,)`."""
        return np.logical_or(self.terminated, self.truncated)


class VecCoopEnv:
    """`num_envs` envs of one coop task with auto-reset, stepped one env at a time."""

    def __init__(
        self,
        task: str,
        num_envs: int,
        partner_mode: str = "none",
        config=None,
        seed: int = 0,
    ) -> None:
        """Build the envs and their per-env running episode accumulators.

        Args:
            task: The coop task (`"cartpole"`, `"pendulum"` or `"toss"`).
            num_envs: Number of envs (each is a separate MuJoCo model/data pair).
            partner_mode: Observation visibility mode, `"none"` or `"oracle"`.
            config: The task config; the task's default config when `None`.
            seed: Base episode seed; `reset(seed)` seeds env `i` with `seed + i` and the
                auto-resets continue from `seed + num_envs`.
        """
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        validate_partner_mode(partner_mode)
        self.task = task
        self.num_envs = num_envs
        self.partner_mode = partner_mode
        self.config = config if config is not None else coop_config(task)
        self._base_seed = int(seed)
        self._next_seed = self._base_seed + num_envs
        self._envs = [
            make_coop_env(task, config=self.config, partner_mode=partner_mode)
            for _ in range(num_envs)
        ]
        self.obs_dim = int(self._envs[0].observation_space[AGENT_KEYS[0]].shape[0])
        self.action_dim = int(self._envs[0].action_space[AGENT_KEYS[0]].shape[0])
        self._obs = {
            key: np.zeros((num_envs, self.obs_dim), dtype=np.float32) for key in AGENT_KEYS
        }
        self._episode_return = np.zeros(num_envs, dtype=np.float64)
        self._episode_length = np.zeros(num_envs, dtype=np.int64)

    def _empty_obs(self) -> dict[str, np.ndarray]:
        """Zero-filled observation buffers for both agents."""
        return {
            key: np.zeros((self.num_envs, self.obs_dim), dtype=np.float32) for key in AGENT_KEYS
        }

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset every env and return the first observation of the new episodes.

        Args:
            seed: Base episode seed; `None` uses the wrapper's base seed. Env `i` is reset
                with `seed + i`.
        """
        base = self._base_seed if seed is None else int(seed)
        for index, env in enumerate(self._envs):
            obs, _info = env.reset(seed=base + index)
            self._store_obs(obs, index)
            self._episode_return[index] = 0.0
            self._episode_length[index] = 0
        return {key: self._obs[key].copy() for key in AGENT_KEYS}

    def step(self, actions: dict[str, np.ndarray]) -> VecCoopStep:
        """Step all envs once, auto-reset the done ones and collect the finished episodes.

        Args:
            actions: normalized one-sided commands per role, `(num_envs, action_dim)` each.
        """
        step_actions = self._parse_actions(actions)
        obs_next = self._empty_obs()
        final_obs = self._empty_obs()
        reward = np.zeros(self.num_envs, dtype=np.float64)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict] = []
        finished_returns: list[float] = []
        finished_lengths: list[int] = []
        finished_successes: list[bool] = []
        finished_angles: list[float] = []
        for index, env in enumerate(self._envs):
            obs, step_reward, env_terminated, env_truncated, info = env.step(
                {key: step_actions[key][index] for key in AGENT_KEYS}
            )
            reward[index] = step_reward
            terminated[index] = env_terminated
            truncated[index] = env_truncated
            infos.append(info)
            self._episode_return[index] += step_reward
            self._episode_length[index] += 1
            if not (env_terminated or env_truncated):
                self._store_obs(obs, index)
                for key in AGENT_KEYS:
                    obs_next[key][index] = obs[key]
                continue
            for key in AGENT_KEYS:
                final_obs[key][index] = obs[key]
            finished_returns.append(float(self._episode_return[index]))
            finished_lengths.append(int(self._episode_length[index]))
            finished_successes.append(bool(info["success"]))
            finished_angles.append(task_diagnostic(self.task, info))
            # Auto-reset: the next policy call must see a fresh episode, while `final_obs`
            # (above) keeps the true last observation for the truncation bootstrap.
            obs, _info = env.reset(seed=self._next_seed)
            self._next_seed += 1
            self._episode_return[index] = 0.0
            self._episode_length[index] = 0
            self._store_obs(obs, index)
            for key in AGENT_KEYS:
                obs_next[key][index] = obs[key]
        return VecCoopStep(
            obs=obs_next,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            final_obs=final_obs,
            infos=infos,
            finished_returns=finished_returns,
            finished_lengths=finished_lengths,
            finished_successes=finished_successes,
            finished_angles=finished_angles,
        )

    def close(self) -> None:
        """Close every env (a no-op for the MuJoCo env, kept for symmetry with gymnasium)."""
        for env in self._envs:
            env.close()

    def _store_obs(self, obs: dict[str, np.ndarray], index: int) -> None:
        """Copy one env's observation dict into the wrapper's per-env buffers."""
        for key in AGENT_KEYS:
            self._obs[key][index] = obs[key]

    def _parse_actions(self, actions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Validate and copy the per-role action arrays."""
        if set(actions) != set(AGENT_KEYS):
            raise ValueError(f"action keys must be exactly {AGENT_KEYS}, got {sorted(actions)}")
        parsed: dict[str, np.ndarray] = {}
        for key in AGENT_KEYS:
            array = np.asarray(actions[key], dtype=np.float64)
            expected = (self.num_envs, self.action_dim)
            if array.shape != expected:
                raise ValueError(f"action[{key!r}] must have shape {expected}, got {array.shape}")
            parsed[key] = array
        return parsed
