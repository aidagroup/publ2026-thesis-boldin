"""BenchMARL Task contract bridge for callosum's FaceTurn-v0 / TwoSO100-v0 envs.

Registers the already-registered ManiSkill gymnasium envs as BenchMARL
``Task`` entries so the built-in ``IPPO``/``MAPPO`` trainers
(``benchmarl.algorithms.ippo.Ippo`` / ``mappo.Mappo``) can run them.

Environment contract (server-only to materialise):
    - The gym ids ``FaceTurn-v0`` / ``TwoSO100-v0`` are registered by
      ``callosum.envs.face_turn`` / ``callosum.envs.two_so100_base`` via
      ``mani_skill.utils.registration.register_env`` -- importing those modules
      is the side effect that creates the ids. Those imports pull mani_skill
      (SAPIEN, Linux+CUDA only), so they are DEFERRED to inside
      ``get_env_fun`` / ``_make_gym_env``; this module stays importable on
      macOS/CI where only ``benchmarl``+``torchrl``+``torch`` are present.

BenchMARL version note (important):
    The user-facing contract referenced as ``benchmarl/tasks/task.py`` with
    ``GymTask``/``TorchRLTask``/`get_obs_act_spaces`/`supports_rewards` does
    NOT exist in the pinned BenchMARL 1.5.2 (uv.lock). The structure has always
    lived in ``benchmarl/environments/common.py`` as ``Task`` (Enum) +
    ``TaskClass`` (abstract base). The 12 abstract methods of ``TaskClass`` are
    implemented below -- see ``CallosumTaskClass``.

What this skeleton does vs. what is server-only:
    - Concrete (importable on dev): ``TaskConfig`` schemas, the gym-space ->
      BenchMARL ``Composite`` spec builders, ``CallosumTaskClass`` (all 12
      abstract overrides) and ``CallosumTask`` (the enum), plus the
      side-effect registry patch in :func:`register`.
    - Server-only (deferred imports / ``TODO(review)``): materialising
      the gym env, the gymnasium -> torchrl EnvBase bridge that maps the
      per-agent dict actions/observations into BenchMARL's grouped
      ``(group, "observation")`` / ``(group, "action")`` TensorDict layout,
      and the auto-reset / final_observation / ``RewardSum`` interplay.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import MISSING, dataclass
from typing import Any

import torch
from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING
from tensordict import TensorDictBase
from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase

# ---------------------------------------------------------------------------
# Task group / agent identity
# ---------------------------------------------------------------------------
# One shared group holds both arms. IPPO gives each arm its own (independent)
# critic within the group (centralised=False); MAPPO routes the centralized
# critic through ``state_spec``. Use share_param_critic=False in IPPO to get
# per-arm parameters, mirroring callosum's decoupled two-agent trainer
# (``callosum.training.ippo`` builds two separate Actor networks).
GROUP_NAME = "agents"

# env_id -> (gym register name). FaceTurn-v0 / TwoSO100-v0 are registered by
# importing callosum.envs.face_turn / callosum.envs.two_so100_base (server only).
_ENV_IDS: dict[str, str] = {"face_turn": "FaceTurn-v0", "two_so100": "TwoSO100-v0"}

# Default max episode length; the gym id carries max_episode_steps=100 (see
# the @register_env(..., max_episode_steps=100) on both envs), read back via
# mani_skill.utils.gym_utils.find_max_episode_steps_value at runtime.
_DEFAULT_MAX_STEPS = 100


# ---------------------------------------------------------------------------
# (Optional) config schemas -- mirror the vmas/pettingzoo TaskConfig pattern.
# These let Hydra type-check the ``conf/task/callosum/<task>.yaml`` fields.
# ---------------------------------------------------------------------------
@dataclass
class FaceTurnTaskConfig:
    task: str = MISSING
    max_steps: int = _DEFAULT_MAX_STEPS
    # Mirror the env's partner_obs ablation toggle (callosum.configs.face_turn
    # does not carry it; it is a TwoSO100Base ctor kwarg). Exposed here so the
    # hydra config can pick the oracle/none condition.
    partner_obs: str = "full"


@dataclass
class TwoSO100TaskConfig:
    task: str = MISSING
    max_steps: int = _DEFAULT_MAX_STEPS
    partner_obs: str = "full"


_TASK_CONFIG_FOR_TASK = {"face_turn": FaceTurnTaskConfig, "two_so100": TwoSO100TaskConfig}


# ---------------------------------------------------------------------------
# gymnasium Dict-space -> BenchMARL grouped Composite spec builders.
# ---------------------------------------------------------------------------
def _flat_dim(space: Any) -> int:
    """Product of leaf sizes of a gymnasium space (Dict/Box/Box, no numpy).

    Mirrors mani_skill.utils.common.flatten_state_dict's concatenation order:
    it recurses the gym ``Dict``/``spaces`` mapping in insertion order and sums
    the leaf element counts. No ``gym``/``numpy`` import is needed -- the gym
    space object is duck-typed (``.spaces`` for Dict, ``.shape``/``.n`` for
    leaf spaces), so this is importable and unit-testable on macOS/CI.
    """
    if hasattr(space, "spaces"):
        return sum(_flat_dim(s) for s in space.spaces.values())
    if hasattr(space, "n"):  # Discrete
        return int(space.n)
    return int(math.prod(space.shape)) if tuple(space.shape) else 1


def _per_agent_obs_dim(single_observation_space: Any, agent_uid: str, agent_idx: int = 0) -> int:
    """Flat dim of ONE agent's DECENTRALIZED policy input (proprio + own extra).

    Under ``obs_mode="state_dict"`` the gym space is
    ``Dict({"agent": {uid: {qpos, qvel}}, "extra": {...}})``. This mirrors
    ``callosum.training._agent_obs.select_agent_extra_fields`` on the gym
    *space*: the partner's ``agent_a_*``/``agent_b_*`` TCP fields are DROPPED
    so the IPPO actor never ingests the partner's raw pose -- the previous
    version summed the ENTIRE ``extra`` block, leaking the partner TCP into
    the supposedly-decentralized actor input (review Б4). The centralized
    critic still sees the full state via :func:`_state_spec` (unchanged); that
    actor/critic asymmetry is the CTDE contract. ``agent_idx`` picks which
    agent (0 or 1) the dim is computed for, hence which partner prefix to drop,
    and keeps obs_dim identical whether ``partner_obs`` is ``"full"`` or
    ``"none"`` (when "none", the env emits no partner-prefixed fields, so the
    drop is a no-op -- Л5).
    """
    agent_space = single_observation_space["agent"][agent_uid]
    extra_space = single_observation_space.get("extra", {})
    # Partner-prefix rule matches _OTHER_AGENT_PREFIX in _agent_obs.py:
    # agent 0 drops "agent_b_*" (its partner), agent 1 drops "agent_a_*".
    other_prefix = "agent_b_" if agent_idx == 0 else "agent_a_"
    extra_spaces = getattr(extra_space, "spaces", {}) or {}
    own_extra = {k: v for k, v in extra_spaces.items() if not k.startswith(other_prefix)}
    return _flat_dim(agent_space) + sum(_flat_dim(s) for s in own_extra.values())


def _grouped_observation_spec(
    single_observation_space: Any, agent_uids: list[str], group: str = GROUP_NAME
) -> Composite:
    """BenchMARL observation spec: one ``(group, "observation")`` entry of shape
    ``(n_agents, D)`` -- the agent dimension BenchMARL's model expects
    (``input_has_agent_dim=True``)."""
    n_agents = len(agent_uids)
    d = _per_agent_obs_dim(single_observation_space, agent_uids[0])
    return Composite(
        {group: Composite({"observation": Unbounded(shape=(n_agents, d))}, shape=(n_agents,))},
        shape=(),
    )


def _grouped_action_spec(
    single_action_space: Any, agent_uids: list[str], group: str = GROUP_NAME
) -> Composite:
    """BenchMARL action spec: ``(group, "action")`` of shape ``(n_agents, A)``."""
    n_agents = len(agent_uids)
    a = _flat_dim(single_action_space[agent_uids[0]])
    return Composite(
        {group: Composite({"action": Unbounded(shape=(n_agents, a))}, shape=(n_agents,))},
        shape=(),
    )


def _state_spec(single_observation_space: Any, agent_uids: list[str]) -> Composite:
    """Centralized-critic (``state``) spec for MAPPO.

    Returns ``Composite({"state": Unbounded(shape=(D_state,))})`` -- exactly ONE
    top-level key (BenchMARL's ``Algorithm._check_specs`` requires this). It is
    the global state fed to MAPPO's centralized critic
    (``Mappo.get_critic``: ``input_has_agent_dim=False, centralised=True``);
    IPPO ignores it (independent critic over ``observation_spec[group]``).

    ``D_state`` is the flat dim of the *whole* state_dict observation
    (both agents' proprio + the shared extra). IPPO callers can pass
    ``state_spec=None`` to disable the centralized critic entirely; see
    :meth:`CallosumTaskClass.state_spec`.
    """
    d_state = _flat_dim(single_observation_space)
    return Composite({"state": Unbounded(shape=(d_state,))})


# ---------------------------------------------------------------------------
# gymnasium -> torchrl EnvBase bridge (SERVER-ONLY instantiation).
# ---------------------------------------------------------------------------
class _CallosumManiSkillEnv(EnvBase):
    """torchrl EnvBase over a vectorised ManiSkill gymnasium env.

    Bridges the gymnasium multi-agent dict API into BenchMARL's grouped
    TensorDict layout:

      * gym observation ``{"agent": {uid: {qpos,qvel}}, "extra": {...}}``
        -> ``("next", GROUP_NAME, "observation")`` of shape ``(B, n_agents, D)``
      * gym action ``{uid: array(B, A)}``
        <- ``(GROUP_NAME, "action")`` of shape ``(B, n_agents, A)``   [the
        dict-action mapping the task asked for]
      * shared reward -> ``("next", "reward")`` (BenchMARL expands this to
        ``("next", group, "reward")`` per group in ``Algorithm.process_batch``)
      * global state -> ``("next", "state")`` (MAPPO centralized critic)

    ``batch_size`` is ``(num_envs,)`` (the ManiSkill GPU sim is already
    vectorised), so BenchMARL uses the env directly -- no SerialEnv wrap.

    NOTE: this class is only INSTANTIATED on the server (mani_skill/SAPIEN are
    Linux+CUDA only). Its spec builders and the dict<->TensorDict mapping are
    sketched here to show the exact overrides; the gymnasium auto-reset /
    final_observation / RewardSum interplay is flagged ``# TODO(review)``.
    """

    batch_locked: bool = True  # num_envs fixed at construction

    def __init__(
        self,
        gym_env: Any,
        num_envs: int,
        agent_uids: list[str],
        max_steps: int = _DEFAULT_MAX_STEPS,
        **_: Any,
    ) -> None:
        # num_envs fixed at construction -> batched GPU sim, no SerialEnv wrap.
        super().__init__(
            device=getattr(gym_env, "device", "cpu"),
            batch_size=torch.Size([num_envs]),
        )
        self._gym = gym_env  # gymnasium VectorEnv (ManiSkillVectorEnv)
        self.num_envs = num_envs
        self.agent_uids = list(agent_uids)
        self.max_steps_val = max_steps
        self._build_specs()

    # -- attribute pass-through the TaskClass spec builders rely on ----------
    @property
    def single_observation_space(self) -> Any:
        return self._gym.single_observation_space  # gym Dict, see env contract

    @property
    def single_action_space(self) -> Any:
        return self._gym.single_action_space  # gym Dict keyed by uid

    @property
    def group_map(self) -> dict[str, list[str]]:
        return {GROUP_NAME: list(self.agent_uids)}

    def _build_specs(self) -> None:
        uids = self.agent_uids
        n_agents = len(uids)
        d = _per_agent_obs_dim(self.single_observation_space, uids[0])
        d_state = _flat_dim(self.single_observation_space)
        # full_observation_spec holds BOTH the grouped per-agent observation AND
        # the global "state" (TorchRL treats a top-level "state" key as the env's
        # full state / centralized-critic input). BenchMARL's Task.state_spec
        # extracts "state" into its own single-key spec; Task.observation_spec
        # exposes only the grouped (agent) part to the actor. Mirror TicTacToeEnv.
        self.full_observation_spec = Composite(
            {
                GROUP_NAME: Composite(
                    {"observation": Unbounded(shape=(n_agents, d))}, shape=(n_agents,)
                ),
                "state": Unbounded(shape=(d_state,)),
            },
            shape=(),
        )
        self.action_spec = _grouped_action_spec(self.single_action_space, uids)
        # Shared (not per-group): one reward + one done/terminated per env.
        # BenchMARL's IPPO/MAPPO.process_batch expands the shared ("next","reward")
        # and ("next","done") into per-group keys internally.
        self.reward_spec = Composite({"reward": Unbounded(shape=(1,))})
        self.full_done_spec = Composite(
            {
                "done": Unbounded(shape=(1,)),
                "terminated": Unbounded(shape=(1,)),
                "truncated": Unbounded(shape=(1,)),
            }
        )

    # -- the gym dict <-> grouped TensorDict mapping -------------------------
    def _agent_obs(self, obs: dict) -> torch.Tensor:
        """Stack each agent's DECENTRALIZED flat obs (proprio + own extra only)
        into ``(B, n_agents, D)``, dimension-matched to :func:`_per_agent_obs_dim`.

        Uses ``build_agent_obs(include_partner=False)`` so the partner's raw
        TCP pose is dropped from the IPPO actor input (review Б4: the previous
        version concatenated the FULL ``extra`` block -- including the partner
        TCP -- into every agent's obs, leaking CTDE partner info into the
        decentralized baseline). The centralized critic still sees the full
        state via ``_global_state``; that actor/critic split is the CTDE
        contract.
        """
        from callosum.training._agent_obs import build_agent_obs  # torch-only, dev-safe

        parts = [
            build_agent_obs(obs, i, tuple(self.agent_uids), include_partner=False)
            for i in range(len(self.agent_uids))
        ]
        return torch.stack(parts, dim=1)  # (B, n_agents, D)

    def _global_state(self, obs: dict) -> torch.Tensor:
        """Shared centralized state: concat of all agents' flat obs (B, D_state)."""
        from callosum.training._agent_obs import flatten_dict_to_tensor  # dev-safe

        return flatten_dict_to_tensor(obs)  # flattens agent + extra in order

    def _reset(self, reset_td: TensorDictBase) -> TensorDictBase:
        # TODO(review): ManiSkillVectorEnv.reset returns a torch obs dict;
        # gymnasium 1.3.0 auto-reset bookkeeping is handled by the wrapper, but
        # the final-observation / ResetInfo semantics must align with _step
        # output below. Convention (cf. torchrl's TicTacToeEnv): _reset returns
        # TOP-LEVEL keys -- EnvBase.step/reset nest them under "next".
        obs, _info = self._gym.reset()
        B = reset_td.shape if reset_td.shape else torch.Size([self.num_envs])
        td = self.full_observation_spec.zero(B)
        td.set((GROUP_NAME, "observation"), self._agent_obs(obs))
        td.set("state", self._global_state(obs))
        td.update(self.full_done_spec.zero(B))  # done=terminated=truncated=False
        return td

    def _step(self, state: TensorDictBase) -> TensorDictBase:
        # --- maps the multi-agent dict actions INTO the env (the asked part) ---
        action = state.get((GROUP_NAME, "action"))  # (B, n_agents, A), top-level
        gym_actions = {uid: action[:, i].cpu().numpy() for i, uid in enumerate(self.agent_uids)}
        # TODO(review): reward/done shapes -- ManiSkillVectorEnv.step
        # returns rew (B,), term (B,) bool, trunc (B,) bool, infos dict.
        obs, rew, term, trunc, _infos = self._gym.step(gym_actions)

        B = state.batch_size
        next_td = self.full_observation_spec.zero(B)
        next_td.set(
            (GROUP_NAME, "observation"), self._agent_obs(obs)
        )  # -> ("next",group,"observation")
        next_td.set("state", self._global_state(obs))  # -> ("next","state")
        # TODO(review): bool dtype on done (torchrl prefers bool here).
        done = (term | trunc).to(torch.bool).unsqueeze(-1)  # (B, 1)
        next_td.set("reward", rew.to(torch.float32).unsqueeze(-1))  # shared -> ("next","reward")
        next_td.set("done", done)
        next_td.set("terminated", term.to(torch.bool).unsqueeze(-1))
        next_td.set("truncated", trunc.to(torch.bool).unsqueeze(-1))
        # TODO(review): final_observation / episode info wiring into the
        # shared reward (RewardSum transform) and the next_done bootstrap that
        # IPPO/MAPPO GAE relies on.
        return next_td

    def _set_seed(self, seed: int | None) -> None:
        # TODO(review): seed the underlying vector env RNGs.
        self._gym.reset(seed=seed) if seed is not None else None


def _make_gym_env(
    env_id: str,
    num_envs: int,
    seed: int | None,
    device: DEVICE_TYPING,
    partner_obs: str = "full",
    render_backend: str = "none",
    control_mode: str | None = None,
    partial_reset: bool = True,
    reconfiguration_freq: int | None = None,
) -> Any:
    """Construct the (side-effect-registered) ManiSkill vector gym env.

    Mirrors ``callosum.training.ippo._make_env`` (same ``obs_mode`` /
    ``sim_backend`` / ``control_mode`` / ``reconfiguration_freq`` plumbing) and,
    since review Б4, ALSO threads the env-level ``partner_obs`` toggle into
    ``gym.make`` -- previously the TaskConfig read it but never passed it, so
    the env always fell back to its ``"full"`` default and the toggle was
    dead. ``partner_obs`` controls what the env EMITs into the ``extra`` block;
    the centralized critic consumes it (CTDE), the decentralized actor drops it
    via :meth:`_CallosumManiSkillEnv._agent_obs`.

    ``ignore_terminations`` is set to ``not partial_reset`` to match ippo.py's
    contract (``partial_reset=True`` -> the env auto-resets on termination ->
    ``ignore_terminations=False``). Review С3: this was hardcoded ``True``,
    which both lied about parity with ippo and disabled auto-reset differently
    than the trainer. Defaults mirror ippo's defaults.
    """
    import gymnasium as gym  # server-only
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    # Side-effect import registers FaceTurn-v0 / TwoSO100-v0 gym ids.
    # (Imported lazily because mani_skill is absent on macOS/CI.)
    import callosum.envs.face_turn
    import callosum.envs.two_so100_base  # noqa: F401

    env_kwargs: dict[str, Any] = {
        "obs_mode": "state_dict",
        "sim_backend": "physx_cuda",
        "partner_obs": partner_obs,
        "render_backend": render_backend,
    }
    if control_mode is not None:
        env_kwargs["control_mode"] = control_mode
    base = gym.make(
        env_id,
        num_envs=num_envs,
        reconfiguration_freq=reconfiguration_freq,
        **env_kwargs,
    )
    vec = ManiSkillVectorEnv(
        base,
        num_envs,
        ignore_terminations=not partial_reset,
        record_metrics=True,
    )
    return vec


def _agent_uids_from_env(gym_env: Any) -> list[str]:
    """``('so100-0', 'so100-1')`` -- the two SO-100 arms, in group_map order.

    Uses the same accessor as ``callosum.training.ippo``::

        envs.unwrapped.agent.agents_dict.keys()
    """
    return list(gym_env.unwrapped.agent.agents_dict.keys())


def _max_steps_from_env(gym_env: Any) -> int:
    from mani_skill.utils import gym_utils

    return gym_utils.find_max_episode_steps_value(gym_env._env) or _DEFAULT_MAX_STEPS


# ---------------------------------------------------------------------------
# The BenchMARL TaskClass (the actual deliverable: concrete overrides).
# ---------------------------------------------------------------------------
class CallosumTaskClass(TaskClass):
    """BenchMARL ``TaskClass`` for the FaceTurn-v0 / TwoSO100-v0 ManiSkill envs.

    Implements all 12 ``TaskClass`` abstract methods. Specs are built from
    ``single_observation_space`` / ``single_action_space`` (gym Dict spaces
    keyed by agent uid), per the task contract -- see the ``_grouped_*_spec``
    helpers. The gym id this task targets lives in ``self.config["task"]``
    (set via the hydra yaml's ``task:`` field).
    """

    @property
    def env_id(self) -> str:
        return _ENV_IDS[self.name.lower()]

    # -- core: build the torchrl env (server-only) ---------------------------
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: int | None,
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        # IPPO/MAPPO are continuous-action (Box) on these envs; the
        # continuous_actions flag is honoured but not branching here.
        assert continuous_actions, "FaceTurn/TwoSO100 are continuous-control tasks"
        env_id = self.env_id
        max_steps = self.config.get("max_steps", _DEFAULT_MAX_STEPS)
        # Б4: thread the env-level partner_obs toggle into gym.make (previously
        # read by the config but silently dropped -> env always "full").
        # "full" keeps partner info available for the centralized critic
        # (CTDE); the decentralized actor drops it via _agent_obs.
        partner_obs = self.config.get("partner_obs", "full")
        control_mode = self.config.get("control_mode")  # None -> env default
        partial_reset = self.config.get("partial_reset", True)  # matches ippo default

        def _build() -> EnvBase:
            gym_env = _make_gym_env(
                env_id,
                num_envs,
                seed,
                device,
                partner_obs=partner_obs,
                control_mode=control_mode,
                partial_reset=partial_reset,
            )
            uids = _agent_uids_from_env(gym_env)
            assert len(uids) == 2, f"expected 2 agents, got {uids}"
            ms = _max_steps_from_env(gym_env) or max_steps
            return _CallosumManiSkillEnv(gym_env, num_envs, uids, max_steps=ms)

        return _build

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        # State-based training only; no pixel rendering on the headless server.
        return False

    def max_steps(self, env: EnvBase) -> int:
        return int(self.config.get("max_steps", _DEFAULT_MAX_STEPS))

    def group_map(self, env: EnvBase) -> dict[str, list[str]]:
        # env is _CallosumManiSkillEnv (or a TransformedEnv wrapping it);
        # single_observation_space exposes the agent uids in order.
        return (
            env.group_map
            if hasattr(env, "group_map")
            else {GROUP_NAME: list(env.single_observation_space["agent"].keys())}
        )

    def _obs_spec(self, env: EnvBase) -> Composite:
        uids = self._agent_uids(env)
        return _grouped_observation_spec(env.single_observation_space, uids)

    def observation_spec(self, env: EnvBase) -> Composite:
        # Strip BenchMARL-foreign keys (info/action_mask) like PettingZooClass.
        spec = self._obs_spec(env).clone()
        return spec

    def info_spec(self, env: EnvBase) -> Composite | None:
        return None

    def state_spec(self, env: EnvBase) -> Composite | None:
        # Non-None -> MAPPO gets a centralized state critic over ``"state"``;
        # IPPO ignores it. To force IPPO-style only, return None.
        return _state_spec(env.single_observation_space, self._agent_uids(env))

    def action_spec(self, env: EnvBase) -> Composite:
        uids = self._agent_uids(env)
        return _grouped_action_spec(env.single_action_space, uids)

    def action_mask_spec(self, env: EnvBase) -> Composite | None:
        return None

    @staticmethod
    def env_name() -> str:
        return "callosum"

    def _agent_uids(self, env: EnvBase) -> list[str]:
        if hasattr(env, "agent_uids"):
            return env.agent_uids
        return list(env.single_observation_space["agent"].keys())

    def log_info(self, batch: TensorDictBase) -> dict[str, float]:
        return {}


# ---------------------------------------------------------------------------
# The Task enum (one entry per registered gym id).
# ---------------------------------------------------------------------------
class CallosumTask(Task):
    """BenchMARL task enum for the two callosum envs.

    Members are ``TASK = None`` (loaded automatically from
    ``benchmarl/conf/task/callosum/<task>.yaml`` when run via hydra, or from an
    explicit config dict when constructed programmatically)."""

    FACE_TURN = None
    TWO_SO100 = None

    @staticmethod
    def associated_class():
        return CallosumTaskClass


# ---------------------------------------------------------------------------
# Side-effect registration into BenchMARL's registries.
# ---------------------------------------------------------------------------
def register() -> None:
    """Append ``CallosumTask`` to BenchMARL's ``tasks`` list and rebuild the
    ``"callosum/<task>"`` -> enum + config-schema lookups so that
    ``task=callosum/face_turn`` / ``task=callosum/two_so100`` resolve.

    Importing ``callosum.envs.benchmarl_task`` SIDE-EFFECT-CALLS this once
    (idempotent: it only touches ``benchmarl.environments`` module state, never
    mani_skill/SAPIEN, so it is safe on macOS/CI). The gym-id side-effect
    import (``callosum.envs.face_turn`` / ``two_so100_base``) stays deferred to
    ``get_env_fun`` because mani_skill is Linux+CUDA only -- see
    :func:`_make_gym_env`.

    NOTE: the matching hydra config group ``benchmarl/conf/task/callosum/``
    (face_turn.yaml / two_so100.yaml with ``defaults`` + a ``task:`` field)
    is NOT shipped in this repo and must be placed on the hydra search path
    for the ``benchmarl.train`` hydra-CLI path (review С4 - deferred to
    Этап 3; the programmatic path below is the validated one). Without hydra,
    build the TaskClass programmatically (the runbook §3.4 path)::

        from benchmarl.experiment import Experiment  # server
        task = CallosumTask.FACE_TURN.get_task({"task": "FaceTurn-v0", "max_steps": 100})
        Experiment(task=task, algorithm_config=..., ...)
    """
    from benchmarl.environments import _task_class_registry, task_config_registry, tasks

    if CallosumTask in tasks:
        return
    tasks.append(CallosumTask)
    for task in CallosumTask:
        full = f"{CallosumTask.env_name()}/{task.name.lower()}"
        task_config_registry[full] = task
        schema = _TASK_CONFIG_FOR_TASK.get(task.name.lower())
        if schema is not None:
            _task_class_registry[f"{CallosumTask.env_name()}_{task.name.lower()}"] = schema


# Side-effect: register into BenchMARL's task registry on import (idempotent,
# benchmarl-only). The mani_skill gym-id registration stays deferred (see above).
register()


# TODO(review): place the hydra config group so `task=callosum/face_turn`
# resolves the yaml defaults. Either (a) add ``benchmarl/conf/task/callosum/``
# to BenchMARL's installed package, or (b) register the search path, e.g. in
# callosum's own ``conf/`` tree:
#
#   # conf/task/callosum/face_turn.yaml
#   defaults:
#     - callosum_face_turn_config   # -> FaceTurnTaskConfig dataclass
#     - _self_
#   task: "FaceTurn-v0"
#   max_steps: 100
#   partner_obs: "full"
#
#   # conf/task/callosum/two_so100.yaml
#   defaults:
#     - callosum_two_so100_config
#     - _self_
#   task: "TwoSO100-v0"
#   max_steps: 100
#   partner_obs: "full"
