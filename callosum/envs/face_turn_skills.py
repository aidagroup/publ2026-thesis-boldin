"""FaceTurnSkills-v0: the skill-level cube-assembly environment (openspec tasks 2.1-2.3).

Two SO-ARM101 hands with FIXED roles (agent_a = holder over the multi-face cube's core,
agent_b = rotator of the currently-up layer) execute a discrete skill action space on
top of FaceTurn's physics plumbing (`TwoSO101Base` + the layer lock generalized from
`FaceTurn`): one `env.step` is one DECISION -- both policies pick one primitive id each
(`Dict {"so101_pg-0": Discrete(9), "so101_pg-1": Discrete(7)}`), the choice is executed
for `decision_period` simulation substeps by the phase machinery of
`callosum.experts.skills_runtime` (planning from the CURRENT state; a changed choice
interrupts), and the observation/info contract of `callosum.envs._skills_obs` comes
back: one flat shared vector `[holder_own_block | rotator_own_block]` plus per-agent
proprioception under the agent keys.

The plan is an environment INPUT (`plan_length` random moves per episode, per env --
E1 = 1; Kociemba plans arrive with E3/E4 through the same `AssemblyPlan` shape). A move
on a non-up face needs the holder's flip first (`_plans.flips_to_bring_up`); success =
every move reached its layer-angle target within tolerance AND the cube stayed put.

Episode randomization and failure injection (task 2.3) are the private-randomness
sources of design decision 9: per-episode/per-hand primitive speeds and start delays
(`EpisodeRandomization`), grasp attempts that may be doomed to close on nothing, and
layer turns that may slip mid-motion (both kinematic injections, see
`skills_runtime.InjectionSource`).

The env-level info keys are EXACTLY the `_skills_obs` set (success,
time_to_success_steps, useless_action_holder/rotator, holder/rotator_primitive,
active_layer_angle_deg, plan_index); diagnostics are deliberately not added. The team
reward per decision is `success_bonus * success - step_penalty` (`SkillsRewardConfig`)
-- no dense motor terms; the discount counts decisions in the trainer.

[gpu]: the heavy path (SAPIEN articulation, GPU batched buffers) only runs on Linux;
this module imports mani_skill at module level and is therefore not importable on
macOS, exactly like `callosum.envs.face_turn` (see `scripts/smoke_face_turn_skills.py`
for the [cloud] smoke check and `tests/test_face_turn_skills_logic.py` for the pure
logic).
"""

import dataclasses
import math
import random

import gymnasium.spaces
import numpy as np
import torch
from mani_skill.utils import common
from mani_skill.utils.registration import register_env

from callosum.configs.face_turn_skills import FaceTurnSkillsConfig, SkillsRewardConfig
from callosum.configs.multi_face_cube import MultiFaceCubeConfig
from callosum.envs._face_lock import update_face_lock
from callosum.envs._multi_face_cube import build_multi_face_cube
from callosum.envs._plans import FACES
from callosum.envs._skills import (
    Role,
    num_primitives,
    sample_episode_randomization,
)
from callosum.envs._skills_obs import (
    INFO_ACTIVE_LAYER_ANGLE_DEG,
    INFO_PLAN_INDEX,
    INFO_PRIMITIVE_HOLDER,
    INFO_PRIMITIVE_ROTATOR,
    INFO_SUCCESS,
    INFO_TIME_TO_SUCCESS,
    INFO_USELESS_HOLDER,
    INFO_USELESS_ROTATOR,
)
from callosum.envs.two_so101_base import NUM_ARM_JOINTS, TwoSO101Base
from callosum.experts.skills_runtime import (
    STEP_SECONDS,
    DecisionFacts,
    InjectionSource,
    RuntimeView,
    SkillsRuntime,
    orientation_from_quat,
    own_block_key_layout,
    turn_target_rad,
    up_face_from_quat,
    useless_actions,
)
from callosum.robots.so101_parallel_gripper import ARM_DELTA_LIMIT, GRIPPER_MAX_OPENING

AGENT_UIDS = ("so101_pg-0", "so101_pg-1")
"""Dict action-space keys: agent_a (holder) first -- the trainer's and eval harness's
convention (`callosum.training._skills_agent_obs.ROLES`)."""

_NEAR_TOL = 0.05  # m: a TCP within this of its grasp target counts as "near" (useless facts)


class _EnvInjection(InjectionSource):
    """The env's failure oracle: one python RNG per episode, per-env probabilities."""

    def __init__(self, rng: random.Random, fail_prob: np.ndarray, slip_prob: np.ndarray) -> None:
        self._rng = rng
        self._fail_prob = fail_prob  # (2, n)
        self._slip_prob = slip_prob  # (2, n)

    def fail_grasp(self, arm: int, env: int) -> bool:
        return self._rng.random() < float(self._fail_prob[arm, env])

    def slip_at(self, arm: int, env: int, est_steps: int, now: int) -> int | None:
        if self._rng.random() < float(self._slip_prob[arm, env]):
            frac = 0.3 + 0.5 * self._rng.random()  # slip somewhere in 30-80% of the turn
            return now + max(int(frac * est_steps), 1)
        return None


@register_env(
    "FaceTurnSkills-v0",
    # The BASE env counts sim SUBSTEPS (our step() runs `decision_period` of them per
    # decision), so the base-level limit must be expressed in substeps: 70 decisions
    # x 10 substeps. Decision-level truncation is emitted by the env itself
    # (`max_episode_decisions`), and a `gym.make(max_episode_steps=N)` override wraps
    # at the decision level (the wrapper counts our steps, i.e. decisions).
    max_episode_steps=FaceTurnSkillsConfig().max_episode_decisions
    * FaceTurnSkillsConfig().decision_period,
)
class FaceTurnSkills(TwoSO101Base):
    """The skill-level cube assembly; see the module doc and `callosum.envs._skills_obs`.

    One step = one decision (`decision_period` substeps). The observation is the shared
    skills vector (both own blocks, `_get_obs_extra` emits exactly `own_block_key_layout`'s
    keys in order -- nothing else, so the non-agent fields of the flattened state
    observation are one contiguous run of `shared_vec_size()`). Termination is plan
    success; truncation counts decisions (the gym wrapper's `max_episode_steps` -- 70 by
    default -- is the same unit, one decision per step).

    Layer lock (generalized FaceTurn face lock): while the holder does not grasp the
    core, every layer joint is kinematically clamped at the angle latched when the lock
    engaged, after every physics substep (same mechanism and same GPU-buffer caveats as
    `callosum.envs.face_turn.FaceTurn`; per joint column now).
    """

    def __init__(
        self,
        *args,
        partner_mode: str = "none",
        plan_length: int | None = None,
        grasp_failure_prob_range: tuple[float, float] | None = None,
        slip_prob_range: tuple[float, float] | None = None,
        skills_config: FaceTurnSkillsConfig | None = None,
        cube_config: MultiFaceCubeConfig | None = None,
        **kwargs,
    ):
        config = skills_config if skills_config is not None else FaceTurnSkillsConfig()
        if plan_length is not None:
            config = dataclasses.replace(config, plan_length=int(plan_length))
        if grasp_failure_prob_range is not None:
            lo, hi = grasp_failure_prob_range
            config = dataclasses.replace(
                config,
                randomization=dataclasses.replace(
                    config.randomization, grasp_failure_prob_range=(float(lo), float(hi))
                ),
            )
        if slip_prob_range is not None:
            # dataclasses.replace re-runs __post_init__, so the range is validated there.
            config = dataclasses.replace(
                config, slip_prob_range=(float(slip_prob_range[0]), float(slip_prob_range[1]))
            )
        # Validates partner_mode (and everything else) at construction: fail loudly.
        config = dataclasses.replace(config, partner_mode=partner_mode)
        self.skills_config = config
        self.cube_config = cube_config if cube_config is not None else MultiFaceCubeConfig()
        self._advertise_joint_actions = False
        self._skill_action_space = gymnasium.spaces.Dict(
            {
                uid: gymnasium.spaces.Discrete(num_primitives(role))
                for uid, role in zip(AGENT_UIDS, (Role.HOLDER, Role.ROTATOR), strict=True)
            }
        )
        super().__init__(*args, **kwargs, partner_obs="full")

    # --- action space: the menu the policies see; the joints underneath --------------------

    @property
    def action_space(self):
        """The advertised Dict-of-Discrete skill space; the parent's joint Box space
        while the substep machinery runs (it receives the runtime's delta actions)."""
        if self._advertise_joint_actions:
            return self._base_action_space
        return self._skill_action_space

    @action_space.setter
    def action_space(self, value):
        # The base constructor ASSIGNS the agent's joint Box space here before our
        # machinery exists; keep it for the substep fallback (`_base_action_space`) --
        # the advertised space is always derived, never this stored value directly.
        self._base_action_space = value

    # --- scene ----------------------------------------------------------------------------

    def _load_scene(self, options: dict):
        # Like FaceTurn: re-implement the parent's table setup instead of calling
        # super() -- TwoSO101Base._load_scene also builds the plain loose cube, which
        # this task replaces with the multi-face articulation.
        from mani_skill.utils.scene_builder.table import TableSceneBuilder

        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cube = build_multi_face_cube(
            self.scene, name="multi_face_cube", config=self.cube_config
        )
        self.core_link = self.cube.links_map["core"]
        # qpos column of each layer follows the config's layer order (FACE_NORMALS:
        # U, D, L, R, F, B); the dict below is the single source of that truth.
        self.layer_faces = tuple(spec.face for spec in self.cube_config.layers)
        self.layer_col = {face: i for i, face in enumerate(self.layer_faces)}
        self.layer_links = {
            face: self.cube.links_map[f"layer_{face}"] for face in self.layer_faces
        }

        self.body_init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # Layer-lock state, per env and per joint (see FaceTurn.face_locked).
        self._layer_locked = torch.ones((self.num_envs, 6), dtype=torch.bool, device=self.device)
        self._layer_lock_angle = torch.zeros((self.num_envs, 6), device=self.device)
        self._any_layer_locked = False

        # Per-env episode state (reset in _initialize_episode).
        n = self.num_envs
        plan_cols = max(self.skills_config.plan_length, 1)
        self._plan_faces = torch.zeros((n, plan_cols), dtype=torch.long, device=self.device)
        self._plan_quarters = torch.zeros_like(self._plan_faces)
        self._plan_index = torch.zeros(n, dtype=torch.long, device=self.device)
        self._move_target = torch.zeros(n, device=self.device)
        self._decision_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self._episode_substeps = torch.zeros(n, dtype=torch.long, device=self.device)
        self._success_substep = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self._needed_flip: list = [None] * n
        self._last_selections = np.zeros((2, n), dtype=int)  # refreshed after substeps
        self._last_progresses = np.zeros((2, n))
        self._substeps = 0  # global substep clock (start delays, slip scheduling)
        self._latched_success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._latched_time = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self._inject: InjectionSource | None = None
        self._runtime: SkillsRuntime | None = None

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Places both arms and the cube root (upright, jittered) -- the parent's reset.
        super()._initialize_episode(env_idx, options)
        cfg = self.skills_config
        b = len(env_idx)
        n = self.num_envs
        with torch.device(self.device):
            self.cube.set_qpos(torch.zeros((b, 6)))
            self.cube.set_qvel(torch.zeros((b, 6)))
            # The holder is not grasping at reset: all layers locked at angle 0.
            self._layer_locked[env_idx] = True
            self._layer_lock_angle[env_idx] = 0.0
            self.body_init_pos[env_idx] = self.cube.pose.p[env_idx]
            # Fresh random plan per env (torch RNG, the FaceTurn jitter precedent; the
            # trainer seeds torch globally).
            length = cfg.plan_length
            if length > 0:
                self._plan_faces[env_idx] = torch.randint(0, len(FACES), (b, length))
                self._plan_quarters[env_idx] = torch.randint(1, 3, (b, length))
            self._plan_index[env_idx] = 0
            self._decision_count[env_idx] = 0
            self._episode_substeps[env_idx] = 0
            self._success_substep[env_idx] = -1

        # Episode-private randomness (design decision 9): one stdlib RNG seeded from
        # the global torch generator, one `EpisodeRandomization` sample per env through
        # the contract sampler (fixed draw order, deterministic per seed).
        base_seed = int(torch.randint(0, 2**31 - 1, (1,)))
        speeds = np.ones((2, b))
        delays = np.zeros((2, b))
        fail_prob = np.zeros((2, b))
        slip_prob = np.zeros((2, b))
        rng = random.Random(base_seed)
        lo, hi = cfg.slip_prob_range
        for i in range(b):
            episode = sample_episode_randomization(cfg.randomization, base_seed + 1 + i)
            speeds[0, i] = episode.arm_multiplier(0)
            speeds[1, i] = episode.arm_multiplier(1)
            delays[0, i], delays[1, i] = episode.start_delay
            fail_prob[0, i], fail_prob[1, i] = episode.grasp_failure_prob
            slip_prob[0, i] = rng.uniform(lo, hi)
            slip_prob[1, i] = rng.uniform(lo, hi)
        self._inject = _EnvInjection(rng, fail_prob, slip_prob)
        if self._runtime is None:
            self._runtime = SkillsRuntime(n, inject=self._inject, delta_limit=ARM_DELTA_LIMIT)
        else:
            self._runtime.inject = self._inject
        self._runtime.reset(
            self._view(), speeds, np.round(delays / STEP_SECONDS).astype(int), env_idx.cpu().numpy()
        )
        self._needed_flip = self._compute_needed_flip()
        self._last_selections = self._runtime.selections()
        # A partial reset zeroes the progress bookkeeping of the reset envs only; the
        # surviving envs keep their live primitive progress for the next observation.
        self._last_progresses[:, env_idx.cpu().numpy()] = 0.0
        self._refresh_move_targets(env_idx.cpu().numpy())

    # --- layer lock (generalized face lock) -------------------------------------------------

    def _holder_holds(self) -> torch.Tensor:
        """The holder clamps the cube BODY: the core is fully enclosed by the six
        layer slabs, so the jaws can only ever touch a slab -- and every layer is
        rigidly locked to the core while it is not held (the lock's whole point), so
        grasping ANY link of the articulation IS grasping the body."""
        held = self.agent_a.is_grasping(self.core_link)
        for link in self.cube.links:
            held = held | self.agent_a.is_grasping(link)
        return held

    def _before_control_step(self):
        if not self.skills_config.lock_layers_unless_held:
            return
        held = self._holder_holds()
        want = ~held  # every layer locks while the core is not held
        qpos = self.cube.get_qpos()
        for col in range(6):
            self._layer_locked[:, col], self._layer_lock_angle[:, col] = update_face_lock(
                self._layer_locked[:, col],
                self._layer_lock_angle[:, col],
                want,
                qpos[:, col],
            )
        self._any_layer_locked = bool(self._layer_locked.any())

    def _after_simulation_step(self):
        # Clamp the locked envs' layer joints after every physics substep (the FaceTurn
        # mechanism; per joint column now, GPU fetch/apply as there).
        if not (self.skills_config.lock_layers_unless_held and self._any_layer_locked):
            return
        if self.gpu_sim_enabled:
            # TODO(review): GPU path unverified (no GPU on the dev machine), same as the
            # FaceTurn face lock it generalizes.
            px = self.scene.px
            px.gpu_fetch_articulation_qpos()
            px.gpu_fetch_articulation_qvel()
        locked = self._layer_locked
        qpos = torch.where(locked, self._layer_lock_angle, self.cube.get_qpos())
        qvel = torch.where(locked, torch.zeros_like(qpos), self.cube.get_qvel())
        self.cube.set_qpos(qpos)
        self.cube.set_qvel(qvel)
        if self.gpu_sim_enabled:
            px.gpu_apply_articulation_qpos()
            px.gpu_apply_articulation_qvel()
            px.gpu_update_articulation_kinematics()

    # --- plans ----------------------------------------------------------------------------

    def _face_tensor(self) -> torch.Tensor:
        """`(n,)` FACES index of each env's current move (0 when the plan is done)."""
        idx = self._plan_index.clamp(max=self._plan_faces.shape[1] - 1)
        return torch.gather(self._plan_faces, 1, idx.unsqueeze(1)).squeeze(1)

    def _quarter_tensor(self) -> torch.Tensor:
        idx = self._plan_index.clamp(max=self._plan_quarters.shape[1] - 1)
        return torch.gather(self._plan_quarters, 1, idx.unsqueeze(1)).squeeze(1)

    def _active_layer_angle(self) -> torch.Tensor:
        """`(n,)` rad: the current plan move's layer angle (0 when the plan is done)."""
        if self.skills_config.plan_length == 0:
            return torch.zeros(self.num_envs, device=self.device)
        face = self._face_tensor()
        cols = torch.as_tensor(
            [self.layer_col[f] for f in FACES], device=self.device
        )
        rows = torch.arange(self.num_envs, device=self.device)
        return self.cube.get_qpos()[rows, cols[face]]

    def _active_layer_link(self, faces_idx: int):
        return self.layer_links[FACES[int(faces_idx)]]

    def _refresh_move_targets(self, envs) -> None:
        """(Re)compute the authoritative move target of every env in `envs`."""
        if self.skills_config.plan_length == 0:
            return
        quarters_np = self._plan_quarters.cpu().numpy()
        angle = self._active_layer_angle().cpu().numpy()
        index = self._plan_index.cpu().numpy()
        target = self._move_target.cpu().numpy()
        for env in envs:
            i = int(env)
            if index[i] >= self.skills_config.plan_length:
                continue
            target[i] = turn_target_rad(float(angle[i]), int(quarters_np[i, index[i]]))
        self._move_target.copy_(torch.as_tensor(target, device=self.device))

    def _advance_plan(self) -> None:
        """Mark plan moves done when their layer reached the target within tolerance."""
        if self.skills_config.plan_length == 0:
            return
        angle = self._active_layer_angle()
        within = (self._plan_index < self.skills_config.plan_length) & (
            (angle - self._move_target).abs() < self.skills_config.angle_tol_rad
        )
        if not within.any():
            return
        self._plan_index[within] += 1
        self._refresh_move_targets(torch.flatnonzero(within).cpu().numpy())

    def _compute_needed_flip(self) -> list:
        """Per env: the single `Flip` that brings the current move's face up (or None)."""
        needed: list = [None] * self.num_envs
        if self.skills_config.plan_length == 0:
            return needed
        quat = self.cube.pose.q.cpu().numpy()
        faces = self._plan_faces.cpu().numpy()
        index = self._plan_index.cpu().numpy()
        for env in range(self.num_envs):
            i = int(index[env])
            if i >= self.skills_config.plan_length:
                continue
            orientation = orientation_from_quat(quat[env])
            flips, _ = orientation.flips_to_bring_up(FACES[int(faces[env, i])])
            needed[env] = flips[0] if flips else None
        return needed

    # --- the runtime's view of the state -----------------------------------------------------

    def _view(self) -> RuntimeView:
        qpos = [self._arm_qpos(arm) for arm in range(2)]
        qvel = [self._arm_qvel(arm) for arm in range(2)]
        bases = [
            self._agent(arm).robot.pose.to_transformation_matrix().cpu().numpy()
            for arm in range(2)
        ]
        angle = self._active_layer_angle()
        target = self._move_target
        layer_at_target = (
            (angle - target).abs() < self.skills_config.angle_tol_rad
        ).cpu().numpy()
        return RuntimeView(
            qpos=[q[:, :NUM_ARM_JOINTS].cpu().numpy() for q in qpos],
            arm_speed=[q[:, :NUM_ARM_JOINTS].abs().max(dim=1).values.cpu().numpy() for q in qvel],
            jaw_speed=[q[:, NUM_ARM_JOINTS:].abs().max(dim=1).values.cpu().numpy() for q in qvel],
            base_pos=[b[:, :3, 3] for b in bases],
            base_rot=[b[:, :3, :3] for b in bases],
            cube_pos=self.cube.pose.p.cpu().numpy(),
            cube_quat=self.cube.pose.q.cpu().numpy(),
            layer_angle=angle.cpu().numpy(),
            turn_target=target.cpu().numpy(),
            layer_at_target=layer_at_target,
            needed_flip=self._needed_flip,
            t=self._substeps,
        )

    def _agent(self, arm: int):
        return self.agent_a if arm == 0 else self.agent_b

    def _arm_qpos(self, arm: int) -> torch.Tensor:
        return self._agent(arm).robot.get_qpos()

    def _arm_qvel(self, arm: int) -> torch.Tensor:
        return self._agent(arm).robot.get_qvel()

    # --- stepping --------------------------------------------------------------------------

    def _decode_actions(self, action) -> list[torch.Tensor]:
        """`[holder_ids, rotator_ids]`, each `(n,)` long on the sim device.

        Accepts the dict form of the trainer/eval harness (`{uid: (n,) tensor}`) and
        the scalar form of an unbatched `action_space.sample()` (one id per agent,
        broadcast to every env).
        """
        if isinstance(action, dict):
            try:
                items = [action[uid] for uid in AGENT_UIDS]
            except KeyError as exc:
                raise ValueError(f"action dict keys {tuple(action)} do not match {AGENT_UIDS}") \
                    from exc
        else:
            items = list(action)
            if len(items) != 2:
                raise ValueError(f"expected a two-agent action, got {action!r}")
        ids = []
        for arm, value in enumerate(items):
            tensor = common.to_tensor(value).to(device=self.device)
            tensor = tensor.reshape(-1).long()
            if tensor.numel() == 1:
                tensor = tensor.expand(self.num_envs).clone()
            limit = num_primitives(Role.HOLDER if arm == 0 else Role.ROTATOR)
            if bool((tensor < 0).any() or (tensor >= limit).any()):
                raise ValueError(
                    f"agent {AGENT_UIDS[arm]} action ids must be in [0, {limit}), "
                    f"got {tensor.tolist()}"
                )
            ids.append(tensor)
        return ids

    def _decision_facts(self) -> DecisionFacts:
        holder_grasp = self._holder_holds().cpu().numpy()
        rotator_grasp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        rotator_near = torch.zeros_like(rotator_grasp)
        holder_near = (
            torch.linalg.norm(self.agent_a.tcp_pos - self.cube.pose.p, dim=1) < _NEAR_TOL
        )
        if self.skills_config.plan_length > 0:
            face = self._face_tensor()
            for face_idx in torch.unique(face):
                mask = face == face_idx
                link = self._active_layer_link(int(face_idx))
                rotator_grasp |= mask & self.agent_b.is_grasping(link)
                rotator_near |= mask & (
                    torch.linalg.norm(self.agent_b.tcp_pos - link.pose.p, dim=1) < _NEAR_TOL
                )
        up_is_move = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.skills_config.plan_length > 0:
            quat = self.cube.pose.q.cpu().numpy()
            face_np = self._face_tensor().cpu().numpy()
            index = self._plan_index.cpu().numpy()
            for env in range(self.num_envs):
                if index[env] < self.skills_config.plan_length:
                    up_is_move[env] = up_face_from_quat(quat[env]) == FACES[int(face_np[env])]
        return DecisionFacts(
            holder_grasp=holder_grasp,
            rotator_grasp=rotator_grasp.cpu().numpy(),
            holder_near_core=holder_near.cpu().numpy(),
            rotator_near_layer=rotator_near.cpu().numpy(),
            up_face_is_move_face=up_is_move.cpu().numpy(),
        )

    def step(self, action):
        """One decision: decode both choices, execute them for `decision_period` substeps."""
        ids = self._decode_actions(action)
        facts = self._decision_facts()
        useless_h = useless_actions(Role.HOLDER, ids[0].cpu().numpy(), facts)
        useless_r = useless_actions(Role.ROTATOR, ids[1].cpu().numpy(), facts)
        self._needed_flip = self._compute_needed_flip()
        view = self._view()
        self._runtime.set_decision(0, ids[0].cpu().numpy(), view)
        self._runtime.set_decision(1, ids[1].cpu().numpy(), view)
        self._decision_count += 1

        obs = None
        for _ in range(self.skills_config.decision_period):
            joint = self._runtime.actions(self._view())
            self._advertise_joint_actions = True
            try:
                # The parent's full control step (hooks, GPU buffers, obs/reward/info);
                # the per-substep rewards are discarded -- the decision-level team
                # reward is computed once below from the final state.
                obs, _sub_reward, _sub_term, _sub_trunc, _sub_info = super().step(
                    {
                        uid: torch.as_tensor(
                            joint[arm], dtype=torch.float32, device=self.device
                        )
                        for arm, uid in enumerate(AGENT_UIDS)
                    }
                )
            finally:
                self._advertise_joint_actions = False
            self._substeps += 1
            self._episode_substeps += 1
            # Latch this decision's success (and its sim-step count) immediately: if
            # the parent's machinery auto-resets a done env within the loop, the
            # post-loop evaluate() would otherwise report the fresh episode instead.
            latched = _sub_info[INFO_SUCCESS]
            if bool(latched.any()):
                fresh = latched & (self._latched_time < 0)
                self._latched_success |= latched
                self._latched_time[fresh] = _sub_info[INFO_TIME_TO_SUCCESS][fresh]
            self._runtime.substep(self._view())
            self._advance_plan()
            self._last_selections = self._runtime.selections()
            self._last_progresses = self._runtime.progresses()

        # The decision-level view of the episode (the last substep's info was computed
        # before this loop's final runtime/plan bookkeeping; recompute it fresh).
        info = self.evaluate()
        success = info[INFO_SUCCESS] | self._latched_success
        info[INFO_SUCCESS] = success
        missing_time = self._latched_success & (self._latched_time >= 0)
        info[INFO_TIME_TO_SUCCESS] = torch.where(
            missing_time,
            self._latched_time,
            info[INFO_TIME_TO_SUCCESS],
        )
        self._latched_success[:] = False
        self._latched_time[:] = -1
        info[INFO_USELESS_HOLDER] = torch.as_tensor(useless_h, device=self.device)
        info[INFO_USELESS_ROTATOR] = torch.as_tensor(useless_r, device=self.device)
        reward = (
            self.skills_config.reward.success_bonus * success.float()
            - self.skills_config.reward.step_penalty
        )
        terminated = success.clone()
        truncated = self._decision_count >= self.skills_config.max_episode_decisions
        return obs, reward, terminated, truncated, info

    # --- observations ------------------------------------------------------------------------

    def _get_obs_extra(self, info: dict):
        # ONLY the shared skills vector (holder block, then rotator's, keys exactly in
        # `own_block_key_layout` order): the trainer's SkillsObsBuilder requires every
        # non-agent field to form one contiguous run of `shared_vec_size()`, so the
        # parent's TCP/cube fields are deliberately NOT inherited here.
        cfg = self.skills_config
        n = self.num_envs
        selections = self._last_selections
        progresses = self._last_progresses
        rows = torch.arange(n, device=self.device)
        onehot_face = torch.zeros((n, len(FACES)), device=self.device)
        quarter = torch.zeros(n, device=self.device)
        remaining = torch.zeros(n, device=self.device)
        angle = self._active_layer_angle()
        if cfg.plan_length > 0:
            pending = (self._plan_index < cfg.plan_length).float()
            onehot_face[rows, self._face_tensor()] = 1.0
            onehot_face = onehot_face * pending.unsqueeze(1)
            quarter = self._quarter_tensor().float() / 2.0 * pending
            remaining = (cfg.plan_length - self._plan_index).float() / cfg.plan_length
        holder_oh = torch.zeros((n, num_primitives(Role.HOLDER)), device=self.device)
        rotator_oh = torch.zeros((n, num_primitives(Role.ROTATOR)), device=self.device)
        holder_oh[rows, torch.as_tensor(selections[0], device=self.device)] = 1.0
        rotator_oh[rows, torch.as_tensor(selections[1], device=self.device)] = 1.0
        time_fraction = (self._decision_count.float() / cfg.max_episode_decisions).clamp(0.0, 1.0)

        def block(arm: int, onehot: torch.Tensor) -> dict:
            qpos = self._arm_qpos(arm)
            return {
                "time_fraction": time_fraction.unsqueeze(1),
                "plan_face_onehot": onehot_face if arm == 0 else onehot_face.clone(),
                "plan_quarter_turns": (quarter if arm == 0 else quarter.clone()).unsqueeze(1),
                "plan_moves_remaining": (
                    remaining if arm == 0 else remaining.clone()
                ).unsqueeze(1),
                "cube_position": self.cube.pose.p if arm == 0 else self.cube.pose.p.clone(),
                "cube_orientation_quat": (
                    self.cube.pose.q if arm == 0 else self.cube.pose.q.clone()
                ),
                "active_layer_angle": (angle / math.pi if arm == 0 else angle.clone() / math.pi)
                .unsqueeze(1),
                "own_qpos": qpos,
                "own_qvel": self._arm_qvel(arm),
                "own_gripper_opening": (qpos[:, NUM_ARM_JOINTS] / GRIPPER_MAX_OPENING)
                .clamp(0.0, 1.0)
                .unsqueeze(1),
                "own_primitive_onehot": onehot,
                "own_primitive_progress": torch.as_tensor(
                    progresses[arm], dtype=torch.float32, device=self.device
                ).unsqueeze(1),
            }

        values = (block(0, holder_oh), block(1, rotator_oh))
        obs = {}
        for role, fields in zip((Role.HOLDER, Role.ROTATOR), values, strict=True):
            prefix = "holder_" if role is Role.HOLDER else "rotator_"
            for name, _size in own_block_key_layout(role):
                # `fields` is keyed unprefixed; the flat obs carries the prefixed
                # `own_block_key_layout` keys (the shared-vector contract).
                obs[name] = fields[name.removeprefix(prefix)]
        return obs

    # --- reward and success -------------------------------------------------------------------

    def evaluate(self):
        cfg = self.skills_config
        done = self._plan_index >= cfg.plan_length
        pos_drift = torch.linalg.norm(self.cube.pose.p - self.body_init_pos, dim=1)
        layer_speed = self.cube.get_qvel().abs().max(dim=1).values
        settled = layer_speed < cfg.body_settle_qvel
        stable = (pos_drift < cfg.body_pos_tol) & settled
        success = done & stable
        # Latch the sim-step count of the first success (idempotent; evaluate runs per
        # substep AND once more for the decision-level info).
        first = success & (self._success_substep < 0)
        if bool(first.any()):
            self._success_substep[first] = self._episode_substeps[first]
        selections = self._last_selections
        return {
            INFO_SUCCESS: success,
            INFO_TIME_TO_SUCCESS: torch.where(
                success,
                self._success_substep.clamp(min=0),
                torch.zeros_like(self._success_substep),
            ),
            INFO_PRIMITIVE_HOLDER: torch.as_tensor(
                selections[0], dtype=torch.long, device=self.device
            ),
            INFO_PRIMITIVE_ROTATOR: torch.as_tensor(
                selections[1], dtype=torch.long, device=self.device
            ),
            INFO_ACTIVE_LAYER_ANGLE_DEG: torch.rad2deg(self._active_layer_angle()),
            INFO_PLAN_INDEX: self._plan_index,
        }

    def compute_dense_reward(self, obs, action: torch.Tensor, info: dict):
        cfg: SkillsRewardConfig = self.skills_config.reward
        return cfg.success_bonus * info[INFO_SUCCESS].float() - cfg.step_penalty

    def compute_normalized_dense_reward(self, obs, action: torch.Tensor, info: dict):
        # The skill reward is already the per-decision team quantity in [-0.7, 1]: the
        # normalised and dense forms coincide by construction.
        return self.compute_dense_reward(obs=obs, action=action, info=info)
