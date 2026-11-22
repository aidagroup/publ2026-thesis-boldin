"""FaceTurnSkills-v0: the skill-level cube-assembly environment (openspec tasks 2.1-2.3).

Two SO-ARM101 hands with FIXED roles (agent_a = holder over the multi-face cube's core,
agent_b = rotator of the currently-up layer) execute a discrete skill action space on
top of FaceTurn's physics plumbing (`TwoSO101Base` + the layer lock generalized from
`FaceTurn`): one `env.step` is one DECISION -- both policies pick one primitive id each
(`Dict {"so101_pg-0": Discrete(9), "so101_pg-1": Discrete(7)}`), the choice is executed
for `decision_period` simulation substeps by the phase machinery of
`callosum.experiments.arm.skills_runtime` (planning from the CURRENT state; a changed choice
interrupts), and the observation/info contract of `callosum.experiments.arm.skills_obs` comes
back: one flat shared vector `[holder_own_block | rotator_own_block]` plus per-agent
proprioception under the agent keys.

The plan is an environment INPUT: by default `plan_length` random moves per episode, per env
(E1 = 1; Kociemba plans arrive with E3/E4 through the same `AssemblyPlan` shape); a FIXED
plan in standard notation (`skills_config.plan_notation`, e.g. the pif-paf `"R U R' U'"`)
replaces the random draw for every env and episode, with primes already expanded by
`_plans.parse_face_turn_sequence`. A move on a non-up face needs the holder's flip first
(`_plans.flips_to_bring_up`); success = every move reached its layer-angle target within
tolerance AND the cube stayed put.

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
macOS, exactly like `callosum.experiments.arm.face_turn_env` (see `scripts/experiments/arm/smoke_face_turn_skills.py`
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

from callosum.experiments.arm.face_lock import update_face_lock
from callosum.experiments.arm.scene import NUM_ARM_JOINTS, TwoSO101Base
from callosum.experiments.arm.skills import (
    ROTATOR_PRIMITIVES,
    Role,
    num_primitives,
    sample_episode_randomization,
)
from callosum.experiments.arm.skills_config import FaceTurnSkillsConfig, SkillsRewardConfig
from callosum.experiments.arm.skills_cube import build_multi_face_cube
from callosum.experiments.arm.skills_cube_config import MultiFaceCubeConfig
from callosum.experiments.arm.skills_obs import (
    INFO_ACTIVE_LAYER_ANGLE_DEG,
    INFO_PLAN_INDEX,
    INFO_PRIMITIVE_HOLDER,
    INFO_PRIMITIVE_ROTATOR,
    INFO_ROTATOR_GRASP_FACT,
    INFO_SUCCESS,
    INFO_TIME_TO_SUCCESS,
    INFO_USELESS_HOLDER,
    INFO_USELESS_ROTATOR,
)
from callosum.experiments.arm.skills_plans import FACES
from callosum.experiments.arm.skills_runtime import (
    LAYER_DETENTS_RAD,
    STEP_SECONDS,
    DecisionFacts,
    InjectionSource,
    RuntimeView,
    SkillsRuntime,
    orientation_from_quat,
    own_block_key_layout,
    sapien_quat_to_xyzw,
    turn_target_rad,
    up_face_from_quat,
    useless_actions,
)
from callosum.robots.so101_parallel_gripper import ARM_DELTA_LIMIT, GRIPPER_MAX_OPENING

AGENT_UIDS = ("so101_pg-0", "so101_pg-1")
"""Dict action-space keys: agent_a (holder) first -- the trainer's and eval harness's
convention (`callosum.experiments.arm.skills_agent_obs.ROLES`)."""

_NEAR_TOL = 0.05  # m: a TCP within this of its grasp target counts as "near" (useless facts)

_GRASP_MIN_FORCE = 0.5  # N: per-jaw contact force that counts as a grip (the robot's default)
_GRASP_MAX_ANGLE_DEG = (
    85.0  # deg: max angle between a jaw's opening direction and its contact force
)
"""Grasp thresholds of `SO101ParallelGripper.is_grasping` (Panda's convention), reused by the
env's multi-link grasp check below."""


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
    """The skill-level cube assembly; see the module doc and `callosum.experiments.arm.skills_obs`.

    One step = one decision (`decision_period` substeps). The observation is the shared
    skills vector (both own blocks, `_get_obs_extra` emits exactly `own_block_key_layout`'s
    keys in order -- nothing else, so the non-agent fields of the flattened state
    observation are one contiguous run of `shared_vec_size()`). Termination is plan
    success; truncation counts decisions (the gym wrapper's `max_episode_steps` -- 70 by
    default -- is the same unit, one decision per step).

    Layer lock (generalized FaceTurn face lock): while the holder does not grasp the
    core, every layer joint is kinematically clamped at the angle latched when the lock
    engaged, after every physics substep (same mechanism and same GPU-buffer caveats as
    `callosum.experiments.arm.face_turn_env.FaceTurn`; per joint column now).
    """

    def __init__(
        self,
        *args,
        partner_mode: str = "none",
        plan_length: int | None = None,
        plan_notation: str | None = None,
        grasp_failure_prob_range: tuple[float, float] | None = None,
        slip_prob_range: tuple[float, float] | None = None,
        skills_config: FaceTurnSkillsConfig | None = None,
        cube_config: MultiFaceCubeConfig | None = None,
        **kwargs,
    ):
        config = skills_config if skills_config is not None else FaceTurnSkillsConfig()
        if plan_length is not None:
            config = dataclasses.replace(config, plan_length=int(plan_length))
        if plan_notation is not None:
            # Validated (and plan_length derived from the prime expansion) in __post_init__.
            config = dataclasses.replace(config, plan_notation=str(plan_notation))
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
        self._fixed_plan = config.fixed_plan
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
        self.layer_links = {face: self.cube.links_map[f"layer_{face}"] for face in self.layer_faces}

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
        # The fixed plan (plan_notation) is static: build its column tensors once here and
        # scatter them into every env at reset instead of drawing random moves.
        if self._fixed_plan is not None:
            self._fixed_plan_faces = torch.tensor(
                [FACES.index(move.face) for move in self._fixed_plan],
                dtype=torch.long,
                device=self.device,
            )
            self._fixed_plan_quarters = torch.tensor(
                [move.quarter_turns for move in self._fixed_plan],
                dtype=torch.long,
                device=self.device,
            )
        else:
            self._fixed_plan_faces = None
            self._fixed_plan_quarters = None
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
        # Bias the cube home toward the rotator (see FaceTurnSkillsConfig.cube_home_x):
        # applied right after the parent's centred placement, BEFORE body_init_pos is
        # latched below, so the spawn AND the post-flip drag-home target both carry it.
        if self.skills_config.cube_home_x:
            from mani_skill.utils.structs.pose import Pose

            shifted = self.cube.pose.p[env_idx].clone()
            shifted[:, 0] += self.skills_config.cube_home_x
            self.cube.set_pose(Pose.create_from_pq(p=shifted, q=self.cube.pose.q[env_idx]))
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
            # trainer seeds torch globally) -- or the SAME fixed notation plan for every
            # env when `skills_config.plan_notation` is set (the pif-paf demo).
            length = cfg.plan_length
            if length > 0:
                if self._fixed_plan_faces is not None:
                    self._plan_faces[env_idx] = self._fixed_plan_faces.unsqueeze(0).expand(b, -1)
                    self._plan_quarters[env_idx] = self._fixed_plan_quarters.unsqueeze(0).expand(
                        b, -1
                    )
                else:
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

    def _jaw_press(self, agent, link) -> tuple[torch.Tensor, torch.Tensor]:
        """`(left jaw presses link, right jaw presses link)` in the `is_grasping` convention.

        Same force/angle test as `SO101ParallelGripper.is_grasping`, but evaluated per
        LINK instead of against one object: on the ARTICULATED multi-face cube the two
        jaws squeeze two DIFFERENT layer slabs (e.g. clamp_1 on `layer_L` and clamp_2 on
        `layer_R`), so the robot's pairwise `is_grasping(any_single_link)` can never be
        true -- one jaw always reads zero force on that link. That is why the holder
        looked "not grasping" while physically lifting the cube, every layer stayed
        locked and no face turn could ever execute.
        """
        from mani_skill.utils import common

        finger1, finger2 = agent.finger1_link, agent.finger2_link
        left_forces = self.scene.get_pairwise_contact_forces(finger1, link)
        right_forces = self.scene.get_pairwise_contact_forces(finger2, link)
        left_force = torch.linalg.norm(left_forces, dim=1)
        right_force = torch.linalg.norm(right_forces, dim=1)
        left_dir = finger1.pose.to_transformation_matrix()[..., :3, 0]
        right_dir = -finger2.pose.to_transformation_matrix()[..., :3, 0]
        # Zero-force contact pairs give a NaN angle; the comparison below is then False.
        left_angle = torch.nan_to_num(
            torch.rad2deg(common.compute_angle_between(left_dir, left_forces)), nan=180.0
        )
        right_angle = torch.nan_to_num(
            torch.rad2deg(common.compute_angle_between(right_dir, right_forces)), nan=180.0
        )
        left = (left_force >= _GRASP_MIN_FORCE) & (left_angle <= _GRASP_MAX_ANGLE_DEG)
        right = (right_force >= _GRASP_MIN_FORCE) & (right_angle <= _GRASP_MAX_ANGLE_DEG)
        return left, right

    def _grasp_any(self, agent, links) -> torch.Tensor:
        """Per env: both jaws press SOME link of the (articulated) object.

        For a multi-link object the two jaws may press different links, so the per-link
        `is_grasping` calls are OR-ed per jaw before AND-ing the two jaws.
        """
        left = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        right = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for link in links:
            left_link, right_link = self._jaw_press(agent, link)
            left |= left_link
            right |= right_link
        return left & right

    def _holder_holds(self) -> torch.Tensor:
        """The holder clamps the cube BODY: the core is fully enclosed by the six
        layer slabs, so the jaws can only ever touch a slab -- and every layer is
        rigidly locked to the core while it is not held (the lock's whole point), so
        grasping ANY link of the articulation IS grasping the body."""
        return self._grasp_any(self.agent_a, self.cube.links)

    _TURN_IDS = frozenset(p.index for p in ROTATOR_PRIMITIVES if p.name in ("turn90", "turn180"))
    _ROTATOR_GRASPISH_IDS = frozenset(
        p.index for p in ROTATOR_PRIMITIVES if p.name in ("grasp", "turn90", "turn180")
    )

    def _turn_assist(self) -> None:
        """Kinematic turn assist -- see `FaceTurnSkillsConfig.turn_assist` for the
        rationale (scripted motorics by the thesis's assumption; the coordination
        policy is the subject, the physical rim-pinch torque transfer was measured
        as unreliable: the grip slips and the cube drifts ~9 cm without the face
        ever turning). ACTIVE ONLY ON A GRASP: per env, the rotator's selected
        primitive must be turn90/turn180 AND the active layer link must be grasped
        under the CONFIGURED criterion (`_grasp_any` contact, plus the proximity
        pinch in `turn_grasp_mode="proximity"`, exactly like `rotator_grasp_fact`)
        -- otherwise nothing happens and a turn on an unheld cube fails as before. The layer joint then follows the authoritative move target at the
        arm delta-limit speed, one commanded step per substep, in synchrony with
        the rotator's wrist roll command (the wrist and the layer rotate
        together, so the jaws do not scrape the rim).

        TODO(review): GPU path follows the same fetch/apply pattern as the layer
        lock (unverified on the dev machine, no GPU there).
        """
        if not (self.skills_config.turn_assist and self.skills_config.plan_length):
            return
        selections = self._runtime.selections()[1]
        turning = np.isin(selections, tuple(self._TURN_IDS))
        if not turning.any():
            return
        # A grasp of the ACTIVE layer only, with the SAME criterion as the env's exported
        # `rotator_grasp_fact`: in `turn_grasp_mode="proximity"` (the documented
        # scripted-motorics simplification) a navigated proximity pinch counts exactly as
        # it does for the conductor, otherwise the assist was gated on a physical pinch
        # that the design itself calls unreliable (side layers after a flip cannot be
        # pinched: both arms cramp over the 5.7 cm cube) -- so the fact said "grasping"
        # while the assist refused to turn, and every post-flip move deadlocked.
        proximity = self.skills_config.turn_grasp_mode == "proximity"
        pinching = torch.as_tensor(
            np.isin(selections, tuple(self._ROTATOR_GRASPISH_IDS)), device=self.device
        )
        turning_t = torch.as_tensor(turning, device=self.device)
        face = self._face_tensor()
        grasping = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for face_idx in torch.unique(face[turning_t]):
            mask = turning_t & (face == face_idx)
            link = self._active_layer_link(int(face_idx))
            # Per-jaw test as well: the rim pinch presses the same layer link with both
            # jaws, but the pairwise `is_grasping` call also fails if their contact
            # forces land on different sub-links of the slab.
            grasping |= mask & self._grasp_any(self.agent_b, [link])
            if proximity:
                near = torch.linalg.norm(self.agent_b.tcp_pos - link.pose.p, dim=1) < _NEAR_TOL
                grasping |= mask & near & pinching
        active = turning_t & grasping & (self._plan_index < self.skills_config.plan_length)
        if not bool(active.any()):
            return
        angle = self._active_layer_angle()
        delta = (self._move_target - angle).clamp(-ARM_DELTA_LIMIT, ARM_DELTA_LIMIT)
        qpos = self.cube.get_qpos()
        qvel = self.cube.get_qvel()
        cols = torch.as_tensor([self.layer_col[f] for f in FACES], device=self.device)[face]
        sel = active.nonzero(as_tuple=True)[0]
        qpos[sel, cols[sel]] += delta[sel]
        qvel[sel, cols[sel]] = 0.0
        if self.gpu_sim_enabled:
            px = self.scene.px
            px.gpu_fetch_articulation_qpos()
            px.gpu_fetch_articulation_qvel()
        self.cube.set_qpos(qpos)
        self.cube.set_qvel(qvel)
        if self.gpu_sim_enabled:
            px.gpu_apply_articulation_qpos()
            px.gpu_apply_articulation_qvel()
            px.gpu_update_articulation_kinematics()

    def _before_control_step(self):
        # Clamp every layer while the holder does not grasp the core (the FaceTurn
        # face-lock rule lifted to all six layers; a guarantee, not a reward).
        if not self.skills_config.lock_layers_unless_held:
            return
        held = self._holder_holds()
        want = ~held  # every layer locks while the core is not held
        qpos = self.cube.get_qpos()
        was_locked = self._layer_locked.clone()
        for col in range(6):
            self._layer_locked[:, col], self._layer_lock_angle[:, col] = update_face_lock(
                self._layer_locked[:, col],
                self._layer_lock_angle[:, col],
                want,
                qpos[:, col],
            )
        self._any_layer_locked = bool(self._layer_locked.any())
        # A freshly engaged lock latches at the NEAREST DETENT, not the raw angle:
        # the detents are the rest states the solver itself prefers, so the
        # corrective pull converges instead of fighting it (a mid-air raw latch --
        # e.g. right before the set-down's tumble -- left slabs at angles the
        # solver continuously argued with, which the GPU depenetration expelled
        # as a constant-velocity glide).
        engaged = self._layer_locked & ~was_locked
        if bool(engaged.any()):
            detents = torch.as_tensor(LAYER_DETENTS_RAD, device=self.device)
            angle = self._layer_lock_angle
            snapped = detents[torch.argmin((angle[..., None] - detents).abs(), dim=-1)]
            self._layer_lock_angle = torch.where(engaged, snapped, angle)
        # The one PER-DECISION position correction (rate-limited): pulling back a
        # layer that deviated past the tolerance. Never per substep -- continuous
        # kinematic writes fight the GPU solver (see the config's lock notes).
        if self._any_layer_locked:
            deviation = qpos - self._layer_lock_angle
            pull = self.skills_config.lock_pull_step_rad
            correct = self._layer_lock_angle + deviation.clamp(-pull, pull)
            qpos = torch.where(
                self._layer_locked & (deviation.abs() >= self.skills_config.lock_write_tol_rad),
                correct,
                qpos,
            )
            if self.gpu_sim_enabled:
                # TODO(review): GPU path unverified (no GPU on the dev machine), same as the
                # FaceTurn face lock it generalizes.
                px = self.scene.px
                px.gpu_fetch_articulation_qpos()
                px.gpu_fetch_articulation_qvel()
            self.cube.set_qpos(qpos)
            if self.gpu_sim_enabled:
                px.gpu_apply_articulation_qpos()
                px.gpu_update_articulation_kinematics()

    def _after_simulation_step(self):
        # Zero the locked layers' joint velocities after every physics substep
        # (the FaceTurn mechanism generalized; per joint column, GPU fetch/apply as
        # there). VELOCITIES ONLY: a velocity write darts the layer's wiggle but
        # cannot inject position, so the solver's contact cycle has nothing to
        # expel. Position corrections happen once per DECISION (in
        # `_before_control_step`), never per substep -- continuous kinematic
        # position writes fight the GPU solver into a constant-velocity glide.
        if not (self.skills_config.lock_layers_unless_held and self._any_layer_locked):
            return
        if self.gpu_sim_enabled:
            # TODO(review): GPU path unverified (no GPU on the dev machine), same as the
            # FaceTurn face lock it generalizes.
            px = self.scene.px
            px.gpu_fetch_articulation_qpos()
            px.gpu_fetch_articulation_qvel()
        qvel = self.cube.get_qvel()
        qvel = torch.where(self._layer_locked, torch.zeros_like(qvel), qvel)
        self.cube.set_qpos(self.cube.get_qpos())  # no-op passthrough (see above)
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
        cols = torch.as_tensor([self.layer_col[f] for f in FACES], device=self.device)
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
        self._refresh_move_targets(torch.nonzero(within).flatten().cpu().numpy())

    def _compute_needed_flip(self) -> list:
        """Per env: the single `Flip` that brings the current move's face up (or None)."""
        needed: list = [None] * self.num_envs
        if self.skills_config.plan_length == 0:
            return needed
        quat = sapien_quat_to_xyzw(self.cube.pose.q).cpu().numpy()
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
            self._agent(arm).robot.pose.to_transformation_matrix().cpu().numpy() for arm in range(2)
        ]
        angle = self._active_layer_angle()
        target = self._move_target
        layer_at_target = ((angle - target).abs() < self.skills_config.angle_tol_rad).cpu().numpy()
        return RuntimeView(
            qpos=[q[:, :NUM_ARM_JOINTS].cpu().numpy() for q in qpos],
            arm_speed=[q[:, :NUM_ARM_JOINTS].abs().max(dim=1).values.cpu().numpy() for q in qvel],
            jaw_speed=[q[:, NUM_ARM_JOINTS:].abs().max(dim=1).values.cpu().numpy() for q in qvel],
            base_pos=[b[:, :3, 3] for b in bases],
            base_rot=[b[:, :3, :3] for b in bases],
            cube_pos=self.cube.pose.p.cpu().numpy(),
            cube_quat=sapien_quat_to_xyzw(self.cube.pose.q).cpu().numpy(),
            cube_home=self.body_init_pos.cpu().numpy(),
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
                raise ValueError(
                    f"action dict keys {tuple(action)} do not match {AGENT_UIDS}"
                ) from exc
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
        holder_near = torch.linalg.norm(self.agent_a.tcp_pos - self.cube.pose.p, dim=1) < _NEAR_TOL
        if self.skills_config.plan_length > 0:
            face = self._face_tensor()
            proximity = self.skills_config.turn_grasp_mode == "proximity"
            selections_rot = self._runtime.selections()[1] if proximity else None
            _GRASPISH = self._ROTATOR_GRASPISH_IDS
            for face_idx in torch.unique(face):
                mask = face == face_idx
                link = self._active_layer_link(int(face_idx))
                rotator_grasp |= mask & self.agent_b.is_grasping(link)
                rotator_near |= mask & (
                    torch.linalg.norm(self.agent_b.tcp_pos - link.pose.p, dim=1) < _NEAR_TOL
                )
                if proximity:
                    # The scripted-motorics proximity pinch (see
                    # `turn_grasp_mode`): navigating near the layer with a
                    # grasp/turn selected establishes the grasp fact; contact
                    # above still qualifies on its own.
                    # NOTE: np.isin(array, frozenset) is all-False on numpy >= 2 (the set is
                    # read as a 0-d object, not an iterable of values), which silently killed
                    # the whole proximity pinch. Always pass a tuple/list here.
                    pinching = np.isin(selections_rot, tuple(_GRASPISH))
                    rotator_grasp |= (
                        mask & rotator_near & torch.as_tensor(pinching, device=self.device)
                    )
        up_is_move = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.skills_config.plan_length > 0:
            quat = sapien_quat_to_xyzw(self.cube.pose.q).cpu().numpy()
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
                        uid: torch.as_tensor(joint[arm], dtype=torch.float32, device=self.device)
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
            self._turn_assist()
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
                "plan_moves_remaining": (remaining if arm == 0 else remaining.clone()).unsqueeze(1),
                "cube_position": self.cube.pose.p if arm == 0 else self.cube.pose.p.clone(),
                "cube_orientation_quat": (
                    sapien_quat_to_xyzw(self.cube.pose.q)
                    if arm == 0
                    else sapien_quat_to_xyzw(self.cube.pose.q).clone()
                ),
                "active_layer_angle": (
                    angle / math.pi if arm == 0 else angle.clone() / math.pi
                ).unsqueeze(1),
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
        facts_rotator_grasp = self._decision_facts().rotator_grasp
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
            INFO_ROTATOR_GRASP_FACT: torch.as_tensor(facts_rotator_grasp, device=self.device),
        }

    def compute_dense_reward(self, obs, action: torch.Tensor, info: dict):
        cfg: SkillsRewardConfig = self.skills_config.reward
        return cfg.success_bonus * info[INFO_SUCCESS].float() - cfg.step_penalty

    def compute_normalized_dense_reward(self, obs, action: torch.Tensor, info: dict):
        # The skill reward is already the per-decision team quantity in [-0.7, 1]: the
        # normalised and dense forms coincide by construction.
        return self.compute_dense_reward(obs=obs, action=action, info=info)
