"""RubikCube-v0: solve a full 3x3x3 cube by repeated FaceTurn-style quarter turns.

Physically identical to FaceTurn (same articulation, same holder/rotator grasp
handles, same per-turn staircase reward) -- see `callosum.envs.face_turn`. On
top of that, each env carries a LOGICAL `_rubik.RubikState` (colours only, no
geometry): whenever a quarter turn completes (the joint reaches +-TARGET_FACE_
ANGLE while both arms are gripping), the move is read off from which face the
rotator is looking at (`_rubik.facing_face`) and the turn's sign, applied to
the colours via the move tables, and the physical joint is snapped back to 0
so the next turn starts from the same "just gripped, angle zero" state
FaceTurn was tuned against. A held cube can therefore be turned any number of
times per episode, not just once.
"""

import numpy as np
import torch
from mani_skill.utils.registration import register_env

from callosum.configs.rubik import RubikRewardConfig
from callosum.envs import _rubik
from callosum.envs.face_turn import TARGET_FACE_ANGLE, FaceTurn

_DEFAULT_REWARD_CONFIG = RubikRewardConfig()

# _rubik.MOVES fixes the row order of the (12, 54) move-table tensor below.
_MOVE_IDS = {move: i for i, move in enumerate(_rubik.MOVES)}


@register_env("RubikCube-v0", max_episode_steps=300)
class RubikCube(FaceTurn):
    """Bimanual Rubik's-cube task on top of FaceTurn's holder/rotator mechanism.

    Success is the LOGICAL state being solved (all 54 facelets match their
    face's centre), not a single completed turn -- see `evaluate`.
    """

    def __init__(
        self,
        *args,
        reward_config: RubikRewardConfig = _DEFAULT_REWARD_CONFIG,
        **kwargs,
    ):
        super().__init__(*args, reward_config=reward_config, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # (12, 54) gather table, built once so applying a completed turn to
        # however many envs finished one this step is a single batched
        # torch.gather -- no python loop over envs. Row i is the permutation
        # for _rubik.MOVES[i]; _MOVE_IDS maps a move name back to that row.
        move_perms = np.stack([_rubik.MOVE_TABLES[m] for m in _rubik.MOVES])
        self._move_tables = torch.tensor(move_perms, dtype=torch.long, device=self.device)
        # Facelet colour == face index once solved (_rubik.SOLVED); centres
        # never move, so this reference never needs recomputing per env.
        self._solved_ref = torch.tensor(_rubik.SOLVED, dtype=torch.long, device=self.device)

        self.cube_colours = torch.zeros(
            (self.num_envs, _rubik.N_FACELETS), dtype=torch.long, device=self.device
        )
        self.moves_applied = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._turn_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._initial_solved_facelets = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Places both arms + the cube body, and zeroes the face joint --
        # see FaceTurn._initialize_episode.
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            # _rubik.scramble is one cube at a time (pure numpy, no batch
            # axis), so this loop is over the RESET batch only -- at most
            # num_envs, and only at episode boundaries, not every step. The
            # per-step move application below has no such loop.
            colours = np.empty((b, _rubik.N_FACELETS), dtype=np.int64)
            for k in range(b):
                state, _ = _rubik.scramble(self.reward_config.scramble_depth, self._batched_episode_rng[env_idx[k]])
                colours[k] = state.colours
            self.cube_colours[env_idx] = torch.from_numpy(colours).to(self.device)
            self.moves_applied[env_idx] = 0
            self._initial_solved_facelets[env_idx] = (
                self.cube_colours[env_idx] == self._solved_ref
            ).sum(dim=1)

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if "state" in self.obs_mode:
            # (num_envs, 54, 6) one-hot, flattened to (num_envs, 324). Shared
            # task field (no agent_x_ prefix): both arms need to see the
            # cube's logical state, not just their own proprioception.
            # Bi-JEPA fix: narrow shared cube-colours field. Full 324-dim is bit-identical
            # between agents (near-trivial target). Use face-centre colours (54) plus
            # a small learned encoding later; for now keep raw but document issue.
            one_hot = torch.zeros((self.num_envs, _rubik.N_FACELETS, 6), device=self.device)
            one_hot.scatter_(2, self.cube_colours.unsqueeze(-1), 1.0)
            obs["cube_colours"] = one_hot.reshape(self.num_envs, -1)  # TODO(3.2): encode to < 54 dims
        return obs

    def _apply_completed_turns(self, env_idx: torch.Tensor, face_angle: torch.Tensor) -> None:
        """Permute colours for `env_idx` (envs whose turn just completed) and
        reset their physical joint so the next turn starts from angle zero."""
        # facing_face is pure-numpy, one cube at a time -- this loop is over
        # only the envs that finished a turn THIS step (usually a handful),
        # never all num_envs. The colour update right after is one batched
        # gather across all of them at once.
        quats = self.cube.pose.q[env_idx].detach().cpu().numpy()
        signs = face_angle[env_idx].detach().cpu().numpy()
        move_ids = torch.empty(env_idx.shape[0], dtype=torch.long, device=self.device)
        for k in range(env_idx.shape[0]):
            face = _rubik.facing_face(quats[k], (0.0, 1.0, 0.0))
            # Positive angle -> counter-clockwise (right-hand rule, axis +y) -> "X'";
            # negative -> "X". See handoff item 2 / audit C.
            move_ids[k] = _MOVE_IDS[face + "'" if signs[k] > 0 else face]

        perm = self._move_tables[move_ids]  # (n, 54)
        self.cube_colours[env_idx] = torch.gather(self.cube_colours[env_idx], 1, perm)
        self.moves_applied[env_idx] += 1

        # Snap back to angle zero so the next turn starts from the same
        # "just gripped" state FaceTurn's staircase reward was tuned against.
        #
        # `joint.qpos` is a live view into px.cuda_articulation_qpos, so the
        # write lands in the buffer -- but PhysX only reads that buffer on an
        # explicit apply, and the next fetch would otherwise overwrite it. The
        # qpos SETTER is no use here either: it masks by scene._reset_mask,
        # which outside a reset does not select the envs we mean.
        self.face_link.joint.qpos[env_idx] = 0.0
        self.face_link.joint.qvel[env_idx] = 0.0
        self.scene._gpu_apply_all()
        self.scene.px.gpu_update_articulation_kinematics()
        self.scene._gpu_fetch_all()

    def evaluate(self):
        cfg = self.reward_config
        base = super().evaluate()
        face_angle = base["face_angle"]

        holder_grasped = self.agent_a.is_grasping(self.body_link)
        rotator_grasped = self.agent_b.is_grasping(self.face_link)
        turn_done = (
            (torch.abs(face_angle) >= TARGET_FACE_ANGLE - cfg.angle_tol)
            & holder_grasped
            & rotator_grasped
        )
        # A one-step flag, not the cumulative count: a term proportional to
        # moves_applied pays every step for turns already made, so freezing
        # after one turn beats doing anything else.
        self._turn_completed = turn_done
        if turn_done.any():
            # Reads the PRE-reset face_angle above for this step's obs/reward
            # (the policy gets credit for the completed turn); the joint
            # reset here only affects what the NEXT evaluate() call reads.
            self._apply_completed_turns(torch.nonzero(turn_done, as_tuple=True)[0], face_angle)

        solved_facelets = (self.cube_colours == self._solved_ref).sum(dim=1)
        success = solved_facelets == _rubik.N_FACELETS

        return {
            "success": success,
            "face_angle": face_angle,
            "is_body_stable": base["is_body_stable"],
            "is_lifted": base["is_lifted"],
            "solved_facelets": solved_facelets,
            "moves_applied": self.moves_applied,
            "turn_completed": self._turn_completed,
        }

    def compute_dense_reward(self, obs, action: torch.Tensor, info: dict):
        cfg = self.reward_config
        reward = super().compute_dense_reward(obs=obs, action=action, info=info)
        facelets_progress = (
            info["solved_facelets"] - self._initial_solved_facelets
        ).float() / _rubik.N_FACELETS
        return (
            reward
            + cfg.weight_facelets * facelets_progress
            + cfg.weight_move * info["turn_completed"].float()
            + cfg.bonus_solved * info["success"].float()
        )

    @property
    def reward_normalization_divisor(self) -> float:
        cfg = self.reward_config
        return (
            super().reward_normalization_divisor
            + cfg.weight_facelets
            + cfg.weight_move
            + cfg.bonus_solved
        )
