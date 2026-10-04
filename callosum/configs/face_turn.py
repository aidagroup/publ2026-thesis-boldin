"""Reward-weight and success-tolerance config for the FaceTurn task."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class FaceTurnRewardConfig:
    """Dense-reward term weights, shaping options and success tolerances for FaceTurn-v0.

    See `callosum.envs.face_turn.FaceTurn.compute_dense_reward` (weights, shaping) and
    `.evaluate` (tolerances) for how each field is used.
    """

    # Term weights.
    weight_rotator_reach: float = 1.0
    weight_grasp: float = 1.0
    weight_angle_progress: float = 3.0
    weight_holder_reach: float = 1.0
    weight_holder_grasp: float = 1.0
    """Holder (agent_a) grasping the cube body. Same weight as the rotator's grasp."""
    # Separate weights: position (m) and rotation (rad) drift are not on
    # commensurate scales, and neither are their tolerances below.
    weight_body_pos_drift: float = 5.0
    weight_body_rot_drift: float = 5.0

    # Shaping options.
    gate_rotator_on_holder: bool = True
    """Scale the rotator's grasp and angle-progress terms by a gate that depends on the holder
    grasping the body (the rotator's reach term stays ungated). The order matters physically:
    two open grippers next to the cube collide, so the holder has to clamp the body first."""
    rotator_gate_floor: float = 0.5
    """Gate value while the holder is NOT grasping: `gate = floor + (1 - floor) * holder_grasp`.
    0.0 is a hard 0/1 gate, 1.0 no gate. A soft floor is the default: a hard gate leaves the
    rotator without any grasp/turn gradient until the holder has found its (binary, contact- and
    force-based) grasp, and `is_grasping` can flicker while the face is being turned, which would
    make the rotator's reward flicker with it. With 0.5 the right order is still worth twice as
    much, but the rotator can learn to grasp and turn while the holder is still learning."""
    hinge_drift_penalty: bool = True
    """Penalise body drift only beyond the success tolerances: `max(0, drift - tol)`. Without
    the hinge any contact-induced micro-drift is punished (and the holder is taught to let go)
    even though it is within what counts as success."""
    angle_progress_shape: Literal["linear", "tanh"] = "linear"
    """"linear": `clamp(angle / target, 0, 1)`, a constant gradient over the whole 0 to 90 deg
    range. "tanh": `1 - tanh(2 * remaining)`, ManiSkill's turn_faucet draft shape, which is flat
    (about 0.004) around 0 deg."""

    # Success bonus.
    success_bonus: float = 100.0
    """One-step bonus on the step where `info["success"]` is True, in *normalised*-reward units
    (the dense reward gets `success_bonus * max_positive_reward`, so dense and normalised stay
    consistent). Why it exists: success terminates the episode, and every other term is positive
    on every step (about 0.6 near the goal, at most 1.0), so with gamma = 0.99 a policy that
    lingers next to the goal collects about as much as one that finishes: finishing is not the
    best outcome. Server evidence (30M-step IPPO run): eval success_once 0.81-0.875 but
    success_at_end 0, and late in training the train return fell while train success stayed
    about 0.5, because successful episodes end early and collect less return.

    Default 100 = 1 / (1 - gamma) for gamma = 0.99. The normalised per-step reward is at most 1,
    so any continuation is worth less than `sum_k gamma^k * 1 = 100`, and finishing (bonus plus
    the step's own reward) beats every continuation strictly, whatever the policy does. This
    needs the trainer not to bootstrap on success: the trainer treats success as a true
    terminal (value 0 afterwards, only time-limit truncations bootstrap), otherwise the bonus
    would feed back through V(final obs) and inflate the value function. A smaller bonus is
    enough if the policy's achieved per-step reward near the goal is below 1: it has to exceed
    `r / (1 - gamma)` for that r (about 0.6 / 0.01 = 60 at the rewards observed late in
    training), and a smaller value is gentler on the value function. Value targets move from
    about 60 to about 100 to 160; the critic and the actor are separate networks, advantages
    are normalised per minibatch and Adam is insensitive to the gradient scale, so this is
    expected to be benign. If the value loss destabilises, lower it to about 60 first. Scale
    with the discount: `1 / (1 - gamma)` for other gamma. 0 disables the bonus."""

    # Success/stability tolerances.
    angle_tol: float = 0.05  # rad (~3 deg) from the pi/2 target
    body_pos_tol: float = 0.01  # m of body-center drift from its initial pose
    body_rot_tol: float = 0.1  # rad (~6 deg) of body rotation from its initial pose

    def __post_init__(self) -> None:
        if not 0.0 <= self.rotator_gate_floor <= 1.0:
            raise ValueError(f"rotator_gate_floor must be in [0, 1], got {self.rotator_gate_floor}")
        if self.success_bonus < 0:
            raise ValueError(f"success_bonus must be >= 0, got {self.success_bonus}")
        if self.angle_progress_shape not in ("linear", "tanh"):
            raise ValueError(
                f"angle_progress_shape must be 'linear' or 'tanh', got {self.angle_progress_shape!r}"
            )

    @property
    def max_positive_reward(self) -> float:
        """Sum of the positive, bounded ([0, 1]) term weights: the divisor of the normalised
        dense reward. The drift penalty is excluded (it is ~0 in the successful case) and the
        gate only scales terms down, so the normalised reward stays <= 1 (not counting the one-step
        success bonus)."""
        return (
            self.weight_rotator_reach
            + self.weight_grasp
            + self.weight_angle_progress
            + self.weight_holder_reach
            + self.weight_holder_grasp
        )

    @property
    def dense_success_bonus(self) -> float:
        """The success bonus in raw dense-reward units (`success_bonus * max_positive_reward`)."""
        return self.success_bonus * self.max_positive_reward

    def add_success_bonus(self, dense_reward, success):
        """`dense_reward` plus the dense success bonus on the entries where `success` is True.

        Works on any tensor-like pair (`success` needs `.to(dtype)`), e.g. torch tensors of shape
        `(num_envs,)`; no torch import here so the config stays dependency-free.
        """
        return dense_reward + self.dense_success_bonus * success.to(dense_reward.dtype)
