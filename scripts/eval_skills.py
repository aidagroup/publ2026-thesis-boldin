"""[cloud] Evaluation harness of the skill-level cube assembly (openspec tasks 4.1-4.4).

Runs trained FaceTurnSkills-v0 checkpoint pairs in the registered env and reports the
coordination metrics of `callosum.eval.metrics`: success rate, time to success in
decisions (mean +- std over episodes), per-hand useless-action rate and partner-primitive
prediction accuracy. `--cross-play` evaluates every ordered (holder, rotator) checkpoint
pair -- the self-play diagonal against zero-shot cross-play, the thesis's main claim;
`--recovery` adds a second, failure-injected run per pair and compares it with the clean
one. Every run prints one summary line, logs `eval/<pair>/*` scalars to TensorBoard and
lands in `--out` as JSON together with the full config (reproducible launch).

Checkpoints are the IPPO trainer's run directories (their `latest.pt` is used) or `.pt`
files directly; several at once and/or as glob patterns:

    uv run python scripts/eval_skills.py \
        --checkpoints 'runs/skills_oracle_s*' --partner-mode oracle \
        --episodes 100 --cross-play --recovery --out results.json

Policies are loaded the way `callosum.training.ippo --eval-only` loads them: the actor
weights come from the checkpoint payload's per-agent state dicts (agent_a = holder,
agent_b = rotator), driven deterministically. The categorical policies are agent B's
`callosum.training._categorical.CategoricalPolicy` and the per-agent inputs are cut by
agent B's `callosum.training._skills_agent_obs.SkillsObsBuilder` (the documented eval
contract). The env itself is agent A's `callosum.envs.face_turn_skills`, still being
written in parallel: its module import and the `gym.make` kwargs below carry the remaining
`TODO(integration)` marks. All heavy imports (torch, gym, mani_skill) stay inside the
functions so this file remains lintable and importable on macOS.

Partner-primitive predictions are NOT recorded yet: `predicted`/`predicted-direct` need
the externally-supplied classifier block (task 3.3 / the spring Bi-JEPA change), so those
modes exit early, and the oracle/none runs record no predictions either
(TODO(integration); `partner_prediction_accuracy` stays None until then).

Never run this on the Mac: it needs the GPU sim. The metric layer it reports is tested
on the Mac in `tests/test_eval_metrics.py`.
"""

import argparse
import glob
import json
import math
import re
import sys
import time
import zlib
from collections.abc import Sequence
from pathlib import Path

from callosum.envs._skills import (
    PARTNER_VISIBILITY_MODES,
    RandomizationConfig,
    Role,
    num_primitives,
    validate_partner_visibility,
)
from callosum.envs._skills_obs import (
    INFO_SUCCESS,
    INFO_TIME_TO_SUCCESS,
    INFO_USELESS_HOLDER,
    INFO_USELESS_ROTATOR,
)
from callosum.eval.metrics import (
    CrossPlayMatrix,
    EpisodeRecord,
    crossplay_matrix,
    recovery_summary,
    success_rate,
    summarize,
    summary_dict,
)


def expand_checkpoints(patterns: Sequence[str]) -> list[tuple[str, Path]]:
    """Resolve `--checkpoints` patterns into `(label, checkpoint file)` pairs.

    A pattern may be a glob; each match is either a run directory (its `latest.pt` is
    used) or a `.pt` file. The label is the run directory's name (or the file's stem) --
    the matrix row/column identity -- made unique with a numeric suffix on collisions.
    The result is sorted by label, so pair grids and matrices are deterministic.
    """
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"--checkpoints pattern matched nothing: {pattern}")
        paths.extend(Path(match) for match in matches)
    resolved: list[Path] = []
    for path in paths:
        target = path / "latest.pt" if path.is_dir() else path
        if not target.is_file():
            raise FileNotFoundError(
                f"no checkpoint at {path} (expected a run dir with latest.pt or a .pt file)"
            )
        if target not in resolved:
            resolved.append(target)
    labeled: list[tuple[str, Path]] = []
    used: set[str] = set()
    for target in resolved:
        base = target.parent.name if target.name == "latest.pt" else target.stem
        label, suffix = base, 1
        while label in used:
            suffix += 1
            label = f"{base}_{suffix}"
        used.add(label)
        labeled.append((label, target))
    return sorted(labeled, key=lambda item: item[0])


def pair_grid(
    checkpoints: Sequence[tuple[str, Path]], cross_play: bool
) -> list[tuple[tuple[str, Path], tuple[str, Path]]]:
    """`(holder, rotator)` pairs to evaluate.

    Default: one self-play pair per checkpoint. With `cross_play`: every ordered pair --
    policy A of seed 1 as holder with policy B of seed 2 as rotator and vice versa, the
    self-play diagonal included.
    """
    if not cross_play:
        return [(checkpoint, checkpoint) for checkpoint in checkpoints]
    return [(holder, rotator) for holder in checkpoints for rotator in checkpoints]


def pair_seed(base: int, holder_label: str, rotator_label: str) -> int:
    """A deterministic reset seed per (base, pair); `hash()` is salted, crc32 is not."""
    return zlib.crc32(f"{base}|{holder_label}|{rotator_label}".encode())


def _tag(holder_label: str, rotator_label: str) -> str:
    """The TensorBoard/JSON pair tag (labels sanitised to tag-safe characters)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{holder_label}_x_{rotator_label}")


# --- [cloud] environment and policies (lazy imports) ---------------------------------------


def _make_env(args: argparse.Namespace, inject: bool):
    """The vectorised FaceTurnSkills env, as `ippo.make_envs` builds it (state, no render)."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers the stock ManiSkill envs)

    from callosum.envs._sim_compat import prepare_sim_backend

    prepare_sim_backend(args.sim_backend)
    kwargs: dict = {
        "num_envs": args.num_envs,
        "obs_mode": "state",
        "sim_backend": args.sim_backend,
        "render_backend": "none",
        "render_mode": None,
        "partner_mode": args.partner_mode,
        "plan_length": args.plan_length,
    }
    if args.max_decisions is not None:
        # One env step is one decision (design decision 5), so the episode limit counts
        # decisions.
        kwargs["max_episode_steps"] = args.max_decisions
    if inject:
        kwargs["grasp_failure_prob_range"] = tuple(args.grasp_failure_prob)
    else:
        # The clean arm of the recovery comparison must have NO failure injection at
        # all: grasp failures and slips both off (delays/speeds randomisation stays,
        # recovery isolates the injections by design).
        kwargs["grasp_failure_prob_range"] = (0.0, 0.0)
        kwargs["slip_prob_range"] = (0.0, 0.0)
    return gym.make(args.env_id, **kwargs)


def _agent_uids(env) -> tuple[str, str]:
    """The two agent uids from the env's dict action space, holder first."""
    uids = tuple(env.action_space.spaces)
    if len(uids) != 2:
        raise ValueError(f"expected a two-agent env, got action space keys {uids}")
    # TODO(integration): verify the first uid is the holder (agent_a), as in FaceTurn-v0.
    return uids[0], uids[1]


def _np(x):
    """A 1-D numpy view of a per-env tensor/array/scalar from the env's info dicts."""
    import numpy as np

    arr = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return np.atleast_1d(arr)


def _load_pair(holder_path: Path, rotator_path: Path, device, obs_dims: Sequence[int]):
    """The `(holder, rotator)` categorical policies with actor weights from checkpoints.

    Loads the way `ippo --eval-only` does (per-agent state dicts of the payload,
    agent_a = holder, agent_b = rotator) and rejects checkpoints whose input widths do
    not match the obs builder's `obs_dims` (`check_warm_start_compat`).
    """
    from callosum.training._categorical import CategoricalPolicy
    from callosum.training._checkpoint import check_warm_start_compat, load_checkpoint

    holder = CategoricalPolicy(obs_dims[0], num_primitives(Role.HOLDER)).to(device)
    rotator = CategoricalPolicy(obs_dims[1], num_primitives(Role.ROTATOR)).to(device)
    holder_payload = load_checkpoint(holder_path, map_location=device)
    rotator_payload = (
        holder_payload
        if rotator_path.resolve() == holder_path.resolve()
        else load_checkpoint(rotator_path, map_location=device)
    )
    check_warm_start_compat(holder_payload, list(obs_dims))
    check_warm_start_compat(rotator_payload, list(obs_dims))
    holder.load_state_dict(holder_payload["agents"]["agent_a"])
    rotator.load_state_dict(rotator_payload["agents"]["agent_b"])
    return holder.eval(), rotator.eval()


def evaluate_pair(
    holder_ckpt: tuple[str, Path],
    rotator_ckpt: tuple[str, Path],
    args: argparse.Namespace,
    *,
    inject: bool,
    writer=None,
    tag: str = "",
) -> list[EpisodeRecord]:
    """Run `args.episodes` episodes of one `(holder, rotator)` pair; the records.

    One env step is one decision. The loop keeps per-env accumulators (decision count,
    useless flags, and -- later -- predictions) and emits one `EpisodeRecord` per episode
    as it finishes (success terminates, the decision limit truncates, auto-reset
    continues the batch until enough episodes are in). The reset seed is derived from
    `(seed, pair)`, so a pair's episodes are reproducible from the config alone.
    """
    import numpy as np
    import torch

    import callosum.envs.face_turn_skills  # noqa: F401  TODO(integration): agent A's module
    from callosum.training._skills_agent_obs import SkillsObsBuilder

    env = _make_env(args, inject)
    try:
        device = env.device
        uid_holder, uid_rotator = _agent_uids(env)
        obs, infos = env.reset(seed=pair_seed(args.seed, holder_ckpt[0], rotator_ckpt[0]))
        builder = SkillsObsBuilder.from_env_obs(
            env.unwrapped.get_obs(unflattened=True),
            obs,
            args.partner_mode,
            (uid_holder, uid_rotator),
        )
        holder, rotator = _load_pair(holder_ckpt[1], rotator_ckpt[1], device, builder.obs_dims)
        n = env.num_envs
        limit = args.max_decisions
        if limit is None:
            try:
                from mani_skill.utils import gym_utils

                limit = int(gym_utils.find_max_episode_steps_value(env))
            except Exception:  # noqa: BLE001  (any backend quirk falls back to a guard)
                limit = 200
        guard = (math.ceil(args.episodes / max(1, n)) + 1) * max(1, limit)

        records: list[EpisodeRecord] = []
        decisions = np.zeros(n, dtype=np.int64)
        useless_holder = np.zeros(n, dtype=np.int64)
        useless_rotator = np.zeros(n, dtype=np.int64)
        steps = 0
        with torch.no_grad():
            while len(records) < args.episodes and steps < guard:
                inputs = builder(obs, infos)
                # `get_action(deterministic=True)` is the logits' argmax: primitive ids
                # `(batch,)` long. TODO(integration): agent A's action space must take
                # exactly these per-env id tensors.
                actions = {
                    uid_holder: holder.get_action(inputs[0], deterministic=True),
                    uid_rotator: rotator.get_action(inputs[1], deterministic=True),
                }
                obs, _, terminated, truncated, infos = env.step(actions)
                steps += 1
                decisions += 1
                useless_holder += _np(infos.get(INFO_USELESS_HOLDER, 0)).astype(np.int64)
                useless_rotator += _np(infos.get(INFO_USELESS_ROTATOR, 0)).astype(np.int64)
                # TODO(integration): record partner-primitive predictions here once the
                # predictor of task 3.3 exists (predicted / predicted-direct modes).
                done = _np(terminated).astype(bool) | _np(truncated).astype(bool)
                success = _np(infos.get(INFO_SUCCESS, 0)).astype(bool)
                sim_steps = infos.get(INFO_TIME_TO_SUCCESS)
                sim_steps_np = None if sim_steps is None else _np(sim_steps)
                for e in np.nonzero(done)[0]:
                    ok = bool(success[e])
                    records.append(
                        EpisodeRecord(
                            success=ok,
                            time_to_success=int(decisions[e]) if ok else None,
                            decisions=int(decisions[e]),
                            sim_steps_to_success=(
                                int(sim_steps_np[e]) if ok and sim_steps_np is not None else None
                            ),
                            useless_holder=int(useless_holder[e]),
                            useless_rotator=int(useless_rotator[e]),
                        )
                    )
                    decisions[e] = 0
                    useless_holder[e] = 0
                    useless_rotator[e] = 0
        if len(records) < args.episodes:
            print(
                f"WARNING: pair {holder_ckpt[0]}_x_{rotator_ckpt[0]}: only {len(records)} "
                f"of {args.episodes} episodes finished within the step guard"
            )
        if writer is not None and tag:
            _log_pair(writer, tag, records)
        return records[: args.episodes]
    finally:
        env.close()


# --- reporting ----------------------------------------------------------------------------


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def _pm(mean: float | None, std: float | None) -> str:
    return "-" if mean is None else f"{mean:.3f}+-{std:.3f}"


def _pair_line(holder: str, rotator: str, summary) -> str:
    return (
        f"pair holder={holder} rotator={rotator} | {summary.episodes} episodes | "
        f"succ {summary.success_rate:.3f} | time {_pm(summary.time_to_success_mean, summary.time_to_success_std)} "
        f"decisions | useless h {summary.useless_rate_holder:.3f} "
        f"r {summary.useless_rate_rotator:.3f}/decision | "
        f"pred {_fmt(summary.partner_prediction_accuracy)}"
    )


def _matrix_text(matrix: CrossPlayMatrix) -> str:
    width = max([len("holder\\rotator")] + [len(x) for x in matrix.holders + matrix.rotators])
    lines = ["cross-play matrix (rows = holder, cols = rotator):"]
    lines.append(
        "holder\\rotator".ljust(width)
        + "".join(rotator.rjust(width + 3) for rotator in matrix.rotators)
    )
    for holder in matrix.holders:
        cells = "".join(
            _fmt(matrix.cell(holder, rotator)).rjust(width + 3) for rotator in matrix.rotators
        )
        lines.append(holder.ljust(width) + cells)
    lines.append(
        f"self-play mean {_fmt(matrix.self_play_mean)} | "
        f"cross-play mean {_fmt(matrix.cross_play_mean)}"
    )
    return "\n".join(lines)


def _log_pair(writer, tag: str, records: list[EpisodeRecord]) -> None:
    summary = summarize(records)
    for name, value in (
        ("success_rate", summary.success_rate),
        ("time_to_success_mean", summary.time_to_success_mean),
        ("time_to_success_std", summary.time_to_success_std),
        ("useless_rate_holder", summary.useless_rate_holder),
        ("useless_rate_rotator", summary.useless_rate_rotator),
        ("partner_prediction_accuracy", summary.partner_prediction_accuracy),
    ):
        if value is not None:
            writer.add_scalar(f"eval/{tag}/{name}", value, 0)


def _make_writer(args: argparse.Namespace):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("WARNING: tensorboard unavailable; scalars go to stdout/--out only")
        return None
    log_dir = Path("runs") / f"eval_skills_{Path(args.out).stem}_{int(time.time())}"
    return SummaryWriter(str(log_dir))


# --- CLI ----------------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Run directories (their latest.pt) or .pt files; globs allowed. Several "
        "entries build the pair grid / cross-play matrix.",
    )
    parser.add_argument("--env-id", default="FaceTurnSkills-v0", help="(default: %(default)s)")
    parser.add_argument(
        "--episodes", type=int, default=100, help="Episodes per pair. (default: %(default)s)"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="Vectorised envs per pair run. (default: %(default)s)",
    )
    parser.add_argument(
        "--partner-mode",
        choices=PARTNER_VISIBILITY_MODES,
        default="none",
        help="Partner-visibility mode the policies were trained in. (default: %(default)s)",
    )
    parser.add_argument(
        "--plan-length",
        type=int,
        default=1,
        help="Plan moves per episode; must match the training task (E1: 1, E2: 2-3, E3: 5+). "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=None,
        help="Episode limit in decisions (default: the env's registered limit).",
    )
    parser.add_argument(
        "--cross-play",
        action="store_true",
        help="Evaluate every ordered (holder, rotator) checkpoint pair -- cross-play "
        "matrix with the self-play diagonal.",
    )
    parser.add_argument(
        "--recovery",
        action="store_true",
        help="Also run each pair with failure injection and compare against the clean "
        "run (recovery summary).",
    )
    parser.add_argument(
        "--grasp-failure-prob",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        default=list(RandomizationConfig().grasp_failure_prob_range),
        help="Per-hand grasp-failure probability range of the injected run "
        "(--recovery). (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base eval seed. (default: %(default)s)"
    )
    parser.add_argument(
        "--sim-backend",
        default="gpu",
        help="ManiSkill sim backend ([cloud] gpu). (default: %(default)s)",
    )
    parser.add_argument(
        "--out", default="results.json", help="JSON report path. (default: %(default)s)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: `python scripts/eval_skills.py [flags]` ([cloud] only)."""
    # The server runs this detached under nohup with stdout redirected to a file:
    # flush per line, like the trainer.
    sys.stdout.reconfigure(line_buffering=True)
    args = _parse_args(argv)
    validate_partner_visibility(args.partner_mode)
    if args.partner_mode in ("predicted", "predicted-direct"):
        raise SystemExit(
            f"--partner-mode {args.partner_mode} needs the external predictor block "
            "(task 3.3 / Bi-JEPA integration); TODO(integration)"
        )
    RandomizationConfig(grasp_failure_prob_range=tuple(args.grasp_failure_prob))  # validates
    checkpoints = expand_checkpoints(args.checkpoints)
    if args.cross_play and len(checkpoints) < 2:
        print("WARNING: --cross-play with a single checkpoint measures self-play only")
    pairs = pair_grid(checkpoints, args.cross_play)
    writer = _make_writer(args)
    print(
        f"eval {args.env_id} | partner-mode {args.partner_mode} | plan {args.plan_length} | "
        f"{args.episodes} episodes x {len(pairs)} pair(s) | sim {args.sim_backend} | "
        f"seed {args.seed}"
    )

    clean_records: dict[tuple[str, str], list[EpisodeRecord]] = {}
    pair_results: dict[tuple[str, str], float] = {}
    payload: dict = {
        "config": dict(vars(args), checkpoints=None),
        "checkpoints": {label: str(path) for label, path in checkpoints},
        "pairs": {},
    }
    for holder, rotator in pairs:
        records = evaluate_pair(
            holder, rotator, args, inject=False, writer=writer, tag=_tag(holder[0], rotator[0])
        )
        key = (holder[0], rotator[0])
        clean_records[key] = records
        pair_results[key] = success_rate(records)
        payload["pairs"][_tag(holder[0], rotator[0])] = dict(
            summary_dict(records), seed=pair_seed(args.seed, holder[0], rotator[0])
        )
        print(_pair_line(holder[0], rotator[0], summarize(records)), flush=True)

    if args.cross_play:
        matrix = crossplay_matrix(pair_results)
        print(_matrix_text(matrix))
        payload["crossplay"] = matrix.to_dict()

    if args.recovery:
        payload["recovery"] = {}
        for holder, rotator in pairs:
            tag = _tag(holder[0], rotator[0])
            records_with = evaluate_pair(
                holder, rotator, args, inject=True, writer=writer, tag=f"{tag}_inject"
            )
            recovery = recovery_summary(records_with, clean_records[(holder[0], rotator[0])])
            # The inject arm reuses the SAME pair seed as the clean arm (paired
            # comparison, common random numbers); record the actual seed.
            payload["recovery"][tag] = dict(
                recovery.__dict__,
                seed_inject=pair_seed(args.seed, holder[0], rotator[0]),
            )
            print(
                f"recovery holder={holder[0]} rotator={rotator[0]} | "
                f"succ {recovery.success_with:.3f} vs {recovery.success_without:.3f} "
                f"(delta {recovery.success_delta:+.3f}) | "
                f"time {_fmt(recovery.time_with)} vs {_fmt(recovery.time_without)} | "
                f"recovered {recovery.recovered_rate:.3f}",
                flush=True,
            )

    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
