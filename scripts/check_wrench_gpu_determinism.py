"""Determinism check of the GPU CubeWrench-v0 backend (`mujoco_warp` + CUDA graph).

The GPU experiment numbers are only trustworthy if the same seed/actions produce the same
trajectory. This script checks, on the GPU pod:

1. **env determinism**: two independently built `WrenchCubeBatchGPU` instances with the same
   config/seed, reset with the same episode seeds and stepped with the same action tensors,
   stay BIT-IDENTICAL (`torch.equal`, not just close) in `qpos`/`qvel`/observations/reward;
2. **CUDA-graph replay == eager loop**: an env rebuilt with `use_cuda_graph=False` (the
   launch-bound eager substep loop) reproduces the graph-replayed trajectory bit-identically;
3. **auto-reset determinism**: the same check across episode boundaries (the runs are long
   enough to finish episodes and re-seed the auto-reset stream).

Run on the pod (a plain `uv sync --frozen` removes the optional Warp stack):

    uv pip install --python /root/.callosum/venv/bin/python mujoco-warp==4.0.1
    python scripts/check_wrench_gpu_determinism.py --worlds 8 --steps 80

Exit code 0 when every comparison is bit-identical, 1 otherwise.
"""

import argparse
import sys

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI (worlds, steps, seed, device, partner mode)."""
    parser = argparse.ArgumentParser(
        description="Bit-identical determinism check of WrenchCubeBatchGPU."
    )
    parser.add_argument("--worlds", type=int, default=8, help="worlds per env")
    parser.add_argument("--steps", type=int, default=80, help="control steps to compare")
    parser.add_argument("--seed", type=int, default=0, help="base episode seed")
    parser.add_argument("--device", default="cuda", help="CUDA device")
    parser.add_argument(
        "--partner-mode", choices=("none", "oracle"), default="oracle", help="env partner mode"
    )
    args = parser.parse_args(argv)
    if args.worlds < 1 or args.steps < 1 or args.seed < 0:
        parser.error("--worlds/--steps must be >= 1 and --seed >= 0")
    return args


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    """Max abs difference (`0.0` for empty inputs)."""
    if a.size == 0:
        return 0.0
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def main(argv: list[str] | None = None) -> int:
    """Run the three determinism comparisons and print PASS/FAIL."""
    args = parse_args(argv)
    import torch

    from callosum.experiments.cube_wrench.config import wrench_cube_config
    from callosum.experiments.cube_wrench.env_gpu import WrenchCubeBatchGPU
    from callosum.experiments.cube_wrench.logic import AGENT_KEYS

    config = wrench_cube_config("default")
    envs = {
        "graph-a": WrenchCubeBatchGPU(
            args.worlds,
            config=config,
            partner_mode=args.partner_mode,
            seed=args.seed,
            device=args.device,
        ),
        "graph-b": WrenchCubeBatchGPU(
            args.worlds,
            config=config,
            partner_mode=args.partner_mode,
            seed=args.seed,
            device=args.device,
        ),
        "eager": WrenchCubeBatchGPU(
            args.worlds,
            config=config,
            partner_mode=args.partner_mode,
            seed=args.seed,
            device=args.device,
            use_cuda_graph=False,
        ),
    }
    print(
        f"determinism: worlds={args.worlds} steps={args.steps} seed={args.seed} "
        f"partner_mode={args.partner_mode} | graph={envs['graph-a'].cuda_graph} "
        f"| versions: {envs['graph-a'].software_versions()}",
        flush=True,
    )
    generator = torch.Generator().manual_seed(args.seed + 17)
    actions = [(torch.rand((args.worlds, 6), generator=generator) * 2.0 - 1.0) for _ in range(8)]
    for env in envs.values():
        env.reset(seed=args.seed)
    comparisons = {
        "graph-a vs graph-b (rng/order)": ("graph-a", "graph-b"),
        "graph vs eager substep loop": ("graph-a", "eager"),
    }
    max_diff = {key: {"qpos": 0.0, "qvel": 0.0, "obs": 0.0, "reward": 0.0} for key in comparisons}
    bit_identical = dict.fromkeys(comparisons, True)
    for step in range(args.steps):
        action = {role: actions[step % len(actions)].to(args.device) for role in AGENT_KEYS}
        steps = {}
        for name, env in envs.items():
            steps[name] = env.step(action)
        for label, (left, right) in comparisons.items():
            left_state = envs[left].state()
            right_state = envs[right].state()
            for field in ("qpos", "qvel"):
                a = left_state[field].cpu().numpy()
                b = right_state[field].cpu().numpy()
                max_diff[label][field] = max(max_diff[label][field], _max_abs(a, b))
                bit_identical[label] = bit_identical[label] and bool(
                    torch.equal(left_state[field], right_state[field])
                )
            for role in AGENT_KEYS:
                a = steps[left].obs[role].cpu().numpy()
                b = steps[right].obs[role].cpu().numpy()
                max_diff[label]["obs"] = max(max_diff[label]["obs"], _max_abs(a, b))
                bit_identical[label] = bit_identical[label] and bool(
                    torch.equal(steps[left].obs[role], steps[right].obs[role])
                )
            a = steps[left].reward.cpu().numpy()
            b = steps[right].reward.cpu().numpy()
            max_diff[label]["reward"] = max(max_diff[label]["reward"], _max_abs(a, b))
            bit_identical[label] = bit_identical[label] and bool(
                torch.equal(steps[left].reward, steps[right].reward)
            )
    failures: list[str] = []
    for label, diffs in max_diff.items():
        status = "BIT-IDENTICAL" if bit_identical[label] else "DIFFERS"
        print(
            f"  {label}: {status} | max |diff| qpos {diffs['qpos']:.1e} "
            f"qvel {diffs['qvel']:.1e} obs {diffs['obs']:.1e} reward {diffs['reward']:.1e}"
        )
        if not bit_identical[label]:
            failures.append(label)
    for name, env in envs.items():
        env.close()
    if failures:
        print("DETERMINISM FAIL: " + ", ".join(failures))
        return 1
    print("DETERMINISM PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
