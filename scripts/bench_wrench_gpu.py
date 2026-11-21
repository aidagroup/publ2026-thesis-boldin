"""Throughput benchmark: CPU `VecWrenchCube` loop vs the batched GPU `WrenchCubeBatchGPU`.

Measures environment steps per second of the two CubeWrench-v0 backends for the world counts
of the plan's GPU grid (256 / 1024 / 4096; the CPU loop can also be timed at the same counts to
show the speedup). "Steps/s" counts CONTROL steps summed over worlds
(`nworld * steps / elapsed`), the number the trainer's SPS counter reports; the GPU path also
prints the per-control-step wall time of one batch.

The benchmark uses one fixed random action batch for both backends (the reduced model has no
contacts, so the physics cost does not depend on the action values). The GPU numbers include
the tensor-side control logic (latency queue, observations, reward, auto-reset) and the small
per-step host synchronisation of the finished-episode statistics, i.e. what the trainer pays.

The GPU backend needs the optional `mujoco-warp` install (the `sandbox-gpu` extra,
Linux/CUDA only; the pod's plain `uv sync --frozen` removes it), so on the pod run:

    uv pip install --python /root/.callosum/venv/bin/python mujoco-warp
    python scripts/bench_wrench_gpu.py --nworld 256 1024 4096 --steps 100

Run it on the pod only (the CPU loop is slow and the plan forbids heavy sim on the laptop).
"""

import argparse
import sys
import time

import numpy as np

from callosum.envs._wrench_logic import AGENT_KEYS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark CubeWrench-v0: CPU Python-loop vectorisation vs mujoco_warp."
    )
    parser.add_argument(
        "--nworld",
        type=int,
        nargs="+",
        default=[256, 1024, 4096],
        help="world counts to benchmark (both backends)",
    )
    parser.add_argument("--steps", type=int, default=100, help="control steps per measurement")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("cpu", "warp"),
        default=["cpu", "warp"],
        help="backends to measure",
    )
    parser.add_argument(
        "--partner-mode", choices=("none", "oracle"), default="none", help="env partner mode"
    )
    parser.add_argument("--device", default="cuda", help="CUDA device of the warp backend")
    parser.add_argument("--seed", type=int, default=0, help="base episode seed")
    parser.add_argument(
        "--warmup", type=int, default=5, help="GPU warm-up steps (kernel compilation)"
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="disable the captured physics-substep graph (eager loop; ~20x slower launch-bound)",
    )
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error(f"--steps must be >= 1, got {args.steps}")
    if any(nworld < 1 for nworld in args.nworld):
        parser.error(f"--nworld values must be >= 1, got {args.nworld}")
    return args


def _fixed_actions(nworld: int, device: str | None = None):
    """One deterministic action batch per role (numpy, or CUDA tensors for the GPU path)."""
    rng = np.random.default_rng(0)
    actions = {
        role: rng.uniform(-1.0, 1.0, size=(nworld, 6)).astype(np.float32) for role in AGENT_KEYS
    }
    if device is None:
        return {role: value.astype(np.float64) for role, value in actions.items()}
    import torch

    return {
        role: torch.as_tensor(value, device=device, dtype=torch.float32)
        for role, value in actions.items()
    }


def bench_cpu(nworld: int, steps: int, partner_mode: str, seed: int) -> float:
    """Control steps per second of the CPU Python-loop vectorisation."""
    from callosum.training._vec_wrench import VecWrenchCube

    vec = VecWrenchCube(nworld, partner_mode, seed=seed)
    try:
        vec.reset(seed=seed)
        actions = _fixed_actions(nworld)
        start = time.perf_counter()
        for _ in range(steps):
            vec.step(actions)
        elapsed = time.perf_counter() - start
    finally:
        vec.close()
    return nworld * steps / elapsed


def bench_warp(
    nworld: int,
    steps: int,
    partner_mode: str,
    device: str,
    seed: int,
    warmup: int,
    use_cuda_graph: bool = True,
) -> tuple[float, float, bool]:
    """`(control steps/s, seconds per batch step, cuda-graph-enabled)` of the GPU backend."""
    from callosum.envs.wrench_cube.env_gpu import WrenchCubeBatchGPU

    env = WrenchCubeBatchGPU(
        nworld,
        partner_mode=partner_mode,
        seed=seed,
        device=device,
        use_cuda_graph=use_cuda_graph,
    )
    env.reset(seed=seed)
    actions = _fixed_actions(nworld, device=device)
    for _ in range(warmup):
        env.step(actions)
    env.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        env.step(actions)
    env.synchronize()
    elapsed = time.perf_counter() - start
    graph_enabled = env.cuda_graph
    if env._cuda_graph_error is not None:
        print(f"  (cuda graph unavailable: {env._cuda_graph_error})")
    env.close()
    return nworld * steps / elapsed, elapsed / steps, graph_enabled


def main(argv: list[str] | None = None) -> int:
    """Measure every requested world count and print the comparison table."""
    args = parse_args(argv)
    print(
        f"CubeWrench-v0 throughput | partner_mode={args.partner_mode} | steps={args.steps} "
        f"| device={args.device} | step = control step summed over worlds",
        flush=True,
    )
    print(
        f"{'nworld':>7} {'cpu steps/s':>14} {'warp steps/s':>14} {'speedup':>9} "
        f"{'warp ms/step':>13}"
    )
    cpu_at = {}
    warp_at = {}
    for nworld in args.nworld:
        if "cpu" in args.backends:
            cpu_sps = bench_cpu(nworld, args.steps, args.partner_mode, args.seed)
            cpu_at[nworld] = cpu_sps
        else:
            cpu_sps = float("nan")
        if "warp" in args.backends:
            warp_sps, warp_seconds, graph_enabled = bench_warp(
                nworld,
                args.steps,
                args.partner_mode,
                args.device,
                args.seed,
                args.warmup,
                use_cuda_graph=not args.no_cuda_graph,
            )
            graph_text = "graph" if graph_enabled else "eager"
            if not warp_at:
                print(f"warp physics substep loop: {graph_text}")
            warp_at[nworld] = warp_sps
        else:
            warp_sps, warp_seconds = float("nan"), float("nan")
        speedup = warp_sps / cpu_sps if cpu_sps and np.isfinite(cpu_sps) else float("nan")
        cpu_text = f"{cpu_sps:,.0f}" if np.isfinite(cpu_sps) else "-"
        warp_text = f"{warp_sps:,.0f}" if np.isfinite(warp_sps) else "-"
        speedup_text = f"{speedup:.1f}x" if np.isfinite(speedup) else "-"
        ms_text = f"{warp_seconds * 1e3:.2f}" if np.isfinite(warp_seconds) else "-"
        print(f"{nworld:>7} {cpu_text:>14} {warp_text:>14} {speedup_text:>9} {ms_text:>13}")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
