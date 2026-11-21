#!/usr/bin/env bash
# CubeWrench-v0 GPU backend: parity, benchmark and smoke training on the GPU pod.
#
# Everything here runs ON THE POD (the plan forbids GPU simulation on the laptop). The
# optional Warp stack is the `sandbox-gpu` extra (mujoco-warp is Linux/CUDA only), and the
# pod's plain `uv sync --frozen` removes it from the venv, so step 0 installs it again right
# before the run (a ~10 s cached install).
#
# Usage (from the repo checkout on the pod):
#   bash scripts/experiments/cube_wrench/run_wrench_gpu_smoke.sh                 # parity + benchmark + 2-update smoke
#   RUN_FULL=1 WAIT_FULL=1 bash scripts/experiments/cube_wrench/run_wrench_gpu_smoke.sh
#                                                        # + 300-update grid (none/oracle/predicted)
#   SEEDS="0" MODES="none oracle" RUN_FULL=1 bash scripts/experiments/cube_wrench/run_wrench_gpu_smoke.sh
#
# Env overrides: PYTHON, RUNS_DIR, DEVICE, BENCH_STEPS, NUM_ENVS, ROLLOUT_STEPS, UPDATES,
# SEEDS, MODES, WAIT_FULL, RUN_FULL.
set -euo pipefail

PYTHON="${PYTHON:-/root/.callosum/venv/bin/python}"
UV="${UV:-uv}"
DEVICE="${DEVICE:-cuda}"
RUNS_DIR="${RUNS_DIR:-runs}"
BENCH_STEPS="${BENCH_STEPS:-100}"
NUM_ENVS="${NUM_ENVS:-1024}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-64}"
UPDATES="${UPDATES:-300}"
SEEDS="${SEEDS:-0 1 2}"
MODES="${MODES:-none oracle predicted}"
SMOKE_ENVS="${SMOKE_ENVS:-64}"
SMOKE_ROLLOUT="${SMOKE_ROLLOUT:-16}"

echo "== 0. optional Warp install (uv sync --frozen removes it)"
if ! "$PYTHON" -c 'import mujoco_warp' >/dev/null 2>&1; then
  "$UV" pip install --python "$PYTHON" mujoco-warp==4.0.1
fi
"$PYTHON" -c 'import mujoco, mujoco_warp, warp; print("mujoco", mujoco.__version__,
  "| mujoco_warp", getattr(mujoco_warp, "__version__", "?"), "| warp", warp.__version__)'

echo "== 1. CPU-vs-GPU parity (qpos 1e-5, obs 1e-4)"
# Production path: sampled per-world friction, latency, injections, whole episodes + resets.
"$PYTHON" scripts/experiments/cube_wrench/check_wrench_gpu_parity.py \
  --controller oracle --friction sampled --partner-mode oracle --device "$DEVICE" --steps 120
"$PYTHON" scripts/experiments/cube_wrench/check_wrench_gpu_parity.py \
  --controller oracle --friction sampled --partner-mode none --device "$DEVICE" --steps 60
# Numerical isolation of the whole pipeline under random actions (friction is non-smooth).
"$PYTHON" scripts/experiments/cube_wrench/check_wrench_gpu_parity.py \
  --controller random --friction disabled --partner-mode oracle --device "$DEVICE" --steps 120
# Informational only: random actions + sampled Coulomb friction diverge after a velocity
# sign switch because Warp is float32 (see the parity script docstring); do not gate on it.
echo "-- informational: random actions with sampled friction (expected float32 divergence)"
"$PYTHON" scripts/experiments/cube_wrench/check_wrench_gpu_parity.py \
  --controller random --friction sampled --partner-mode oracle --device "$DEVICE" --steps 60 \
  || echo "-- (expected) the float32 Coulomb-friction switch diverges; see the docstring"

echo "== 2. throughput: CPU loop vs warp batch"
"$PYTHON" scripts/experiments/cube_wrench/bench_wrench_gpu.py \
  --nworld 256 1024 4096 --steps "$BENCH_STEPS" --device "$DEVICE"

echo "== 3. smoke training: backend=warp, 2 updates"
WRENCH_TORCH_THREADS=1 "$PYTHON" -m callosum.experiments.cube_wrench.train \
  --backend warp --device "$DEVICE" --partner-mode oracle \
  --num-envs "$SMOKE_ENVS" --rollout-steps "$SMOKE_ROLLOUT" --total-updates 2 \
  --eval-every 1 --eval-episodes 8 --eval-envs 8 --checkpoint-every 1 \
  --exp-name wrench_warp_smoke_oracle --runs-dir "$RUNS_DIR"

if [ "${RUN_FULL:-0}" != "1" ]; then
  echo "== done (set RUN_FULL=1 for the 300-update grid)"
  exit 0
fi

echo "== 4. full grid: backend=warp, ${UPDATES} updates x ${NUM_ENVS} envs (detached)"
mkdir -p "$RUNS_DIR"
for seed in $SEEDS; do
  for mode in $MODES; do
    name="wrench_warp_${mode}_s${seed}"
    partner_mode="$mode"
    bijepa=()
    if [ "$mode" = "predicted" ]; then
      partner_mode="none"
      bijepa=(--use-bijepa)
    fi
    echo "launch $name"
    WRENCH_TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      setsid nohup "$PYTHON" -m callosum.experiments.cube_wrench.train \
        --backend warp --device "$DEVICE" --partner-mode "$partner_mode" "${bijepa[@]}" \
        --seed "$seed" --exp-name "$name" --total-updates "$UPDATES" \
        --num-envs "$NUM_ENVS" --rollout-steps "$ROLLOUT_STEPS" \
        --eval-every 50 --eval-episodes 32 --eval-envs 32 \
        --runs-dir "$RUNS_DIR" </dev/null > "$RUNS_DIR/$name.log" 2>&1 &
  done
done

if [ "${WAIT_FULL:-0}" = "1" ]; then
  echo "waiting for warp trainers..."
  while pgrep -f 'callosum.experiments.cube_wrench.train.*--backend warp' >/dev/null 2>&1; do
    sleep 15
  done
  echo "all warp trainers finished"
  for seed in $SEEDS; do
    for mode in $MODES; do
      name="wrench_warp_${mode}_s${seed}"
      "$PYTHON" callosum/experiments/cube_wrench/eval.py \
        --checkpoint "$RUNS_DIR/$name/latest.pt" --input-mode "$mode" \
        --backend warp --device "$DEVICE" --num-envs 64 \
        --episodes 128 --seed 0 --out "results_${name}.json"
    done
  done
fi
