#!/usr/bin/env bash
# Run the CubeWrench-v0 experiment grid (the reduced-model ablation of the thesis).
#
# The grid: for each seed, train two independent policies in the `oracle` (privileged
# partner state) and `none` (own observation only) modes, plus the `predicted` Bi-JEPA
# mode (env opt-mode `none`, policy input = own observation | predicted partner latent).
# Then evaluate every checkpoint greedily on a fixed set of episodes.
#
# Everything is CPU; the only acceleration is running the independent seeds as separate
# single-threaded processes (MuJoCo's Python loop is GIL-bound, and torch's default
# thread pool oversubscribes on a many-core pod -> each run exports one torch thread).
#
# Usage (on the RunPod pod, from the repo checkout):
#   bash scripts/experiments/cube_wrench/run_wrench_experiments.sh                  # 3 seeds, 500 updates
#   SEEDS="0" UPDATES=100 bash scripts/experiments/cube_wrench/run_wrench_experiments.sh
#   MODES="none oracle" SEEDS="0 1" bash scripts/experiments/cube_wrench/run_wrench_experiments.sh
#
# The training logs land in $RUNS_DIR/<exp-name>.log; the eval JSONs in
# $RESULTS_DIR/<exp-name>.json. Override RUNS_DIR to point at the network volume.
set -euo pipefail

RUNS_DIR="${RUNS_DIR:-runs}"
RESULTS_DIR="${RESULTS_DIR:-results}"
UPDATES="${UPDATES:-500}"
SEEDS="${SEEDS:-0 1 2}"
MODES="${MODES:-oracle none predicted}"
NUM_ENVS="${NUM_ENVS:-96}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-64}"
EVAL_EPISODES="${EVAL_EPISODES:-128}"
PYTHON="${PYTHON:-python}"

mkdir -p "$RUNS_DIR" "$RESULTS_DIR"

run_one() {  # $1 mode, $2 seed -> launch one detached trainer
  local mode="$1" seed="$2" name="wrench_${mode}_s${seed}"
  local partner_mode="$mode" bijepa=()
  if [ "$mode" = "predicted" ]; then
    partner_mode="none"
    bijepa=(--use-bijepa)
  fi
  echo "launch $name (updates=$UPDATES, envs=$NUM_ENVS)"
  WRENCH_TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    setsid nohup "$PYTHON" -m callosum.experiments.cube_wrench.train \
      --partner-mode "$partner_mode" "${bijepa[@]}" \
      --seed "$seed" --exp-name "$name" \
      --total-updates "$UPDATES" --num-envs "$NUM_ENVS" \
      --rollout-steps "$ROLLOUT_STEPS" --eval-every 50 --eval-episodes 32 \
      --runs-dir "$RUNS_DIR" </dev/null > "$RUNS_DIR/$name.log" 2>&1 &
}

for seed in $SEEDS; do
  for mode in $MODES; do
    run_one "$mode" "$seed"
  done
done

# Wait for every launched trainer before evaluating (the script is a foreground driver).
echo "waiting for trainers..."
while pgrep -f 'callosum.experiments.cube_wrench.train' >/dev/null 2>&1; do
  sleep 15
done
echo "all trainers finished"

for seed in $SEEDS; do
  for mode in $MODES; do
    name="wrench_${mode}_s${seed}"
    "$PYTHON" callosum/experiments/cube_wrench/eval.py \
      --checkpoint "$RUNS_DIR/$name/latest.pt" \
      --input-mode "$mode" --episodes "$EVAL_EPISODES" --seed 0 \
      --out "$RESULTS_DIR/$name.json"
  done
done

echo "done: results in $RESULTS_DIR"
