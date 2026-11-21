#!/usr/bin/env bash
# CubeWrench-v0 GPU experiment driver: randomisation-preset grid, predicted sweep, evals.
#
# Runs ON THE POD (mujoco_warp, A40). The optional Warp stack must be installed first (a plain
# `uv sync --frozen` removes it):
#   uv pip install --python /root/.callosum/venv/bin/python mujoco-warp==4.0.1
#
# Usage (from the repo checkout on the pod):
#   bash scripts/experiments/cube_wrench/run_wrench_gpu_grid.sh grid      # 12 runs: default/harsh x modes x seeds
#   bash scripts/experiments/cube_wrench/run_wrench_gpu_grid.sh sweep     # 4 predicted configs (aux_coef x history_len)
#   bash scripts/experiments/cube_wrench/run_wrench_gpu_grid.sh eval      # 128-episode greedy evals, 3 seeds per ckpt
#   bash scripts/experiments/cube_wrench/run_wrench_gpu_grid.sh summary   # markdown/JSON summary in results_gpu/
#
# Grid (`grid`, K=3 concurrent so the graph replays/PPO are not oversubscribed):
#   default preset: none/oracle/predicted at seeds 1,2  (seed 0 already exists as *_s0)
#   harsh   preset: none/oracle at seeds 0,1,2
# Each run: 300 updates x 1024 envs x rollout 64, eval every 50 updates, 32 eval episodes.
# Checkpoints/logs: runs/wrench_gpu_<preset>_<mode>_s<seed>/ and runs/<name>.log.
#
# Sweep: predicted with aux_coef in {0.03, 0.1} x history_len in {1, 2}, seed 0, 200 updates
# (1024 envs, rollout 64, eval every 50, 32 eval episodes), run 4 at a time.
#
# Eval: every checkpoint of the grid + sweep, greedy, warp backend, 128 episodes for three
# DIFFERENT held-out seed sets (bases 0 / 1_000_000 / 2_000_000; the evaluator adds its
# EVAL_SEED_OFFSET), JSON in results_gpu/<name>_evalseed<base>.json.
#
# Overrides: K, UPDATES, SWEEP_UPDATES, NUM_ENVS, ROLLOUT, EVAL_EPISODES, EVAL_SEEDS.
set -u

cd /root/callosum
export WRENCH_TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="${PYTHON:-/root/.callosum/venv/bin/python}"
K="${K:-3}"
UPDATES="${UPDATES:-300}"
SWEEP_UPDATES="${SWEEP_UPDATES:-200}"
NUM_ENVS="${NUM_ENVS:-1024}"
ROLLOUT="${ROLLOUT:-64}"
EVAL_EPISODES="${EVAL_EPISODES:-128}"
EVAL_SEEDS="${EVAL_SEEDS:-0 1000000 2000000}"
RESULTS="${RESULTS:-results_gpu}"

GRID_NAMES=(
  wrench_gpu_default_none_s1 wrench_gpu_default_none_s2
  wrench_gpu_default_oracle_s1 wrench_gpu_default_oracle_s2
  wrench_gpu_default_predicted_s1 wrench_gpu_default_predicted_s2
  wrench_gpu_harsh_none_s0 wrench_gpu_harsh_none_s1 wrench_gpu_harsh_none_s2
  wrench_gpu_harsh_oracle_s0 wrench_gpu_harsh_oracle_s1 wrench_gpu_harsh_oracle_s2
)
EXISTING_DEFAULT_S0=(
  wrench_gpu_default_none_s0 wrench_gpu_default_oracle_s0 wrench_gpu_default_predicted_s0
)
SWEEP_NAMES=(
  wrench_gpu_sweep_h1_aux003_s0 wrench_gpu_sweep_h1_aux010_s0
  wrench_gpu_sweep_h2_aux003_s0 wrench_gpu_sweep_h2_aux010_s0
)

# The optional Warp stack: reinstall if a previous `uv sync --frozen` removed it.
if ! "$PY" -c 'import mujoco_warp' >/dev/null 2>&1; then
  uv pip install --python "$PY" mujoco-warp==4.0.1
fi

mode_of() {  # $1 run name -> none|oracle|predicted (policy input mode)
  case "$1" in
    *sweep*|*predicted*) echo predicted ;;  # the sweep runs are all predicted/Bi-JEPA
    *oracle*) echo oracle ;;
    *) echo none ;;
  esac
}

train_one() {  # $1 preset, $2 mode, $3 seed, $4 updates
  local preset="$1" mode="$2" seed="$3" updates="$4"
  local name="wrench_gpu_${preset}_${mode}_s${seed}"
  local extra=() env_partner="$mode"
  if [ "$mode" = "predicted" ]; then env_partner="none"; extra=(--use-bijepa); fi
  echo "launch $name $(date -u +%H:%M:%S)"
  "$PY" -m callosum.experiments.cube_wrench.train \
    --backend warp --device cuda --randomization "$preset" \
    --partner-mode "$env_partner" "${extra[@]}" \
    --seed "$seed" --exp-name "$name" --total-updates "$updates" \
    --num-envs "$NUM_ENVS" --rollout-steps "$ROLLOUT" \
    --eval-every 50 --eval-episodes 32 --eval-envs 32 --runs-dir runs \
    > "runs/$name.log" 2>&1
  echo "done $name $(date -u +%H:%M:%S)"
}

sweep_one() {  # $1 history_len, $2 aux_coef tag (003/010)
  local history="$1" aux_tag="$2" aux
  if [ "$aux_tag" = "003" ]; then aux="0.03"; else aux="0.1"; fi
  local name="wrench_gpu_sweep_h${history}_aux${aux_tag}_s0"
  echo "launch $name (history_len=$history aux_coef=$aux) $(date -u +%H:%M:%S)"
  "$PY" -m callosum.experiments.cube_wrench.train \
    --backend warp --device cuda --randomization default \
    --partner-mode none --use-bijepa --history-len "$history" --aux-coef "$aux" \
    --seed 0 --exp-name "$name" --total-updates "$SWEEP_UPDATES" \
    --num-envs "$NUM_ENVS" --rollout-steps "$ROLLOUT" \
    --eval-every 50 --eval-episodes 32 --eval-envs 32 --runs-dir runs \
    > "runs/$name.log" 2>&1
  echo "done $name $(date -u +%H:%M:%S)"
}

eval_one() {  # $1 run name, $2 eval seed
  local name="$1" seed="$2"
  local mode out="$RESULTS/${name}_evalseed${seed}.json"
  if [ -f "$out" ]; then
    echo "skip $out (exists)"
    return 0
  fi
  mode="$(mode_of "$name")"
  "$PY" callosum/experiments/cube_wrench/eval.py \
    --checkpoint "runs/$name/latest.pt" --input-mode "$mode" \
    --backend warp --device cuda --num-envs "$EVAL_EPISODES" \
    --episodes "$EVAL_EPISODES" --seed "$seed" --out "$out" \
    > "runs/${name}_evalseed${seed}.log" 2>&1
  echo "eval $name seed $seed -> $out"
}

case "${1:-}" in
  grid)
    mkdir -p "$RESULTS"
    for spec in \
      "default none 1" "default none 2" \
      "default oracle 1" "default oracle 2" \
      "default predicted 1" "default predicted 2" \
      "harsh none 0" "harsh none 1" "harsh none 2" \
      "harsh oracle 0" "harsh oracle 1" "harsh oracle 2"; do
      set -- $spec
      while [ "$(jobs -rp | wc -l)" -ge "$K" ]; do wait -n || true; done
      train_one "$1" "$2" "$3" "$UPDATES" &
      sleep 2
    done
    wait
    echo "GRID_DONE $(date -u +%H:%M:%S)"
    ;;
  sweep)
    mkdir -p "$RESULTS"
    for spec in "1 003" "1 010" "2 003" "2 010"; do
      set -- $spec
      while [ "$(jobs -rp | wc -l)" -ge 4 ]; do wait -n || true; done
      sweep_one "$1" "$2" &
      sleep 2
    done
    wait
    echo "SWEEP_DONE $(date -u +%H:%M:%S)"
    ;;
  eval)
    mkdir -p "$RESULTS"
    names=("${GRID_NAMES[@]}" "${EXISTING_DEFAULT_S0[@]}" "${SWEEP_NAMES[@]}")
    for name in "${names[@]}"; do
      [ -f "runs/$name/latest.pt" ] || { echo "skip $name (no checkpoint)"; continue; }
      for seed in $EVAL_SEEDS; do
        while [ "$(jobs -rp | wc -l)" -ge "$K" ]; do wait -n || true; done
        eval_one "$name" "$seed" &
        sleep 1
      done
    done
    wait
    echo "EVAL_DONE $(date -u +%H:%M:%S)"
    ;;
  summary)
    "$PY" scripts/analysis/summarize_wrench_gpu.py --results "$RESULTS"
    ;;
  *)
    echo "usage: $0 grid|sweep|eval|summary" >&2
    exit 2
    ;;
esac
