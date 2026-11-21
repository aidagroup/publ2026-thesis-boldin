# CubeWrench-v0 — итоги (reduced-модель, 2026-11-21)

Промежуточный отчёт по главной линии предзащиты: изолированный куб + прикладываемые моменты,
два децентрализованных агента, Bi-JEPA (латентное предсказание партнёра). Полный план и гейты —
`PLAN.md`; спецификации — `openspec/changes/add-wrench-cube-reduced-model/`; теория и протокол —
`docs/thesis/03–04`.

## Что построено

- **Среда `CubeWrench-v0` (MuJoCo, CPU):** core (свободный куб 5.7 см) + активный слой на hinge
  `[0,90°]`, без рук/гравитации; два агента прикладывают непрерывные 6D-моменты (holder стабилизирует
  ядро, rotator крутит слой). Приватная случайность: латентность 0.1–0.3 с, gain/limit, slip (1–3
  решения, p≤0.3), start delay, импульс, трение слоя, начальный угол. Неаддитивность гарантирована
  инвариантом конфига и smoke-контроллером `rotator-only`.
- **Среда `CubeWrench-v0` (GPU, `mujoco_warp`):** те же семантика и раскладка наблюдений, батч
  1024–4096 миров, CUDA graph для 8 физ-подшагов; per-world friction/force; автоматический сброс.
  Parity GPU↔CPU по траекториям: qpos 1.5e-6, obs 1.4e-6 (документированное расхождение ~1e-4 на
  смене знака скорости при sampled-трении — float32 Warp).
- **Тренер:** непрерывный IPPO на двух независимых гауссовых политиках (`_ppo_core`), GAE, чекпоинты,
  resume, TensorBoard; `--backend cpu|warp --device cuda`, веса переносимы между бэкендами.
- **Bi-JEPA:** модуль из `step/3.1-bijepa-module` (энкодер 26→32, предиктор латента партнёра,
  EMA-target, VICReg) интегрирован как aux-лосс + вход `ẑ_j` политики (вход 28+32=60); на исполнении
  данные партнёра не используются; лог `aux/probe/*` (ridge-проба `ẑ_j → wrench партнёра`).
- **Eval-harness:** greedy-оценка success/angle/drift/steps; CPU и batched GPU; протокол
  «один эпизод на фиксированный held-out seed» (в процессе исправления — см. ниже).
- **Демо-рендер:** MuJoCo EGL на поде, три сцены: oracle success, rotator-only failure («момент
  помощи»), learned policy vs oracle; mp4/GIF/stills.
- **Тесты/линт:** `pytest -q` → 513 passed, 2 skipped (CPU; GPU-тесты скипаются без warp);
  `ruff check` чистый; на поде `tests/test_wrench_gpu.py` → 12 passed (warp+CUDA).

## Проверенные результаты

- **G1 (среда валидна):** скриптовый oracle — **0.988 успеха** на 256 эпизодах; `rotator-only` —
  **0.000** (дрейф ядра 29.8° > порога 28.6°) → неаддитивность работает. Повторено на финальной
  версии среды.
- **GPU-ускорение:** env-throughput 10.8×/26.3×/53.5× на 256/1024/4096 мирах; обучение oracle
  300 апдейтов × 1024 env — 543 с (vs ~2.8k sps на CPU-цикле). Параллельно на поде шла сетка
  сидов/пресетов.
- **Демо обученных политик (визуально проверено):** learned policy (`none`) доходит до SUCCESS за
  ~30 control steps (91.0°, drift 3.2 мм/4.2°), oracle — за 55 шагов (87.1°); predicted (Bi-JEPA) —
  SUCCESS за 45 шагов (90.1°). Эпизоды реальные, HUD с углом/дрейфом/шагами.
- **G2/G3 (разрыв oracle−none, вклад Bi-JEPA):** предварительные GPU-числа **требуют пере-мера**
  после исправления протокола eval (ревью нашло bias: batched-eval собирал быстрых финишёров вместо
  фиксированных held-out сидов). После фикса будут даны success по 3 сидам для `none/oracle/predicted`
  и честные кривые; ожидаемая формулировка — разрыв в **sample efficiency** на раннем бюджете, а не
  в асимптотическом потолке. Черновые кривые seed0 (до фикса) сохранены в `demo/learning_curves_s0.png`
  как иллюстрация метода, но в текст диплома не идут.

## Как запускать

```bash
# CPU-среда (macOS/Linux), локально только 1-2 env:
./.venv/bin/python scripts/smoke_wrench_cube.py --envs 2 --episodes 10 --controller oracle
./.venv/bin/python scripts/smoke_wrench_cube.py --envs 2 --episodes 10 --controller rotator-only
./.venv/bin/python scripts/smoke_wrench_cube.py --envs 2 --episodes 10 --controller random

# тесты/линт:
./.venv/bin/python -m pytest -q && ./.venv/bin/ruff check .

# Pod (RunPod A40): сначала warp-стек (uv sync --frozen его удаляет):
cd ~/callosum && git fetch && git checkout -B step/2.4-wrench-cube origin/step/2.4-wrench-cube
env UV_PROJECT_ENVIRONMENT=/root/.callosum/venv uv sync --frozen --extra train --extra dev \
  --extra sandbox --extra sandbox-gpu
# CPU-обучение (fallback):
/root/.callosum/venv/bin/python -m callosum.training.wrench_ppo --partner-mode oracle \
  --seed 0 --exp-name wrench_oracle_s0 --total-updates 300 --num-envs 96
# GPU-обучение (1024 env):
/root/.callosum/venv/bin/python -m callosum.training.wrench_ppo --backend warp --device cuda \
  --partner-mode oracle --seed 0 --exp-name wrench_warp_oracle_s0 \
  --total-updates 300 --num-envs 1024 --rollout-steps 64
# GPU-обучение с Bi-JEPA:
... --backend warp --partner-mode none --use-bijepa --exp-name wrench_warp_pred_s0 ...
# eval:
/root/.callosum/venv/bin/python scripts/eval_wrench_cube.py \
  --checkpoint runs/<run>/latest.pt --input-mode none|oracle|predicted \
  --episodes 128 --seed 0 --out results/<run>.json [--backend warp]
# демо:
MUJOCO_GL=egl /root/.callosum/venv/bin/python scripts/render_wrench_demo.py \
  --out /workspace/callosum-runs/render --scene policy-vs-oracle \
  --checkpoint runs/wrench_warp_none_s0/latest.pt --input-mode none
```

## Артефакты демо (`demo/`, локально; на поде — `/workspace/callosum-runs/render*`)

- `demo_policy_vs_oracle.mp4` + `demo_preview.gif` — **главная демка**: learned policy (`none`) vs
  oracle, обе стороны SUCCESS.
- `demo_policy_vs_oracle_predicted.mp4` — Bi-JEPA (`predicted`) vs oracle.
- `demo_oracle.mp4`, `demo_rotator_only.mp4`, `demo_side_by_side.mp4` — скриптовые сцены:
  успех вдвоём vs провал rotator-only («момент помощи»).
- `policy_vs_oracle_final.png`, `oracle_final.png`, `rotator_only_final.png` — стиллы;
  `demo_policy_summary.json` — метрики эпизодов/ffprobe.

## Известные ограничения / следующие шаги

1. **Eval-протокол** (BLOCKER ревью) — в работе: один эпизод на held-out seed, сбор только первых
   эпизодов, исключение тренировочных сидов; после фикса — пере-мер всех чекпоинтов и финальные
   числа G2/G3.
2. **Асимптотика:** при достаточном бюджете `none`, судя по всему, догоняет `oracle` (реактивная
   координация по общему состоянию куба достаточна); тогда заявляемый вклад Bi-JEPA — ускорение
   обучения (sample efficiency), а не потолок. Это будет сформулировано честно.
3. **Bi-JEPA tuning** (`aux_coef`, `history_len`, latent) — в работе; predicted должен обгонять
   `none` на раннем бюджете.
4. **Harsh-рандомизация** (slip 0.2–0.5, start delay 0–0.5) — сетка на GPU запущена, цель: проверить,
   появляется ли асимптотический разрыв.
5. `mujoco-warp` не входит в обычный `uv sync` (linux-marked extra); команды выше это учитывают.
   Версия запинена в extra/доках.
6. Сравнение с реконструктивным/дистилляционным teammate-моделированием (Dreamer/CLS-DP) и
   RoboFactory — весна, после предзащиты; prior-art позиционирование обновлено в `docs/thesis/02`.
