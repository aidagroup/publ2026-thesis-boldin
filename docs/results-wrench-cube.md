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
- **G2/G3 (честный протокол: один эпизод на фиксированный held-out seed, 128 эпизодов × 3 набора сидов):**

| preset | режим | success (s0 / s1 / s2) | mean |
| --- | --- | --- | --- |
| default | none | 1.000 / 1.000 / 1.000 | **1.000** |
| default | oracle | 0.990 / 0.992 / 0.721 | 0.901 |
| default | predicted (Bi-JEPA) | 0.964 / 0.883 / 0.948 | 0.932 |
| harsh | none | 0.977 / 0.810 / 0.729 | 0.839 |
| harsh | oracle | 0.971 / 0.966 / 0.969 | **0.969** |

  - **default:** асимптотического разрыва нет — `none` решает задачу идеально; `oracle` в среднем
    даже ниже из-за «хрупкого» seed 2 (0.721). Привилегированный партнёр не нужен, когда данных много.
  - **harsh** (приватная случайность сильнее): `oracle` устойчив (0.966–0.971 на всех сидах), `none`
    по-разному (0.729–0.977) → **+0.130 в среднем** (на 2 из 3 сидов положительный, на одном ≈0).
    Механизм — **устойчивость oracle против seed-хрупкости none**, а не равномерный разрыв.
  - **Sample efficiency (честные `train/success`):** oracle достигает 0.5/0.8/0.9 на апдейтах
    166/246/272, none — 198/262/288 (default); на harsh порог 0.5 — 171–186 против 201–216.
    То есть знание партнёра ускоряет обучение, но не поднимает потолок.
  - **Bi-JEPA `predicted`:** в текущей реализации **не обгоняет `none`** (default mean 0.932 vs 1.000;
    sweep `aux_coef {0.03,0.1} × history_len {1,2}` — лучший 0.716 на 200 апдейтах, u150 0.19 vs
    none 0.30). Это **отрицательный промежуточный результат**; утверждение «Bi-JEPA закрывает разрыв»
    в текст диплома пока не идёт. Кривые: `demo/learning_curves_honest_seed0.png`,
    `demo/learning_curves_harsh.png`.

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
/root/.callosum/venv/bin/python -m callosum.experiments.cube_wrench.train --partner-mode oracle \
  --seed 0 --exp-name wrench_oracle_s0 --total-updates 300 --num-envs 96
# GPU-обучение (1024 env):
/root/.callosum/venv/bin/python -m callosum.experiments.cube_wrench.train --backend warp --device cuda \
  --partner-mode oracle --seed 0 --exp-name wrench_warp_oracle_s0 \
  --total-updates 300 --num-envs 1024 --rollout-steps 64
# GPU-обучение с Bi-JEPA:
... --backend warp --partner-mode none --use-bijepa --exp-name wrench_warp_pred_s0 ...
# eval:
/root/.callosum/venv/bin/python callosum/experiments/cube_wrench/eval.py \
  --checkpoint runs/<run>/latest.pt --input-mode none|oracle|predicted \
  --episodes 128 --seed 0 --out results/<run>.json [--backend warp]
# демо:
MUJOCO_GL=egl /root/.callosum/venv/bin/python callosum/experiments/cube_wrench/render.py \
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

1. **Eval-протокол исправлен** (held-out seed, один эпизод на seed, сбор только первых эпизодов);
   все числа выше перемерены; `check_wrench_gpu_determinism.py` — bit-identical повторяемость.
2. **Асимптотика:** `none` догоняет `oracle` — реактивная координация по общему состоянию куба
   достаточна. Заявляемый вклад привилегированной информации — **скорость обучения и устойчивость
   к приватной случайности** (harsh), а не асимптотический потолок.
3. **Bi-JEPA** в текущем виде проигрывает `none`; чтобы получить положительный вклад, нужны:
   aux-warmup (сначала чистый PPO, потом включить предсказание), предобучение энкодера/предиктора
   на данных партнёра (self-supervised), horizon>1 и явная цель «будущее действие партнёра»,
   сравнение с `predicted-direct`/реконструктивным baseline; возможно, применять Bi-JEPA только
   в harsh/occluded-режиме, где обратная связь запаздывает. Это план после предзащиты.
4. **Harsh-рандомизация** (slip 0.2–0.5, start delay 0–0.5) — сетка на GPU запущена, цель: проверить,
   появляется ли асимптотический разрыв.
5. `mujoco-warp` не входит в обычный `uv sync` (linux-marked extra); команды выше это учитывают.
   Версия запинена в extra/доках.
6. Сравнение с реконструктивным/дистилляционным teammate-моделированием (Dreamer/CLS-DP) и
   RoboFactory — весна, после предзащиты; prior-art позиционирование обновлено в `docs/thesis/02`.
