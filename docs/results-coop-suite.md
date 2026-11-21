# Two-robot coop suite: три простых эксперимента (cart-pole, pendulum, toss&catch)

Статус: **в работе (2026-11-21)**. Дизайн задач зафиксирован; таблицы результатов и демо
заполняются после прогонов. С 2026-11-21 suite обучен и на **GPU-бэкенде** (`mujoco_warp`,
`CoopBatchGPU`), который на Linux+CUDA является дефолтом; первые прогоны на поде до этого шли
на CPU-петле (квота ~7.65 CPU-ядер) и заняли часы.

## Зачем эти три задачи

Reduced-куб `CubeWrench-v0` — главная линия предзащиты (см. `docs/results-wrench-cube.md`), но
он один: если координация не сработает, непонятно, свойство ли это куба или метода. Три простые
задачи дают **разные профили координации** при одном и том же протоколе (IPPO, 2 независимые
политики, `none`/`oracle`/`predicted` Bi-JEPA, приватная латентность/случайность):

| задача | env id | что делает пара | неаддитивность / зачем партнёр |
|---|---|---|---|
| cart-pole | `CartPoleCoop-v0` | двое толкают тележку с маятником в разные стороны, удерживают шест и приводят тележку в цель | односторонние толчки + задержка: одному не хватает управления, второй нужен для торможения/удержания |
| pendulum pump | `PendulumPump-v0` | двое раскачивают маятник до ±65° и удерживают 2 полупериода | энергия накачивается синфазно; односторонний толчок насыщается на 50–59° |
| toss & catch | `TossCatch-v0` | двое перекидывают мяч ракетками (3 передачи подряд) | тайминг удара/приёма, скрытый импульс броска; нужна антиципация партнёра |

Все три — reduced MuJoCo (без рук). Бэкенды: CPU-петля `vec_env.VecCoopEnv` (parity/CI/мак) и
батчевый `mujoco_warp` `env_gpu.CoopBatchGPU` (GPU, дефолт на Linux+CUDA).

## Дизайн

- **`callosum/experiments/coop_tasks/base.py`**: `CoopTaskEnv` — Dict obs/action, MJCF, control
  substeps (`dt=0.005`, `control_every=8`), приватная латентность команд, start delay, односторонняя
  активация (для cart-pole/pendulum), общая team-reward, auto-reset; хуки `_sample_randomization`/
  `_on_substep` для задач.
- **`logic.py`**: чистая логика (obs-layout, visibility `own`+shared, success/failure/potential,
  рандомизация, сэмплеры) — тестируется на маке без mujoco.
- **`config.py`**: `CoopRewardConfig`, `CoopRandomization`, `CartPoleConfig`, `PendulumConfig`,
  `TossConfig`, `TASKS`, fail-fast инварианты (success/failure/tolerance ordering).
- **`oracle.py`**: scripted верхние границы — cart-pole: централизованный LQR + Smith-predictor
  задержки; pendulum: энергетическая накачка с компенсацией латентности; toss: баллистический
  перехват + оценка restitution по отскокам.
- **`vec_env.py`**: CPU-векторизация в стиле `_vec_wrench` (auto-reset, диагностика).
- **`env_gpu.py`**: `CoopBatchGPU` — батчевый `mujoco_warp`-бэкенд всех трёх задач (см. раздел
  «GPU-бэкенд» ниже): один Warp `Model`/`Data` на `nworld` миров, CUDA graph на control step,
  per-world `dof_damping`/`geom_solref`/`geom_friction`, для toss — per-substep интерполяция
  MOCAP-ракеток и детекция контактов; правила/наблюдения/награда из `logic.py`.
- **Тренер**: `python -m callosum.experiments.cube_wrench.train --task {cartpole,pendulum,toss}`
  (`none`/`oracle` + `--use-bijepa` → `predicted`); eval:
  `python -m callosum.experiments.coop_tasks.eval --checkpoint ... --input-mode ...`;
  демо: `python -m callosum.experiments.coop_tasks.render --task ... --scene ...`.

## Протокол обучения и eval

- IPPO: 2 независимые гауссовы политики, `num_envs=64`, `rollout_steps=64`, GAE, clip.
- Бюджеты: cart-pole 400 → **1200** апдейтов (400 не хватало: ~0.44 success); pendulum 400;
  toss 400 (contact-rich, ~530 sps/процесс).
- Сиды обучения: 0,1,2; режимы: `none`, `oracle`, `predicted` (Bi-JEPA).
- Held-out eval: **128 эпизодов × 3 набора сидов** (`--seed {0, 1_000_000, 2_000_000}`,
  `EVAL_SEED_OFFSET=10_000_019` — вне обучающих сидов), greedy-политики.
- Железо: RunPod A40. GPU-прогоны — `--backend warp` (дефолт для suite на CUDA), CPU-петля
  использовалась для первых прогонов (квота ~7.65 CPU → параллелизм ≤7 процессов).

## Результаты

_Заполняется после завершения прогонов: success_rate по задачам/режимам/сидам, `oracle−none`,
`predicted−none`, время до успеха._

## GPU-бэкенд (2026-11-21)

`callosum/experiments/coop_tasks/env_gpu.py` (`CoopBatchGPU`) — батчевый `mujoco_warp`-бэкенд всех
трёх задач; оформлен как vec-env контракт `VecCoopEnv` (`torch_io=True`, `reset/step`,
`finished_*`, `final_obs`), чекпоинты не зависят от бэкенда.

- **Что на GPU:** latency-очереди и start-delay, односторонняя активация, `xfrc_applied`,
  8 физических substeps, наблюдения (раскладка из `logic.obs_fields`), team-reward,
  success/failure/potential-shaping, авто-reset с тем же seed-потоком, что у CPU.
  Per-world случайность: `dof_damping` (cartpole/pendulum), `geom_solref`/`geom_friction` (toss).
  Контакты toss: `mocap_pos`/`mocap_quat` ракеток интерполируются по substeps из pose/quat-банка,
  контакты мяч-ракетка детектируются warp-ядрами per-substep, счётчик передач — CPU-семантика.
- **CUDA graph** на весь control step (включая per-substep ядра toss); fallback — eager loop.
- **Дефолт:** `--backend` = `warp` для `--task cartpole|pendulum|toss` на Linux+CUDA
  (`ppo_config.default_backend`); cube по-прежнему CPU; без CUDA — warning и CPU-петля.
  `coop_tasks/eval.py --backend warp --num-envs N` — батчевый held-out eval.
- **Parity (8/8 PASS):** cartpole/pendulum (zero/oracle) qpos ≤ 3e-7…1.6e-6, obs ≤ 8e-7…4e-6,
  reward ≤ 7e-7, finished Δreturn ≤ 3e-7, 0 terminal-flag mismatches; cartpole oracle — порог
  мягче (неустойчивый объект, float32-возмущение растёт до 1.8e-4 за 120 шагов); toss
  zero/oracle — qpos ≤ 2.2e-6/1.8e-3, obs ≤ 3e-6/2.4e-3 (float32-контакты). Детерминизм двух
  одинаковых инстансов — bit-identical (`tests/experiments/coop_tasks/test_coop_gpu.py`, 28 тестов).
- **Throughput (A40, control steps/s, warp vs CPU-петля):**

| задача | 256 миров | 1024 мира | 4096 миров |
|---|---|---|---|
| cartpole | 52k vs 9.9k (**5×**) | 179k vs 10.9k (**16×**) | 449k vs 10.9k (**41×**) |
| pendulum | 104k vs 14.3k (**7×**) | 400k vs 13.8k (**29×**) | 1412k vs 13.2k (**107×**) |
| toss | 37k vs 662 (**56×**) | 117k vs 649 (**181×**) | 276k vs 681 (**405×**) |

- **Обучение (2 апдейта × 1024 env × 64 rollout):** cartpole 1.3–1.5 с/апдейт (49–53k sps),
  pendulum 1.2–2.9 с (55–61k), toss 4.9–5.3 с (12–14k). Оценка 1200 апдейтов: cartpole ~28 мин,
  pendulum ~41 мин, toss ~100 мин. GPU-eval на тех же чекпоинтах даёт **ровно те же метрики**,
  что CPU-eval (cartpole 0/16, 45.19 шагов; toss 0.188 success, 19.94 шагов).
- **Ограничение:** float32 контактный солвер warp редко даёт non-finite состояние на глубоком
  быстром ударе мяча о ракетку; env помечает такой мир FAILED-эпизодом и авто-ресетит
  (`physics_failures`), NaN в политику не попадает. Контактные траектории toss чувствительны к
  float32 → порог parity для toss мягче, чем у contact-free задач.

Команды (на поде: `/root/.callosum/venv/bin/python`):

```bash
python -m callosum.experiments.cube_wrench.train --task cartpole --num-envs 1024 \
    --rollout-steps 64 --total-updates 400 --exp-name coop_cartpole_gpu_s0
python -m callosum.experiments.coop_tasks.eval --checkpoint runs/coop_cartpole_gpu_s0/latest.pt \
    --input-mode none --episodes 128 --backend warp --num-envs 64 --out results.json
python scripts/experiments/coop_tasks/check_coop_gpu_parity.py --task toss --controller oracle --steps 30
python scripts/experiments/coop_tasks/bench_coop_gpu.py --nworld 256 1024 4096
```

## Демо

_Заполняется: `demo/demo_{cartpole,pendulum,toss}_final_*` (policy vs oracle, attempts), GIF-превью._

## Ограничения / честные оговорки

- Cart-pole: чистый PD-oracle неустойчив при латентности 0.1–0.3 с → LQR+Smith; чистый RL без
  привилегий сходится медленнее (нужно 1200 апдейтов).
- Pendulum: target 65° (а не 60°) и damping 0.11–0.15 выбраны, чтобы `only-left` стабильно
  не добирал амплитуду, а oracle добирал.
- Toss: ракетки kinematic MOCAP (position-controlled, интерполяция по substeps), латентность
  0.04–0.16 с; oracle ~0.9 (остаточные провалы — низкий третий перехват); `predicted` для toss
  проверен только по ширинам obs.
- Ранние числа (первые прогоны) — с CPU-бэкенда и одной машины под квотой ~7.65 CPU; после
  2026-11-21 все прогоны и eval — на GPU (`--backend warp`), CPU-петля оставлена для parity/CI.
