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

## Протокол обучения и eval (GPU-first)

- **GPU-first:** все финальные прогоны — `--backend warp` (`mujoco_warp`+CUDA graph, `CoopBatchGPU`);
  CPU-петля `VecCoopEnv` остаётся для parity, unit/CI и 1–2 env smoke. Причина: на поде квота
  ~7.65 CPU-ядер, CPU-обучение трёх задач заняло бы десятки часов.
- IPPO: 2 независимые гауссовы политики, `num_envs=256`, `rollout_steps=64`, GAE, clip.
- Бюджеты (GPU, 256 envs — эквивалент CPU-финала по числу сэмплов): cart-pole **400** апдейтов,
  pendulum **200**, toss **250**. (CPU-пилот: cart-pole 1200, pendulum 400, toss 400 при 64 envs.)
- Сиды обучения: 0,1,2; режимы: `none`, `oracle`, `predicted` (Bi-JEPA `--use-bijepa`).
- Held-out eval: **128 эпизодов × 3 набора сидов** (`--seed {0, 1_000_000, 2_000_000}`,
  `EVAL_SEED_OFFSET=10_000_019` — вне обучающих сидов), greedy-политики; batched GPU-eval
  (`--backend warp --num-envs 128`), то же JSON-схема, что у CPU-eval.
- Параллелизм: 2 GPU-прогона одновременно (6 процессов душили GPU: суммарно ~29k sps против
  ~50k у одного); на поде GPU A40, 46 GB.

## Результаты (GPU, 3 сида × 3 held-out базы, 128 эпизодов)

Все числа — batched GPU-eval (`--backend warp`), greedy; среднее по 3 обучающим сидам
(0,1,2), у каждого — 3 набора held-out сидов (`0 / 1M / 2M` + `EVAL_SEED_OFFSET`).

| задача | режим | success (mean) | по сидам | min–max | доля эпизодов |
|---|---|---|---|---|---|
| cart-pole (400 upd) | none | **0.957** | 0.958 / 0.956 / 0.956 | 0.956–0.958 | 9×128 |
| cart-pole | oracle | **0.909** | 0.943 / 0.823 / 0.961 | 0.823–0.961 | 9×128 |
| cart-pole | predicted | **0.459** | 0.599 / 0.539 / 0.240 | 0.240–0.599 | 9×128 |
| pendulum (200 upd) | none | **0.974** | 0.974 / 0.974 / 0.974 | 0.974–0.974 | 9×128 |
| pendulum | oracle | **0.975** | 0.977 / 0.974 / 0.974 | 0.974–0.977 | 9×128 |
| pendulum | predicted | **0.976** | 0.974 / 0.979 / 0.974 | 0.974–0.979 | 9×128 |
| toss (250 upd, fixed rule) | none | **0.966** | 0.964 / 0.964 / 0.971 | 0.964–0.971 | 9×128 |
| toss | oracle | **0.923** | 0.974 / 0.872 / 0.922 | 0.872–0.974 | 9×128 |
| toss | predicted | **0.939** | 0.927 / 0.943 / 0.948 | 0.927–0.948 | 9×128 |

**Gaps:** cart-pole oracle−none = −0.048, predicted−none = −0.497; pendulum ≈ 0 (+0.001/+0.002);
toss oracle−none = −0.043, predicted−none = −0.027.

Интерпретация:
- **Все три задачи обучаются** (none ≈ 0.96–0.97); scripted oracle — sanity/верхняя граница, но
  не оптимален: на cart-pole и toss обученная пара его **обгоняет** (oracle писался вручную и
  централизованно, а не оптимизировался).
- **Bi-JEPA `predicted` не даёт выигрыша на простых задачах** (cart-pole хуже, pendulum — в шуме
  ±0.002, toss — чуть хуже). Это согласуется с гипотезой «реактивной достаточности»: в этих
  задачах партнёр наблюдаем через общий объект/мяч, латентная антиципация не нужна, а aux-лосс
  тратит ёмкость/шаги. Положительный эффект latent partner prediction ожидается там, где
  информация о партнёре скрыта/зашумлена (harsh-режим куба, occluded-постановки) — план проверки
  в отчёте по кубу.
- Подробности по сидам и JSON — `results_coop_gpu/summary.json` (артефакты eval'ов:
  `results_coop_gpu/gpu_coop_<task>_<mode>_s<seed>_base<base>.json`).

### Исправление exploit'а toss (2026-11-21)

Первая GPU-политика toss «проходила» 3/3 за 9–14 шагов, засчитывая дребезг контактов у пола.
Правило pass ужесточено (CPU и GPU): передача = контакт на половине **принимающей** ракетки
(мяч перелетел центр), минимальный интервал 3 control steps; success требует мяч выше
`floor_z + 0.02`; **failure приоритетнее success**. После фикса успех ≥28 шагов, мяч на success
0.090–0.172 м; pre-fix числа (`results_coop_gpu/exploit_pre_fix/`: none 0.753 за ~11 шагов)
оставлены как пример, почему гейт важен.

### Физика ракеток toss: реальная динамика вместо MOCAP (2026-11-21, локально)

Требование пользователя: ракетки **не должны пересекаться/проходить друг через друга**, физика —
максимально реальная. MOCAP-боксы (kinematic, env писал `mocap_pos/quat` и интерполировал) заменены
на **динамические актуируемые тела**:

- `assets/toss_catch.xml`: каждая плита 0.6×0.6×0.06 м, 0.8 кг — rigid body на двух
  призматических суставах (x, z) + hinge (tilt вокруг +y), на неподвижном основании; 6
  `<position>`-сервоприводов с ограниченными силами (`kp=8000/1500`, `kv=160/17`,
  `forcerange=±800 Н / ±150 Н·м`). MOCAP удалён (`nmocap=0`).
- Контакты: `ball↔paddle` — прежний soft-контакт с рандомизацией restitution/friction на
  reset; `paddle↔paddle` — отдельные жёсткие collision-shells (`contype/conaffinity`), поэтому
  плиты физически не могут пройти друг через друга. Цели серво дополнительно проецируются
  apart (`paddle_model.project_targets_apart`), но решает всё равно контакт.
- `toss_catch.py`/`env_gpu.py`: команда `(dx,dz,dtheta)` двигает **цель** PD-серво (target
  накапливается от предыдущей цели, поэтому серво держит плиту против гравитации — саг ~1 мм;
  суставы не телепортируются). Pass-правило (partner-half + min interval 4 + floor margin +
  failure priority) и успех/награда не менялись. Готовые позы: `pad_start_z=0.22`,
  `|x_ready| = pad_inner_x + 0.22 = 0.27` м (зазор плит в покое 6.7 см — старые 0.6-м плиты на
  MOCAP пересекались на ~17 см, для жёстких тел это невозможно).
- `oracle.py`: планирование в target-space серво, `reach_margin 0.02`, `catch_clearance 0.08`,
  антиципация включается по `|vx|==0` (иначе цель flip-flop'ила и физическая плита махала
  сквозь перехват).
- Диагностика: `pad_min_gap` (SAT-lower-bound расстояния между плитами), `pad_penetration`
  (SAT-глубина перекрытия, 0 = нет), `pad_contact_steps` — в info и в JSON eval
  (`max_pad_penetration`, `episodes_with_pad_contact`, `min_pad_gap`).

**Локальные проверки (CPU MuJoCo, мак, 256 эпизодов):** oracle **0.891** (mean steps 33.9,
min success ball_z 0.09 м), only-left **0.012**, `max pad_penetration == 0.0` и **0 физических
контактов плит** за 256 oracle-эпизодов (min SAT-gap 1.4 мм); на 100 случайных роллаутах —
4 эпизода с кратковременным контактом плит (перекрытие на шаге 0, контакт решается
солвером). Отдельный тест намеренного full-speed рама: плиты не проходят друг через друга,
первый удар даёт транзиент ≤ одного шага сближения (≤0.1 м), зажатое состояние — без
перекрытия; штатный clip целей такой рам не допускает. `pytest`/`ruff` чисты.

**Старые числа toss в таблицах выше получены на MOCAP-ракетках и к новой физике не относятся:**
все toss-чекпоинты нужно переобучить (3 режима × 3 сида × 250 апдейтов) и переоценить
(128 эп. × 3 held-out базы) на поде. Команды — см. следующий раздел (плюс
`check_coop_gpu_parity.py --task toss --controller zero|oracle`).

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

GPU-финальные клипы (policy-vs-oracle + attempts) — в `demo/`:
`demo_{cartpole,pendulum,toss}_gpu_policy_vs_oracle.mp4`, `..._policy_vs_oracle_predicted.mp4`,
`..._gpu_attempts_seed{0,1,2}.mp4` + GIF/PNG/JSON. Для toss используй
`demo_toss_gpu_fixed_*` (после фикса): `demo_toss_gpu_fixed_policy_vs_oracle.mp4` (none, SUCCESS 53),
`..._predicted.mp4` (SUCCESS 59), `..._attempts_seed{0,1,2}.mp4` (3/3 SUCCESS, 45–86 шагов).
JSON содержит `gpu_provenance`: политика обучена на GPU, физика кадров — CPU MuJoCo после parity
1e-6 obs / 1e-3 rollout; рендер EGL на A40. Старые `demo_toss_gpu_*` — пример exploit'а, не
использовать в отчёте. CPU-пилотные клипы `demo_*_final_*` — fallback.

## Реальная динамика ракеток и исправление критического GPU-бага (2026-11-22)

По требованию «ракетки не должны пересекаться, физика максимально реальная» MOCAP-планки
заменены на **динамические тела**: плиты 0.6×0.6×0.06 м, 0.8 кг, 2 призматических сустава (x,z)
+ hinge (tilt), bounded-force сервоприводы; включены контакты **ракетка↔ракетка** (жёсткие
collision shells) и ракетка↔мяч. CPU и GPU (`mujoco_warp`) пути дают parity; oracle на CPU
0.891 (256 эп.), `max_pad_penetration=0`, форс-ram тест без проскока.

**Критический баг GPU-физики (исправлен, коммит `9c80ec6`):** в `env_gpu.py` сырые флаги
контакта мяч↔ракетка (`_flags_warp`) не обнулялись между control steps → `contact_side` залипал,
rising-edge в pass-гейте срабатывал максимум 2 раза за эпизод → **success был физически
недостижим во всех warp-прогонах** (именно поэтому «RL не обучается», 0.0 на 250 апдейтах).
Фикс: `_flags.zero_()` раз в control step (+ `GpuDemoEnv.state()` теперь отдаёт servo
targets/contact). После фикса: parity zero/oracle PASS (qpos 1.9e-4/1.4e-3, 0 флаговых
расхождений), scripted oracle на GPU 0.86, RL учится сразу.

### Итог по toss на реальной физике (GPU, 3 сида × 3 held-out базы × 128 эп.)

| режим | success (mean) | по сидам | min–max |
|---|---|---|---|
| none | **0.955** | 0.938 / 0.964 / 0.964 | 0.938–0.964 |
| oracle | **0.950** | 0.969 / 0.971 / 0.909 | 0.909–0.971 |
| predicted (Bi-JEPA) | **0.950** | 0.961 / 0.909 / 0.979 | 0.909–0.979 |

Gaps: oracle−none ≈ −0.005, predicted−none ≈ −0.005. Все успехи ≥ ~28 control steps, мяч на
success выше пола; `max_pad_penetration=0`, ракетки не пересекаются. Вывод прежний: на этой
задаче Bi-JEPA не нужна (партнёр виден через мяч/траекторию) — hidden-partner тест делается
отдельно (OccludedToss: партнёрская ракетка исключена из наблюдений).

**BC/DAgger — честный отрицательный результат** (`docs/toss-bc-baseline.md`): plain BC 0.000–0.031,
DAgger 0.031–0.062. Причина: oracle компенсирует скрытую latency-очередь (1–4 in-flight команды),
которой нет в 26-мерном наблюдении; одинаковые obs соответствуют разным командам, MSE клонирует
среднее ≈0. Для BC нужно добавить собственные in-flight команды в obs.

Демо (GPU-физика, `frame_physics: gpu (mujoco_warp)`):
`demo/demo_toss_gpuphys_{oracle,policy_vs_oracle,policy_vs_oracle_predicted,policy_vs_oracle_oracle}.mp4`
и `demo_toss_gpuphys_attempts_seed{0,1,2}.mp4` (SUCCESS 42/48/59), плюс GIF/PNG/JSON.

### Hidden-partner OccludedToss (2026-11-22)

Чтобы проверить Bi-JEPA там, где партнёр действительно скрыт, из наблюдений `TossCatch-v0`
убраны все поля **партнёрской ракетки** (поза/скорость/тильт; obs 26→20, в oracle 36→30);
мяч, своя ракетка, цель и время остаются. Физика/награда/рандомизация те же, GPU-only.

Результат (3 обучающих сида × 3 held-out базы × 128 эп., исправленная GPU-физика):

| режим | success (mean) | по сидам | steps-to-success |
|---|---|---|---|
| none | **0.945** | 0.922 / 0.943 / 0.971 | 48.0 |
| oracle | 0.895 | 0.961 / 0.742 / 0.982 | 46.0 |
| predicted (Bi-JEPA) | 0.860 | 0.815 / 0.789 / 0.977 | 47.0 |

Сравнение с полным наблюдением (0.955 / 0.950 / 0.950): скрытие партнёрской ракетки **почти не
вредит** `none` (0.945 vs 0.955) — траектория мяча несёт достаточно информации о партнёре, задача
остаётся реактивно достаточной. `predicted` не обгоняет `none` (0.860), `oracle` шумный
(0.742 на одном сиде) — привилегированное состояние партнёра здесь не даёт устойчивого выигрыша.
Итог: Bi-JEPA не подтверждается и в hidden-partner варианте простой toss-задачи; это
зафиксированная граница применимости, а не сбой обучения (все режимы ≥0.86).

## Ограничения / честные оговорки

- Cart-pole: чистый PD-oracle неустойчив при латентности 0.1–0.3 с → LQR+Smith; чистый RL без
  привилегий сходится медленнее (нужно 1200 апдейтов).
- Pendulum: target 65° (а не 60°) и damping 0.11–0.15 выбраны, чтобы `only-left` стабильно
  не добирал амплитуду, а oracle добирал.
- Toss: с 2026-11-21 ракетки — динамические rigid body с PD-серво и контактами
  ракетка↔ракетка (см. раздел «Физика ракеток toss»), MOCAP-числа устарели; латентность
  0.04–0.16 с; CPU-oracle ~0.89 (остаточные провалы — низкий третий перехват). GPU parity/
  обучение новой физики ещё не прогонялись (под был недоступен).
- Ранние числа (первые прогоны) — с CPU-бэкенда и одной машины под квотой ~7.65 CPU; после
  2026-11-21 все прогоны и eval — на GPU (`--backend warp`), CPU-петля оставлена для parity/CI.
