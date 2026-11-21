# Two-robot coop suite: три простых эксперимента (cart-pole, pendulum, toss&catch)

Статус: **в работе (2026-11-21)**. Черновик отчёта: дизайн задач уже зафиксирован, таблицы
результатов и демо заполняются после прогонов на поде (A40, CPU-бэкенд, квота ~7.65 CPU-ядер).

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

Все три — reduced MuJoCo (без рук), CPU-only в этом suite (96-ядерный хост не нужен: на поде
квота ~7.65 CPU, поэтому прогоны ставятся батчами ≤7).

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
- Железо: RunPod A40, CPU-бэкенд `_vec_coop`; параллелизм ≤7 процессов (квота ~7.65 CPU).

## Результаты

_Заполняется после завершения прогонов: success_rate по задачам/режимам/сидам, `oracle−none`,
`predicted−none`, время до успеха._

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
- Все числа — с CPU-бэкенда и одной машины; GPU-бэкенд для suite не делался (не нужен на этих
  масштабах).
