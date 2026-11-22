# BC-бейзлайн TossCatch-v0 на GPU-физике: результат и диагноз

Краткая версия: поведенческое клонирование скриптового oracle на 26-мерном наблюдении
**не даёт рабочей политики** (held-out success 0.0–0.06), и это не ошибка пайплайна, а
не-марковость самого учителя (см. «Диагноз»). Попутно найден и исправлен критический баг
GPU-физики toss, из-за которого **любой** warp-прогон имел success = 0.

## TL;DR числа

| Что | held-out success (128 эп., warp, 128 envs) |
| --- | --- |
| plain BC (`runs/toss_bc_plain_s0/latest.pt`) | base0 **0.000**, base1M **0.000**, base2M **0.031** |
| DAgger, 3 итерации (`runs/toss_bc_dagger_s0/latest.pt`) | base0 **0.062**, base1M **0.047**, base2M **0.031** |
| BC-инициализация + PPO warm start (208 апдейтов) | eval-success **0.000** (32 эп.) |
| scripted oracle (тот же GPU-env) | **0.862** (1022 эп., seeds 5.0M+) |

Цель ≥0.5 не достигнута. Ниже — что сделано, почему не вышло и что чинить.

## 1. Критический фикс GPU-env (уже закоммичен как `9c80ec6`)

В `callosum/experiments/coop_tasks/env_gpu.py` сырые флаги контакта мяч/ракетка
(`_flags_warp`) **никогда не обнулялись**: `contact_side` залипал после первого контакта и
rising-edge в pass-гейте (`_update_passes`) срабатывал максимум дважды за эпизод, поэтому
3-й пас (и success) были физически недостижимы. Это объясняет success = 0.0 во всех
warp-прогонах toss, включая старый `coop_toss_phys_oracle_long`.

Фикс: `self._flags.zero_()` раз в control step перед подшагами (плюс `GpuDemoEnv` в
`render.py` теперь отдаёт `*_pad_target_*` и `*_contact` в `state()`, без чего oracle на GPU
и рендер падали с KeyError). Проверки после фикса:

* `check_coop_gpu_parity.py --task toss --controller oracle` → **PARITY PASS**
  (qpos 1.4e-3, qvel 6.4e-3, obs 2.0e-3, reward 1.3e-4; 0 расхождений флагов/эпизодов);
* scripted oracle на GPU: 4/4 на seeds 5.0M, а на полном сборе — **0.862** (см. ниже);
* рендер `policy-vs-oracle` на `--backend warp` запускается (Egl, GPU-физика).

## 2. Демонстрации oracle на GPU

`demos/toss_bc.npz` (+ `demos/toss_bc.json`), скрипт
`scripts/experiments/coop_tasks/collect_toss_bc_demos.py`:

* `CoopBatchGPU`, `partner_mode="none"`, `occlusion="none"`, `num_envs=128`, 8 батчей,
  seeds **5_000_000..5_001_023** (вне eval- и train-сидов), первый эпизод каждого world;
* **33 903 transition'а** (27 742 из успешных эпизодов), 1022 complete эпизода
  (+2 обрезанных cap'ом), oracle success **0.862**, 29.7 s на A40;
* формат: `obs (T,2,26) float32`, `actions (T,2,3)`, `episode (T,)`, `episode_success`,
  `episode_complete`, `episode_seed`, `episode_length`; obs (26) и действия (3) — ровно те,
  что ждёт `wrench_ppo`-чекпоинт при `input_mode=none`.

## 3. BC-политика

`scripts/experiments/coop_tasks/train_toss_bc.py`, MLP — штатный
`callosum.common.ppo_core.ActorCritic` (26→256×3 tanh→3), чтобы чекпоинт читался
`coop_tasks/eval.py` и `coop_tasks/render.py` без правок; обучается только actor-mean (MSE к
действиям oracle), нормировка obs сворачивается в первый `Linear` (политика ест сырое obs).

* train/val split по эпизодам (15% эпизодов), 142 эпохи (early stop), Adam 1e-3, batch 2048;
* **val action MSE 0.0082** (left `[0.0087, 0.0155, ~0]`, right `[0.0087, 0.0161, ~0]`;
  в основном x/z, tilt почти нулевой);
* held-out success при обучении (свежие seeds 5_500_000+): **0.023**;
* чекпоинт: `/root/callosum/runs/toss_bc_plain_s0/latest.pt` (vалидный `wrench_ppo`,
  eval читает без ошибок), лог `bc_log.json`.

Eval (`python -m callosum.experiments.coop_tasks.eval --input-mode none --backend warp
--num-envs 128 --episodes 128`) на held-out base {0, 1_000_000, 2_000_000}:

| base | success | passes | steps | JSON |
| --- | --- | --- | --- | --- |
| 0 | 0.000 (0/128) | 0.46/3 | 17.4 | `results_toss_bc/bc_plain_base0.json` |
| 1_000_000 | 0.000 (0/128) | 0.45/3 | 16.8 | `results_toss_bc/bc_plain_base1000000.json` |
| 2_000_000 | 0.031 (4/128) | 0.51/3 | 17.4 | `results_toss_bc/bc_plain_base2000000.json` |

## 4. DAgger (3 итерации) — не помог

`scripts/experiments/coop_tasks/dagger_toss_bc.py`: раскатка текущей BC-политики на GPU,
метки oracle на посещённых состояниях (25% шагов исполняется действие oracle для покрытия),
агрегация β=0, переобучение на 42 028 transition'ах. Лог: `runs/toss_bc_dagger_s0/dagger_log.json`.

| итерация | rollout success (с 25% oracle) | retrain held-out success |
| --- | --- | --- |
| 1 | 0.016 | 0.031 |
| 2 | 0.180 | 0.031 |
| 3 | 0.102 | 0.023 |

Eval лучшего DAgger-чекпоинта (`runs/toss_bc_dagger_s0/latest.pt`):

| base | success | passes | JSON |
| --- | --- | --- | --- |
| 0 | 0.062 (8/128) | 1.31/3 | `results_toss_bc/bc_dagger_base0.json` |
| 1_000_000 | 0.047 (6/128) | 1.24/3 | `results_toss_bc/bc_dagger_base1000000.json` |
| 2_000_000 | 0.031 (4/128) | 1.20/3 | `results_toss_bc/bc_dagger_base2000000.json` |

## 5. BC + PPO warm start — не учится за короткий прогон

`python -m callosum.experiments.cube_wrench.train --resume runs/toss_bc_ppo_s0
--backend warp --device cuda --total-updates 342`: 208 апдейтов (540 672 шагов, 220 s),
все greedy-eval'ы **success 0.000**; остановлен вручную. Чекпоинт
`runs/toss_bc_ppo_s0/latest.pt` (для сравнения, не рекомендуется).

## 6. Диагноз: почему BC от oracle не работает

Oracle — замкнутый планировщик, чья команда зависит от **скрытой очереди латентности**:
`TossOracle._deadbeat` считает желаемое приращение серво-таргета как
`target - servo_target - sum(pending commands)`. В наблюдении 26 есть только **последнее
измеренное** (`action`), свои `latency_s/gain/limit_scale/start_delay`, состояние мяча и
обеих ракеток. Сами in-flight команды (1–4 штуки) не наблюдаемы, поэтому одно и то же obs
соответствует многим разным командам, и их условное среднее — то, что минимизирует MSE BC —
оказывается ≈0 ровно там, где нужна точность (первый перехват). Отсюда:

* val MSE низкий (0.008), но success 0–6%;
* DAgger с 25% oracle-шагов доводит rollout до 18%, но переобученная студенческая политика
  снова 3% — остаточная неоднозначность не устраняется;
* попытки «марковского учителя» (та же планировка, команда из измеренной позы с
  one-step предсказанием; и вариант без компенсации pending) дают **0/32** эпизодов
  (свип гейнов 0–2/8): наивная обратная связь неустойчива при латентности 1–4 шага.

## 7. Что чинить (по приоритету)

1. **Добавить собственный блок in-flight команд в наблюдение** (по аналогии с `oracle`-блоком
   партнёра): например последние L команд или `servo_target + pending_sum`. Тогда
   obs→action становится марковским, и BC от oracle должен клонироваться нормально. Это
   изменение layout'а (26→29/32) и перезапуск RL-прогонов; тесты/parity уже есть.
2. **Марковский учитель с observer'ом**: оценивать `servo_target` и очередь по наблюдаемой
   позе/скорости, затем BC от него. Требует стабилизирующего регулятора (наивные варианты
   провалились), поэтому дороже, чем п.1.
3. **RL**: после фикса GPU-env задача решается (по данным lead'а `none` ~0.55 на 143/250);
   это практический путь к рабочей демо-политике. BC разумен как инициализация, но warm start
   с дефолтным `actor_logstd=-0.5` разрушает поведение: нужен меньший logstd и critic warmup.
4. **Данные**: увеличивать объём oracle-демо бессмысленно при не-марковском учителе
   (проверено: 33.9k transition'ов и DAgger до 42k не меняют картину).

## 8. Воспроизведение

```bash
cd /root/callosum
# 1) демо oracle на GPU (после фикса GPU-env, commit 9c80ec6)
MUJOCO_GL=egl /root/.callosum/venv/bin/python \
    scripts/experiments/coop_tasks/collect_toss_bc_demos.py \
    --num-envs 128 --batches 8 --max-steps 120 --out demos/toss_bc.npz
# 2) BC
/root/.callosum/venv/bin/python scripts/experiments/coop_tasks/train_toss_bc.py \
    --demos demos/toss_bc.npz --exp-name toss_bc_plain_s0 --epochs 300 \
    --eval-every 25 --eval-episodes 128 --device cuda
# 3) DAgger (опционально, не улучшает)
MUJOCO_GL=egl /root/.callosum/venv/bin/python \
    scripts/experiments/coop_tasks/dagger_toss_bc.py --demos demos/toss_bc.npz \
    --exp-name toss_bc_s0 --iterations 3 --num-envs 128 --oracle-prob 0.25
# 4) eval 128 x 3 базы
for base in 0 1000000 2000000; do
  /root/.callosum/venv/bin/python -m callosum.experiments.coop_tasks.eval \
    --checkpoint runs/toss_bc_dagger_s0/latest.pt --input-mode none --backend warp \
    --num-envs 128 --episodes 128 --seed "$base" \
    --out "results_toss_bc/bc_dagger_base${base}.json"
done
```

## 9. Файлы

* Новые (untracked, готовы к коммиту): `scripts/experiments/coop_tasks/collect_toss_bc_demos.py`,
  `scripts/experiments/coop_tasks/train_toss_bc.py`,
  `scripts/experiments/coop_tasks/dagger_toss_bc.py`.
* Изменённые (закоммичены в `9c80ec6`): `callosum/experiments/coop_tasks/env_gpu.py`,
  `callosum/experiments/coop_tasks/render.py`.
* Артефакты на поде: `demos/toss_bc.npz|json`, `demos/toss_bc_dagger.npz|json`,
  `runs/toss_bc_plain_s0/`, `runs/toss_bc_dagger_s0/`, `runs/toss_bc_ppo_s0/`,
  `results_toss_bc/*.json`.
