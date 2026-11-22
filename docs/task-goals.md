# Явные цели задач coop-suite: что должен сделать агент, чтобы эпизод считался успешным

Статус: **актуально с 2026-11-22** (ветка `step/2.4-wrench-cube`). Документ отвечает на
вопрос «что именно требуется от пары агентов» для трёх демонстрационных задач reduced-suite:
`CartPoleCoop-v0`, `PendulumPump-v0`, `TossCatch-v0`. Все числа взяты из
`callosum/experiments/coop_tasks/config.py` (dataclass-конфиги `CartPoleConfig`,
`PendulumConfig`, `TossConfig`) и из правил `callosum/experiments/coop_tasks/logic.py`
(`cartpole_success`, `cartpole_failure`, `pendulum_success`, `pendulum_failure`,
`toss_success`, `toss_floor_failure`). Те же числа показывает рендерер
`callosum/experiments/coop_tasks/render.py` — функция `task_goal()` строит строки
`GOAL:`/`SUCCESS:` из конфига, поэтому демо и этот документ не могут разойтись.

## Таблица целей

| Задача (env id) | Агенты и управление | Цель | Точное условие успеха (числа из конфига) | Бюджет / таймаут | Что означает FAILED |
|---|---|---|---|---|---|
| **cart-pole** (`CartPoleCoop-v0`) | 2 односторонних толкателя тележки: `left` умеет только `+x` (сила ≤ `max_force = 30 N`), `right` — только `-x` (≤ 30 N). Действие — нормированная сила, на каждого агента своя приватная задержка 0.1–0.3 с и свой gain. | Удержать шест и привезти тележку в цель `target_x = +0.50 m` (на рельсе ±`rail_limit = 1.4 m`). | На **одном** control step одновременно: `|cart_x − 0.50| ≤ success_pos_tol = 0.05 m` (5 см), `|pole_angle| ≤ success_angle_tol = 0.1 rad` (5.7°), `|cart_vx| ≤ success_lin_speed_tol = 0.10 m/s`, `|pole_angvel| ≤ success_ang_speed_tol = 0.20 rad/s`. Отдельного «шагов удержания» в конфиге нет: требование «приехал и остановился» задано допусками на скорости. | `max_control_steps = 300` control steps × `control_dt = 0.04 s` = **12.0 s** симуляции; при исчерпании бюджета — `TIMEOUT` (truncated), не FAILED. | `|pole_angle| > failure_angle = 0.5 rad` (**28.6°**) — шест упал, эпизод терминируется досрочно. |
| **pendulum pump** (`PendulumPump-v0`) | 2 односторонних момента на шарнире: `left` — только положительный момент (≤ `max_torque = 0.65 Nm`), `right` — только отрицательный (≤ 0.65 Nm). Задержки 0.075–0.2 с. | Раскачать маятник до амплитуды `target_amplitude = 65.0°` и **удержать** её `hold_cycles = 2` завершённых полупериода (2 = одно полное колебание на амплитуде). | `peak |angle| ≥ 65.0°` на **2** завершённых полупериодах: полупериод засчитывается на переходе через ноль, если пик `|angle|` с прошлого перехода достиг `target_amplitude`. Правило: `completed_cycles ≥ hold_cycles`. | `max_control_steps = 400` × 0.04 s = **16.0 s** симуляции; иначе `TIMEOUT`. | `|angle| > failure_angle = 150°` — маятник почти переворачивается (`hinge_limit = 3 rad`), эпизод терминируется. |
| **toss & catch** (`TossCatch-v0`) | 2 физические ракетки: у каждой агент командует 3 нормированных приращения позы серво-цели — `x / z / tilt`; скорость ≤ `pad_speed = 9 m/s` по осям, скорость наклона ≤ `pad_tilt_rate = 3 rad/s`, наклон ограничен `pad_tilt_limit = 60°`. `left` бросает, `right` возвращает; пас засчитывается только если мяч реально перелетел центр и коснулся ракетки на её половине. | Держать ралли: `n_passes = 3` чередующихся паса (бросок → приём → возврат), не уронив мяч и не выбросив его за корт. | На шаге успеха одновременно: `passes ≥ 3`; мяч **строго выше** `floor_z + success_floor_margin = 0.07 + 0.02 = 0.09 m`; состояние не нарушает `toss_floor_failure` (мяч в корте: `|x| ≤ court_x = 1.35 m`, `|y| ≤ court_y = 0.60 m`, `z ≥ floor_z = 0.07 m`). Между пасами минимум `min_pass_interval_steps = 4` control steps (анти-дребезг контакта). | `max_control_steps = 400` × 0.04 s = **16.0 s** симуляции; иначе `TIMEOUT`. Для честного ралли нужно ≥ 28 control steps на 3 паса (см. docstring `min_pass_interval_steps`), поэтому успешные эпизоды демо длятся ≥ ~1.1 s симуляции. | Мяч упал ниже `floor_z = 0.07 m` («dropped») или ушёл за корт `|x| > 1.35 m` / `|y| > 0.60 m` («left court»). Провал имеет приоритет над успехом: шаг, пересекающий границу провала, не может засчитать success. |

Примечания к таблице:

- В coin-задачах reward — общая командная награда (одинаковая обоим агентам), поэтому
  «успех одного» невозможен: условие проверяется на общем состоянии.
- Успех/провал проверяются **на control step** (каждые `control_every = 8` физических
  подшагов по `dt = 0.005 s`, т.е. `control_dt = 0.04 s`).
- Числа соответствуют дефолтным конфигам. В eval-JSON (`results_coop_gpu/`, `results_toss_phys2/`)
  те же поля (`success`, `passes`, `time_s`, `steps`) записаны по этому же правилу; обученные
  чекпоинты используют конфиг из своего `config.json`.

## Где это видно в демо

Рендерер (`python -m callosum.experiments.coop_tasks.render`) показывает цель **на каждом
кадре**:

- **Верхний блок HUD** (на всю ширину кадра, поверх сцены): строка `GOAL: <цель>` и строка
  `SUCCESS: <точное условие с числами>`. Блок читается и на 640×480, и на 1280×480
  (side-by-side клип: по половине на каждый агент/контроллер).
- **Вступительная карточка** (2.5 s): название задачи, кто такие `left`/`right` и чем они
  управляют, цель, условие успеха, правило FAILED и бюджет эпизода. Карточка нарисована
  поверх затемнённого первого кадра сцены (не чёрный кадр).
- **Финальная карточка** (2.0 s): `SUCCESS` / `FAILED` / `TIMEOUT`, причина, достигнутые
  метрики (для cart-pole: финальные `cart x`, `|angle|`, `|v|`, `|ω|`; для pendulum: пик
  амплитуды и число удержаний; для toss: число пасов, `ball x/z/vz`) и повтор условия успеха.
- Живой прогресс внизу кадра: `passes k/3` и `ball z` для toss, `peak amplitude`/`hold
  cycles` для pendulum, `cart x → goal`/`|pole angle|` для cart-pole, время `t = … s` и
  `step k/max`.

## Длительность клипов

- `--target-seconds` (по умолчанию **12 s**) — целевая длительность **эпизодной** части:
  кадры растягиваются frame-holding'ом (slow motion), т.е. каждый отрендеренный кадр
  повторяется целое число раз.
- `--intro-seconds` (**2.5 s**) и `--outro-seconds` (**2.0 s**) добавляются сверху; frame
  holding округляется **вверх**, поэтому эпизодная часть не короче `--target-seconds`.
  Итоговый клип по умолчанию ≥ **16.5 s** (например, toss-oracle: 32 уникальных кадра × 6
  повторов = 12.8 s эпизода + 2.5 + 2.0 = 17.3 s; самый короткий клип финального набора —
  16.5 s, самый длинный — 23.6 s). Требование «итоговый клип ≥ 12–15 s» выполняется с запасом.
- `--scene attempts` рендерит отдельный клип на каждую попытку (intro-карточка
  `ATTEMPT i/3 · seed s`, outro с итогом попытки), поэтому суммарно это ≥ 3 × 16.5 s ≈ 50 s;
  для каждой попытки в JSON есть `status`, `steps`, `time_s`, `episode_sim_seconds`,
  `video_duration_s`.

JSON сцены (рядом с MP4) содержит: `goal`, `success_condition`, `failure_condition`,
`target_seconds`, `intro_seconds`, `outro_seconds`, `episode_sim_seconds`,
`video_duration_s`, `frame_physics` (`gpu (mujoco_warp)` для `--backend warp`), а также
`episodes`/`attempts` со статусами и метриками.

## Как перерендерить

GPU-физика (`mujoco_warp`, `CoopBatchGPU`) + EGL-рендер; команды — на поде из `/root/callosum`
(venv `/root/.callosum/venv`):

```bash
PY=/root/.callosum/venv/bin/python
OUT=/workspace/callosum-runs/render_goal
MUJOCO_GL=egl PYTHONPATH=. $PY -m callosum.experiments.coop_tasks.render \
    --task toss --scene oracle --seed 0 --backend warp \
    --width 640 --height 480 --fps 15 --target-seconds 12 --out $OUT/oracle

MUJOCO_GL=egl PYTHONPATH=. $PY -m callosum.experiments.coop_tasks.render \
    --task toss --scene policy-vs-oracle --backend warp \
    --checkpoint /root/callosum/runs/coop_toss_phys2_none_s0/latest.pt --input-mode none \
    --out $OUT/policy_vs_oracle

MUJOCO_GL=egl PYTHONPATH=. $PY -m callosum.experiments.coop_tasks.render \
    --task toss --scene policy-vs-oracle --backend warp \
    --checkpoint /root/callosum/runs/coop_toss_phys2_pred_s0/latest.pt --input-mode predicted \
    --out $OUT/predicted

MUJOCO_GL=egl PYTHONPATH=. $PY -m callosum.experiments.coop_tasks.render \
    --task toss --scene attempts --attempts 3 --seed 0 --backend warp \
    --checkpoint /root/callosum/runs/coop_toss_phys2_none_s0/latest.pt --input-mode none \
    --out $OUT/attempts
```

Готовые клипы скачаны в `demo/` с префиксом `demo_<task>_goal_`
(`demo_cartpole_goal_oracle.mp4`, `demo_pendulum_goal_policy_vs_oracle_predicted.mp4`,
`demo_toss_goal_attempts_seed0.mp4` и т.д.) плюс GIF-превью, PNG-кадры и JSON.
