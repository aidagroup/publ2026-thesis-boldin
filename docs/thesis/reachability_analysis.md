# Достижимость граней (измерения)

## Измерения (из `scripts/coarser_sweep.py` и `tests/test_grasp_waypoints.py`)
- Coarser sweep (8 yaw x 5 pan = 40 выборок): best miss = 0.138 м (13.8 см) при ARM_BASE_OFFSET 0.34 м; с уменьшенным ARM_BASE_OFFSET 0.30 м (`_so100_kinematics.py:52`) лучший miss = 0.110 м — всё ещё > 0.02 м порога достижимости (`scripts/coarser_sweep.py`)
- Порог достижимости: < 0.02 м (2 см) — для успешного захвата и поворота
- Вывод: в текущей геометрии (base yaw π/0, offset 0.34 м) только одна грань (`F` / `F'`) достижима для ротатора.

## Вероятности решения (из `docs/thesis/README.md`)
- Скрембл глубины 1 (`F` или `F'`): 2 / 12 = 16.7 %
- Скрембл глубины 2 (без повторения грани): 0 % (по построению — повтор запрещён)
- Скрембл глубины 3: ~0.33 % (только если все три хода — `F` или `F'`)

## Аббревиатуры
- `F`/`F'`: ход по часовой / против часовой стрелки (Rubik's cube notation)
- `miss`: расстояние TCP ротатора до оси грани (м)
- `dq_h`: ошибка суставов держателя (rad)
- `Bi-JEPA`: Bidirectional Joint Embedding Predictive Architecture
- `IPPO`: Independent Proximal Policy Optimization
- `TCP`: Tool Center Point (точка центра инструмента)

## Вариант B: изменение геометрии рига
Для достижения других граней требуется:
- Изменить `ARM_BASE_OFFSET` (текущий: 0.34 м) или `HOLDER_BASE_X` / `ROTATOR_BASE_X` в `_so100_kinematics.py`
- Перестроить базовые позы (`HOLDER_BASE_YAW`, `ROTATOR_BASE_YAW`) в `two_so100_base.py`
- Перерешать waypoints (`WAYPOINTS`) с помощью `scripts/solve_waypoints.py`
- Повторить `tests/test_grasp_waypoints.py` для проверки
- Обновить `bundle` (`make bundle`) и повторить `probe_grasp.py` на сервере

## Аудит знака поворота (из `scripts/check_turn_sign.py` и `docs/handoff-prompt.md`)
- Положительный угол сустава wrist_roll → ход `F` (по часовой стрелке, `rubik.py`); `scripts/check_turn_sign.py` подтверждает соответствие.
- Возможная инверсия по `handoff-prompt.md` item 2: по кинематическому выводу положительный угол — против часовой стрелки снаружи, в то время как `rubik.py` отображает положительный → ход по часовой (`F`). Требуется проверка на сервере (`probe_grasp.py`).

Измерение (coarser sweep): текущий best miss = 0.138 м при ARM_BASE_OFFSET 0.34 м; при ARM_BASE_OFFSET 0.30 м — 0.110 м. Для достижения другой грани требуется miss < 0.02 м.
