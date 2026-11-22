# Proposal

## Why

К предзащите/защите нужен не один куб, а набор простых задач, где **два робота выполняют одну задачу** (раскачивание маятника, перекидывание мяча): несколько маленьких измерений координации дешевле и надёжнее, чем одна большая задача, и позволяют отличить «реактивную» координацию от той, где нужно предсказание партнёра.

## What Changes

- Новый reduced MuJoCo suite: `CartPoleCoop-v0`, `PendulumPump-v0`, `TossCatch-v0` на общем двух-агентном харнессе (obs-контракт, partner-режимы, randomized private state, общий reward, oracle-контроллер, held-out eval).
- Чистая логика `_coop_logic.py` (obs layout/видимость/рандомизация/success/reward) и per-task конфиги; интеграция в тренер через `--task` (cube по умолчанию, старый путь без изменений).
- Скрипты smoke/eval и тесты на macOS (pure logic) + серверные smoke-прогоны.
- Метрики: success, время, `oracle−none` на раннем фиксированном бюджете, (позже) `predicted` Bi-JEPA.

Не входит: перенос на руки SO-ARM101 (весенний мост), GPU-warp для новых задач, внешние бенчмарки.

## Impact

Новые модули в `callosum/experiments/coop_tasks/`, `callosum/experiments/arm/`, `callosum/experiments/`, `scripts/`, `tests/`; существующий `CubeWrench-v0` и arms-линия не меняются.
