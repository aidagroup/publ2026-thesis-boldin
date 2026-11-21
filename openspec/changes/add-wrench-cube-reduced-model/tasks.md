# Tasks

## 0. Каркас

- [x] 0.1 Extra `sandbox = ["mujoco"]` в `pyproject.toml`, `uv lock`; проверка `import mujoco` на macOS и Linux
- [x] 0.2 `callosum/experiments/cube_wrench/` (пакет, MJCF `assets/cube_wrench.xml`), `callosum/experiments/cube_wrench/config.py`
- [x] 0.3 `callosum/experiments/cube_wrench/logic.py`: раскладка obs, видимость `none|oracle`, сэмплер рандомизации, success/reward/failure — без mujoco/torch
- [x] 0.4 `tests/test_wrench_logic.py`: детерминизм/диапазоны рандомизации, видимость партнёра, граничные значения success/reward, валидация конфига
- [x] 0.5 `openspec/changes/add-wrench-cube-reduced-model/` (proposal/design/tasks/specs); обновление `PLAN.md`, `openspec/project.md`

## 1. Среда MVP + oracle

- [x] 1.1 `WrenchCubeEnv`: Gymnasium API (`reset/step`), Dict-action по агентам, obs/info, терминация/обрезка, `success/angle/drift`
- [x] 1.2 Непрерывные 6D-wrench → `xfrc_applied` (holder→core, rotator→layer), лимиты/латентности/slip из конфига
- [x] 1.3 Скриптовый oracle-контроллер (`wrench_cube/oracle.py`): стабилизация core + доведение слоя до 90°
- [x] 1.4 `scripts/smoke_wrench_cube.py --envs 2 --episodes 5 --controller oracle|random|zero`
- [x] 1.5 **Гейт G1 (пройден 2026-11-21):** oracle 0.988 на 256 эпизодах, rotator-only 0.000 с дрейфом 29.8° > порога oracle ≥0.95 на ≥256 эпизодах (RunPod) и стабильный локальный поворот (1–2 env)

## 2. Тренер IPPO

- [x] 2.1 `callosum/training/wrench_ippo.py`: две гауссовы политики (`_ppo_core.ActorCritic`), GAE, клиппинг, TensorBoard, чекпоинты, resume
- [x] 2.2 Векторизация MuJoCo (цикл по envs; при необходимости process pool), замер throughput
- [x] 2.3 `callosum/experiments/cube_wrench/eval.py` поверх `callosum/common/metrics.py`
- [x] 2.4 Прогоны `none`/`oracle`; подбор рандомизации
- [ ] 2.5 **Гейт G2 (в работе):** `oracle` ≥0.9 и разрыв `oracle−none` ≥0.3 (иначе усилить рандомизацию, не идти дальше)

## 3. Bi-JEPA

- [x] 3.1 Влить модуль из `step/3.1-bijepa-module` (`callosum/common/bijepa.py`, `configs/bijepa.py`, `tests/test_bijepa.py`)
- [x] 3.2 Тренер: энкодер своего наблюдения, предиктор латента партнёра, EMA-target, aux-лосс; `ẑ_j` — вход политики
- [x] 3.3 Логирование `latent_stats` (std/effective rank) и прогноза; отсутствие утечки данных партнёра в `none`/`predicted`
- [x] 3.4 `predicted-direct` baseline (регрессор текущего wrench партнёра) — если останется время
- [ ] 3.5 **Гейт G3 (2026-12-05):** измеримый `predicted`; иначе fallback-скоуп предзащиты (`none/oracle` + прогресс)

## 4. Абляция и метрики

- [ ] 4.1 ≥3 сида × `none/oracle/predicted` при одинаковых бюджетах
- [ ] 4.2 Метрики: success rate, угол (град.), дрейф (мм/рад), время до успеха, сбои, закрытая доля разрыва `(S_pred−S_none)/(S_oracle−S_none)`
- [ ] 4.3 Сводка: JSON + TensorBoard + таблица mean ± разброс; конфиги/сиды в репозитории
- [ ] 4.4 **Гейт G4 (2026-12-10):** числа/графики готовы, прогоны воспроизводимы

## 5. Демо и документы

- [ ] 5.1 Демо: стабильный поворот, «момент помощи» (rotator без holder'а → дрейф/провал), slip и восстановление
- [ ] 5.2 Обновить `README.md`, `openspec/project.md`, `docs/thesis/02–04`, `docs/cloud-gpu.md` (команды MuJoCo-прогонов)
- [ ] 5.3 Материал предзащиты: постановка, ablation-таблица, демо, план весеннего моста на руки
