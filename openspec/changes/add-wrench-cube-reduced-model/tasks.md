# Tasks

## 0. Каркас

- [ ] 0.1 Extra `sandbox = ["mujoco"]` в `pyproject.toml`, `uv lock`; проверка `import mujoco` на macOS и Linux
- [ ] 0.2 `callosum/envs/wrench_cube/` (пакет, MJCF `assets/cube_wrench.xml`), `callosum/configs/wrench_cube.py`
- [ ] 0.3 `callosum/envs/_wrench_logic.py`: раскладка obs, видимость `none|oracle`, сэмплер рандомизации, success/reward/failure — без mujoco/torch
- [ ] 0.4 `tests/test_wrench_logic.py`: детерминизм/диапазоны рандомизации, видимость партнёра, граничные значения success/reward, валидация конфига
- [ ] 0.5 `openspec/changes/add-wrench-cube-reduced-model/` (proposal/design/tasks/specs); обновление `PLAN.md`, `openspec/project.md`

## 1. Среда MVP + oracle

- [ ] 1.1 `WrenchCubeEnv`: Gymnasium API (`reset/step`), Dict-action по агентам, obs/info, терминация/обрезка, `success/angle/drift`
- [ ] 1.2 Непрерывные 6D-wrench → `xfrc_applied` (holder→core, rotator→layer), лимиты/латентности/slip из конфига
- [ ] 1.3 Скриптовый oracle-контроллер (`wrench_cube/oracle.py`): стабилизация core + доведение слоя до 90°
- [ ] 1.4 `scripts/smoke_wrench_cube.py --envs 2 --episodes 5 --controller oracle|random|zero`
- [ ] 1.5 **Гейт G1:** oracle ≥0.95 на ≥256 эпизодах (RunPod) и стабильный локальный поворот (1–2 env)

## 2. Тренер IPPO

- [ ] 2.1 `callosum/training/wrench_ippo.py`: две гауссовы политики (`_ppo_core.ActorCritic`), GAE, клиппинг, TensorBoard, чекпоинты, resume
- [ ] 2.2 Векторизация MuJoCo (цикл по envs; при необходимости process pool), замер throughput
- [ ] 2.3 `scripts/eval_wrench_cube.py` поверх `callosum/eval/metrics.py`
- [ ] 2.4 Прогоны `none`/`oracle`; подбор рандомизации
- [ ] 2.5 **Гейт G2:** `oracle` ≥0.9 и разрыв `oracle−none` ≥0.3 (иначе усилить рандомизацию, не идти дальше)

## 3. Bi-JEPA

- [ ] 3.1 Влить модуль из `step/3.1-bijepa-module` (`callosum/agents/bijepa.py`, `configs/bijepa.py`, `tests/test_bijepa.py`)
- [ ] 3.2 Тренер: энкодер своего наблюдения, предиктор латента партнёра, EMA-target, aux-лосс; `ẑ_j` — вход политики
- [ ] 3.3 Логирование `latent_stats` (std/effective rank) и прогноза; отсутствие утечки данных партнёра в `none`/`predicted`
- [ ] 3.4 `predicted-direct` baseline (регрессор текущего wrench партнёра) — если останется время
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
