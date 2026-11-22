# План рефакторинга репозитория (2026-11-21)

Статус: **ВЫПОЛНЕНО 2026-11-21** (S1–S5, коммиты `refactor(s1)…(s5)`; backup-ветка `pre-refactor-2026-11-21`). Итоговое дерево — в разделе «Финальная структура» ниже; все тесты (595 passed, 5 skipped) и `ruff check/format` зелёные.

## Проблемы текущей структуры

- `callosum/envs/`, `configs/`, `training/`, `experts/` — **плоские** пакеты, в которых смешаны три разные линии: reduced cube (MuJoCo), skill-level arms (ManiSkill), joint-level arms (legacy). Понять, что к чему относится, можно только по именам файлов.
- Общие утилиты размазаны: `_ppo_core.py`, `_categorical.py`, `_checkpoint.py`, `_nonfinite.py`, `_metrics.py`, `_diag.py`, `eval/metrics.py`, `configs/_cli.py`, `agents/bijepa.py`, `_partner_obs.py` — используются несколькими линиями, но лежат в чужих подпакетах.
- Дубли: векторизация (`_vec_wrench.py` → будет `_vec_coop.py`), eval-луп (`eval_wrench_cube.py`/`eval_skills.py`/`eval_coop.py`), smoke-обвязка (`_sim_utils.py`, `smoke_*`), рендер (`render_wrench_demo.py` + `_render_utils.py`), анализ (`plot_wrench_curves.py`, `summarize_wrench_gpu.py`).
- `scripts/` — плоский список из ~25 файлов разных назначений (эксперименты, диагностика, облако, анализ).
- `tests/` — плоский список из 31 файла.
- Legacy joint-level линия (BC/демо/`ippo.py`/эксперты) — ~4–5k строк, не участвует в активной линии, но нужна как негативный результат/мотивация диплома.

## Целевая структура

```
callosum/
  common/                     # переиспользуемое, experiment-agnostic
    cli.py                    # из configs/_cli.py
    ppo_core.py               # из training/_ppo_core.py
    categorical.py            # из training/_categorical.py
    checkpoint.py             # из training/_checkpoint.py
    nonfinite.py              # из training/_nonfinite.py
    logging.py                # из training/_metrics.py (+ _diag.py)
    metrics.py                # из eval/metrics.py
    visibility.py             # из envs/_partner_obs.py (общее правило видимости партнёра)
    bijepa.py                 # из agents/bijepa.py
    bijepa_config.py          # из configs/bijepa.py
    randomization.py          # общие хелперы (если выделяются)
    mujoco.py                 # из callosum/common/render_utils.py (загрузка MJCF/рендер-хелперы, без экспериментов)
  experiments/
    cube_wrench/              # reduced MuJoCo cube (текущая линия)
      env.py, env_gpu.py, oracle.py, logic.py, config.py
      train.py                # из training/wrench_ppo.py
      eval.py                 # из callosum/experiments/cube_wrench/eval.py
      render.py               # из callosum/experiments/cube_wrench/render.py
    coop_tasks/               # новый suite: cart-pole, pendulum, toss&catch
      base.py, cartpole.py, pendulum.py, toss_catch.py, logic.py, config.py, train.py, eval.py
    arm_skills/               # skill-level SO-ARM101 (заморожено до весны)
      env.py, obs.py, logic.py, plans.py, cube.py, face_lock.py
      runtime.py, conductor.py                # из experts/skills_*
      train.py                # категориальный IPPO (из training/ippo.py? отдельный путь)
      eval.py                 # из python -m callosum.experiments.arm.skills_eval
    arm_joint_legacy/         # joint-level линия: мотивация/негативный результат
      README.md               # «сохранено как evidence, не развивается»
      env.py, base.py, oracle.py, expert.py, phases.py, tracking.py, noise.py
      train.py (ippo), bc.py, demos.py, config
  robots/                     # кастомный SO-ARM101 (общий)
  assets/so101_parallel_gripper/
```

`scripts/`:
```
scripts/
  common/                     # sim_utils.py, render_utils.py, cli helpers
  experiments/
    cube_wrench/              # smoke, gpu parity/bench/determinism, grid
    coop_tasks/               # smoke/eval/demo
    arm_skills/               # smoke, probe, eval
    arm_joint_legacy/         # collect_demos, probe_face_turn, smoke_env
  analysis/                   # plot_wrench_curves, summarize
  ops/                        # cloud_resume_loop.sh, update_server.sh и пр.
```

`tests/` зеркалит: `tests/common/`, `tests/experiments/{cube_wrench,coop_tasks,arm_skills,arm_joint_legacy}/`.

## Что вырезаем/что архивируем

- **Удаляем без сожаления:** пустые пакеты/`__init__` после переноса; дублирующиеся хелперы; неиспользуемые импорты; untracked `presentation/` (7.7 MB локальных артефактов — в `.gitignore`, из репо не удаляем); `demo/`, `results_gpu/` остаются ignored.
- **Архивируем, не удаляем:** joint-level линия (BC/демо/`training/ippo.py`/`experts/face_turn_expert.py` и пр.) → `experiments/arm_joint_legacy/` с README: это мотивация диплома (BC-негативный результат) и источник эксперта для демо рук; удаление сломало бы отчёт.
- **Не трогаем:** `robots/`, `assets/`, `agents/bijepa` (переносим в common), openspec/docs (обновим только ссылки/пути).

## Порядок работ (после агентов)

1. **S1. Common.** `git mv` общих модулей в `callosum/common/`, автоматическая замена импортов, прогон `pytest -q` + `ruff`.
2. **S2. Cube experiment.** `callosum/experiments/cube_wrench/*`, `_wrench_logic.py`, `configs/wrench_cube.py`, `wrench_ppo.py`, `eval/…` → `experiments/cube_wrench/`; обновить `python -m` команды и доки.
3. **S3. Coop experiment.** Новые файлы агента → `experiments/coop_tasks/`; общий тренер/delegate.
4. **S4. Arms.** skill-level и joint-level — в `experiments/arm_skills/` и `arm_joint_legacy/`; обновить `scripts/smoke_*`, runbook.
5. **S5. Scripts/tests/docs.** Сгруппировать scripts/tests, обновить `README.md`, `PLAN.md`, `docs/*`, `openspec/*`, CI-импорт-тест; `pytest -q`, `ruff check/format`.
6. **S6. Финальный diff-review** против этого плана, удалить оставшиеся дубли/мёртвый код, финальный отчёт.

## Инварианты

- Все 518+ тестов зелёные после каждого шага; `ruff check` чистый.
- Публичные env id (`CubeWrench-v0`, `FaceTurn-v0`, `FaceTurnSkills-v0`, `TwoSO101-v0`) и CLI-контракты (`python -m ...`) сохраняются или явно документируются в новых путях.
- Один шаг = один коммит; ветка `step/2.4-wrench-cube`; без изменения поведения экспериментов.


## Финальное дерево (2026-11-21)

```
callosum/
  common/            cli, ppo_core, categorical, checkpoint, nonfinite, logging, diag,
                     diag_names, metrics, visibility, bijepa(+config,+probe), render_utils
  experiments/
    cube_wrench/     env, env_gpu, logic, config, ppo_config, train, vec_env, eval, render, oracle
    coop_tasks/      base, cartpole, pendulum, logic, config, oracle, vec_env, eval
    arm/             scene, layout, turntable, face_lock, sim_compat, sim_utils, expert,
                     expert_config, phases, program, tracking, noise, ppo_config, trainer,
                     agent_obs, skills, skills_obs, skills_agent_obs, demos, bc, bc_config;
                     skills_env, skills_cube, skills_plans, skills_runtime, skills_conductor,
                     skills_config, skills_cube_config, skills_eval; face_turn_env, face_turn_config
  robots/            so101_parallel_gripper
  assets/            so101_parallel_gripper (URDF/meshes)
scripts/
  experiments/{cube_wrench,coop_tasks,arm}/, analysis/, ops/
tests/
  common/, experiments/{cube_wrench,coop_tasks,arm}/, test_import.py, test_assets.py
```

Удалено/вычищено: пакеты `callosum/envs`, `callosum/configs`, `callosum/experts`,
`callosum/training`, `callosum/eval`, `callosum/agents` (их содержимое перенесено);
дублирующие `_sim_utils`/`_render_utils` вынесены в общие модули; `docs/refactor-plan.md`
зафиксировал политику (legacy joint-level архив — в `experiments/arm`, а не удалён как
мотивация диплома). Артефакты `demo/`, `results_gpu/`, `presentation/` — в `.gitignore`.
