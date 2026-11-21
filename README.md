# AIDA. Final Thesis. Callosum

> Две руки-манипулятора **SO-ARM101** с параллельным захватом Robonine, каждая — самостоятельный децентрализованный агент, совместно собирают кубик Рубика. Координация — через **Bi-JEPA** (латентное предсказание партнёра, без обмена сообщениями). Обучение в GPU-симуляции (ManiSkill) с переносом **sim2real** на реальное железо.

**📂 Рабочие материалы для диплома:** [`docs/thesis/`](docs/thesis/README.md) — заметки-исходники (идея, обзор литературы, метод, эксперименты, стек).
**⚙️ Сетап и деплой:** [`docs/setup.md`](docs/setup.md).

**🧭 Текущий статус (2026-11-21):** главная линия к предзащите 2026-12-14 — **reduced-модель `CubeWrench-v0`** (MuJoCo): изолированный куб «ядро + активный слой», два децентрализованных агента прикладывают непрерывные 6D-моменты; ablation `none/oracle/predicted` (Bi-JEPA). Руки SO-ARM101 (ветка `step/2.3-skill-level-marl`, skill-level MARL) заморожены как весенний мост. Подробности — [`PLAN.md`](PLAN.md), дорожная карта — [`openspec/project.md`](openspec/project.md); бенчмарк-анализ — [`docs/thesis/02-related-work.md`](docs/thesis/02-related-work.md). Данные и окружение на сервере хранятся в постоянном `$HOME` (`~/callosum`), см. [`docs/setup.md`](docs/setup.md) и [`docs/server-runbook.md`](docs/server-runbook.md). Правила работы для ассистента – в [`CLAUDE.md`](CLAUDE.md).

## Результаты reduced-модели (2026-11-21)

К предзащите главная линия — **reduced-модель `CubeWrench-v0`** (MuJoCo): изолированный куб «ядро + активный слой», два децентрализованных агента прикладывают непрерывные 6D-моменты; руки SO-ARM101 — весенний мост. Полный отчёт: [`docs/results-wrench-cube.md`](docs/results-wrench-cube.md), план и гейты — [`PLAN.md`](PLAN.md), спецификации — [`openspec/changes/add-wrench-cube-reduced-model/`](openspec/changes/add-wrench-cube-reduced-model/).

- Среда валидна: скриптовый oracle 0.99 на 256 эпизодах, `rotator-only` 0.0 (неаддитивность).
- GPU-бэкенд (`mujoco_warp`, CUDA graph): 10.8×/26×/53× env-throughput, parity и determinism проверены.
- Честная абляция (held-out, 128 эп. × 3 сида): **none 1.000 / oracle 0.901 / predicted 0.932** на default; на harsh — **oracle 0.969 / none 0.839** (gap +0.13). Асимптотического разрыва нет; привилегированная информация даёт скорость обучения и устойчивость, Bi-JEPA в текущей реализации пока не обгоняет none — отрицательный промежуточный результат, план улучшения в отчёте.
- Демо: [`demo/demo_policy_vs_oracle.mp4`](demo/demo_policy_vs_oracle.mp4), [`demo/demo_preview.gif`](demo/demo_preview.gif), [`demo/demo_wrench_arrows.mp4`](demo/demo_wrench_arrows.mp4) (видно, кто держит/крутит), [`demo/demo_arms_face_turn.mp4`](demo/demo_arms_face_turn.mp4) (реальные две SO-ARM101 в ManiSkill: 90° и SUCCESS), [`demo/learning_curves_honest_seed0.png`](demo/learning_curves_honest_seed0.png), [`demo/learning_curves_harsh.png`](demo/learning_curves_harsh.png).

## Структура репозитория (2026-11-21)

Код организован по экспериментам; переиспользуемое — в общем пакете:

```
callosum/
  common/                      # cli, ppo_core, categorical, checkpoint, nonfinite,
                               # logging, diag, metrics, visibility, bijepa(+config/probe),
                               # render_utils — без импорта симулятора на уровне модуля
  experiments/
    cube_wrench/               # reduced-куб: env(+gpu), logic, config, ppo_config,
                               # train, vec_env, eval, render, oracle
    coop_tasks/                # two-robot suite: base, cartpole, pendulum, logic, config,
                               # oracle, vec_env, eval  (toss&catch — следующий)
    arm/                       # обе линии SO-ARM101: scene/layout/turntable/expert(s),
                               # trainer, agent_obs, demos, bc; skills_* (skill-level),
                               # face_turn_* + bc (joint-level legacy)
  robots/so101_parallel_gripper.py
  assets/so101_parallel_gripper/
scripts/
  experiments/{cube_wrench,coop_tasks,arm}/   # smoke/probe/collect/GPU-grid скрипты
  analysis/                  # plot_wrench_curves, summarize_wrench_gpu
  ops/                       # setup_server, update_server, cloud_resume_loop, diagnose_vulkan
tests/
  common/                    # общие юнит-тесты (ppo/metrics/bijepa/visibility/...)
  experiments/{cube_wrench,coop_tasks,arm}/    # тесты по экспериментам
```

**Entry points:** `python -m callosum.experiments.cube_wrench.{train,eval,render}`, `python -m callosum.experiments.coop_tasks.eval`, `python -m callosum.experiments.arm.{trainer,bc,skills_eval}`, smoke-скрипты — `scripts/experiments/<experiment>/...`. План рефакторинга и финальное дерево: [`docs/refactor-plan.md`](docs/refactor-plan.md).

## Содержание

- [Идея](#идея)
- [Цель](#цель)
- [Задачи](#задачи)
- [Теоретическая часть](#теоретическая-часть)
- [Практическая часть](#практическая-часть)
- [Результаты](#результаты)
- [Выводы](#выводы)
- [Заключение](#заключение)

## Идея

Две робо-руки **SO-ARM101** (открытая 5-DOF рука, LeRobot / TheRobotStudio) с [параллельным захватом Robonine](https://github.com/roboninecom/SO-ARM100-101-Parallel-Gripper) — каждая как отдельный **агент** — учатся совместно собирать кубик Рубика, координируясь **децентрализованно**: у агента только своё наблюдение, без центрального контроллера и без явного канала связи на исполнении. Механизм координации — **Bi-JEPA**: каждый агент по своему наблюдению предсказывает *латентное состояние партнёра* (в духе Joint-Embedding Predictive Architecture) и на этом строит своё действие. «Bi» — оба агента симметрично моделируют друг друга.

Подробнее → [`docs/thesis/01-idea-goals.md`](docs/thesis/01-idea-goals.md).

## Цель

Разработать и экспериментально проверить **JEPA-подход к децентрализованной координации двух манипуляторов** (не-реконструктивное латентное предсказание партнёра) на задаче бимануальной манипуляции, с переносом sim2real на реальные SO-ARM101 — и показать его преимущество над реконструктивными/дистилляционными подходами к моделированию партнёра.

Обоснование и заострение вклада (область оказалась активной) → [`docs/thesis/02-related-work.md`](docs/thesis/02-related-work.md).

## Задачи

1. GPU-симуляция двух SO-ARM101 с параллельным захватом в ManiSkill; модель кубика (turntable-абстракция → гибридная артикуляция).
2. Децентрализованный MARL-бейзлайн (IPPO / MAPPO) на бимануальной задаче.
3. Bi-JEPA: энкодер наблюдений + предсказатель латента партнёра; интеграция в политику.
4. Ablation *oracle / predicted / no-partner* + сравнение с teammate-бейзлайнами (реконструктивный, дистилляционный).
5. Зрение (V-JEPA) и перенос sim2real на реальные SO-ARM101 (сейчас фокус только на симуляции, реальное железо скорее всего не используется).
6. Сведение результатов, выводы.

## Теоретическая часть

- Обзор литературы и позиционирование новизны → [`docs/thesis/02-related-work.md`](docs/thesis/02-related-work.md)
- Метод: формулировка Bi-JEPA + MARL-постановка → [`docs/thesis/03-method-bijepa.md`](docs/thesis/03-method-bijepa.md)

## Практическая часть

- Дизайн экспериментов (задача, среда, obs/action/reward, ablation, бейзлайны, метрики) → [`docs/thesis/04-experiment-design.md`](docs/thesis/04-experiment-design.md)
- Стек, железо, окружение → [`docs/thesis/05-stack-environment.md`](docs/thesis/05-stack-environment.md) и [`docs/setup.md`](docs/setup.md)

## Результаты

_TBD — по мере получения._

## Выводы

_TBD._

## Заключение

_TBD._
