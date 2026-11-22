# Мост «learned reduced policy → arm skills» (предзащита, 2026-11-22)

Явный и измеримый мост между learned-политикой reduced-модели `CubeWrench-v0` и
skill-level линией SO-ARM101 `FaceTurnSkills-v0`. **Это не перенос весов**: пространства
наблюдений и действий разные, поэтому переносится только *намерение* (порядок фаз и роли
примитивов), а arm-часть исполняет его своими примитивами со своими длительностями.

Код: `callosum/experiments/arm/wrench_bridge.py` (rollout + сегментация + расписание +
метрики), `scripts/experiments/arm/bridge_wrench_to_skills.py` (прогон, исполнение на руках,
side-by-side демо), тесты `tests/experiments/arm/test_wrench_bridge.py` (32 unit-теста,
только macOS-safe pure-часть + helpers исполнителя).

## Reduced-экземпляр (learned `none`-политика, GPU `mujoco_warp`, 1 мир)

- чекпойнт `/workspace/callosum-runs/wrench_gpu_default_none_s0/latest.pt` (input mode `none`;
  чекпойнта `wrench_warp_none_*` на поде нет), seed `10000020` — выбран sweep'ом
  (`--sweep-reduced`) как экземпляр с явной фазой `hold` до `turn`;
- **success=True**, 25 control steps = **1.00 s**, финальный угол слоя **91.1°**,
  дрейф ядра **4.9 мм / 0.09°**;
- фазы: `idle` 4 шага (0.16 s) → **`hold` 2 шага** (0.08 s) → **`turn` 19 шагов** (0.76 s);
  в `hold` holder даёт момент ~1.4e-3 Nm при `rotator_tau_z ≈ 0`, в `turn`
  `rotator_tau_z` 0.7–1.7e-3 Nm и угол растёт 0→91°;
- трасса по control step: `{phase, t, holder_force, holder_torque, rotator_tau_z,
  layer_angle, intended_primitive, ...}` (физические силы/моменты восстановлены из
  applied-нормализованного wrench × authority × gain × limit_scale).

## Расписание примитивов (arm decision-time: 1 decision = 10 substeps × 0.05 s = 0.5 s)

| phase | decisions (`share`) | holder | rotator | источник |
| --- | --- | --- | --- | --- |
| idle (engage) | 8 | `approach_body` → `grasp` → `wait` | `approach_face` → `grasp` | arm-native (у reduced нет reach-эквивалента) |
| hold | 3 | `grasp` → `wait` | `grasp` → `wait` | reduced |
| turn | 30 | `wait` | `turn90` | reduced |
| release | 3 | `release` → `retreat` | `release` → `retreat` | arm-native (reduced-эпизод завершается на success) |

Наивный dt-вариант (`--timing-mode dt`) сжимает весь reduced-эпизод в **14 decisions**
(9 idle + 1 hold + 1 turn + 3 release) — на руках это неисполнимо: одна только
approach-нога примитива занимает десятки decisions. Это и есть численная мера
непереносимости абсолютного тайминга.

## Исполнение на руках (`FaceTurnSkills-v0`, fixed plan `U`, детерминированный эпизод)

План `U` = один поворот up-грани на 90° без флипа; выбран потому, что на поде
conductor на плане `U` даёт success 1.0, а план `R` (требует флипа) — 0.0. Плана `R` в
отчёте нет намеренно: его текущий провал — отдельный flip-баг линии skills, не мост.

- **исполнитель — scripted conductor** (skill-level верхняя граница, штатная валидация env):
  **success=True**, 32 decisions (**16.0 s**), финальный угол **89.7°**, план выполнен;
- **open-loop replay расписания** (тот же порядок примитивов, без re-grasp-логики):
  success=True, 30 decisions, **89.6°**; в более раннем прогоне (GPU-бэкенд, seed 0) replay
  застревал: 112 decisions, 89.0°, `plan_index=1`, но env success=False — потеря захвата на
  повороте без восстановления. То есть расписание несёт *намерение*, но не
  skill-level-восстановление — это и есть измеренная потеря моста, а не успех «моста целиком»;
- тайминг (arm-исполнение vs reduced-структура): `1 arm decision = 12.5 reduced control steps`;
  доля turn: reduced **0.76** против arm **0.06** (conductor) / 0.09 (replay);
  hold/turn-подпоследовательности: reduced 0.90 turn, arm 0.15 turn → **L1 1.50**;
  старт поворота: reduced на **25%** эпизода, arm на **94%** (offset **+0.69**): почти весь
  arm-эпизод уходит на arm-native engagement (approach+grasp), которого у reduced-политики
  нет по конструкции; длительность turn: reduced 0.76 s (19 control steps) против arm 1.0 s
  (2 decision — поворот на руках на порядки быстрее в decision-единицах).

## Что теряется при переносе (честно)

1. **Абсолютный тайминг не переносится**: 1 decision = 12.5 reduced steps; dt-расписание
   даёт 14 decisions на весь эпизод при потребности в десятках.
2. **Engagement (reach+grasp) — arm-native**: у reduced-агента нет понятия «дотянуться»; это
   ~56% decisions arm-эпизода и главный источник offset старта поворота.
3. **Skill-level восстановление не выводится из reduced-трассы**: re-grasp после потери
   захвата/слипа — свойство env+conductor, поэтому open-loop replay может застрять, тогда как
   conductor доходит до success.
4. **turn-collapse**: поворот на руках укладывается в 2–3 decisions (45°→72°→89° за decision),
   тогда как reduced-политика поворачивает 19 control steps; поэтому доли фаз почти не
   совпадают при обоюдном успехе.
5. **Физика не общая**: у reduced-модели нет рук/контактов/гравитации, сравниваются только
   нормализованные структуры фаз, не физические величины.

## Артефакты

- `demo/bridge/bridge_wrench_to_skills.mp4` — side-by-side slow-mo (~19.7 s, 10 fps,
  1280×576): слева reduced learned policy (EGL-рендер трассы, камера приближена), справа две
  SO-ARM101 (SAPIEN/lavapipe) исполняют те же намерения примитивами. Правая сцена: close-up
  камера, из multi-face-куба показаны только ядро + активный слой (остальные slabs скрыты
  render-only), на слое — бирюзовая и пурпурная полосы (вращаются со слоем), на столе —
  жёлтая полоса-референс; 90° поворот читается. HUD: `LEARNED REDUCED POLICY → ARM SKILLS`,
  dial 0–90° с двумя стрелками (reduced слой / arm грань), таймлайн HOLD→TURN→RELEASE,
  intended vs realized primitives, onset offset, SUCCESS.
- `demo/bridge/bridge_wrench_to_skills_preview.gif`, `bridge_{engage,hold,turn,success}.png` —
  превью и стиллы ключевых фаз (turn-стилл: бирюзовая полоса повёрнута на ~90° относительно
  жёлтого референса).
- `demo/bridge/bridge_metrics.json` — полный JSON: reduced-трасса, фазы, расписание (share+dt),
  arm-трасса conductor и replay, тайминг-метрики, limitations, CLI.
- `demo/bridge/bridge_metrics_blocked_pre_fix.json` — исторический BLOCKED-прогон до фикса env
  (arm env не был здоров: conductor 0/150 decisions, угол 0°) — оставлен как след диагностики.

## Запуск (под, через `scripts/ops/pod_job.sh`)

```bash
# полный мост + демо (1 world reduced на CUDA, arm на CPU-бэкенде + lavapipe EGL)
scripts/ops/pod_job.sh start bridge-demo -- bash -lc \
  'cd /root/callosum-bridge && PYTHONPATH=/root/callosum-bridge /root/.callosum/venv/bin/python \
   scripts/experiments/arm/bridge_wrench_to_skills.py \
   --checkpoint /workspace/callosum-runs/wrench_gpu_default_none_s0/latest.pt \
   --seed 10000020 --sim-backend cpu --render-backend cpu --render \
   --plan-notation U --out /workspace/callosum-runs/bridge/bridge_metrics.json'
# только reduced-часть (работает без здорового arm env)
... --reduced-only --sweep-reduced 10000019,10000020,10000021 ...
```

Ограничение: reduced-сторона — один greedy-эпизод на фиксированном seed (экземпляр-образец,
не статистика); arm-сторона — один детерминированный эпизод фиксированного плана.
