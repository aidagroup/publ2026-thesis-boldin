# Run card: FaceTurn-v0 scene + policy (JupyterHub terminal)

Companion to `docs/jupyterhub-runbook.md`. This card covers **only** the
FaceTurn-v0 path from a freshly unpacked `callosum.tar.gz` to the IPPO gate.

## 0. Prerequisites (once per pod)

Run the full JupyterHub setup first — it stages PhysX, fixes `libcuda.so`,
Vulkan, and writes `~/.callosum-env.sh`:

```bash
cd ~/callosum
bash scripts/setup_jupyterhub.sh
```

Then in **every** new terminal:

```bash
source .callosum-env.sh
```

> Никогда не запускайте `uv run` без этого `source` — иначе `uv` создаст
> второе `.venv` прямо в репозитории (и не хватит места в `$HOME`).

---

## 1. Prove the scene is solvable (the gate that matters)

The scripted expert lives in `callosum.envs._scripted_expert.PHASES` and is
read-only в чистоте: никакого RL внутри. Если он не решает задачу, ни одно
обучение поверх неё не даст.

```bash
cd ~/callosum
source .callosum-env.sh
uv run python scripts/probe_grasp.py --diag --every 10      # текущий режим: 32 envs, оба арми
```

`--diag` добавляет в конце фаз `hold`/`close` диагностику в локальном
фрейме куба — теперь с **полной ориентацией куба** (`drot`, как в
`FaceTurn.evaluate`, а не только `tilt`, который не видит кручение вокруг
вертикали) и **силой сжима пальцев** (точка входа `is_grasping` = 0.5 Н):

```
[diag hold]  holder TCP vs body_grasp_pos in cube frame (cm): x=+4.77 y=-13.04 z=+0.45
[diag hold]  sim tcp=[ 2.06 -0.73  3.83]cm fk=[ 2.06 -0.73  3.83]cm drift=[0. 0. -0.]cm (|d|=0.00cm)
[diag hold]  cube.mean=[-0.126 -0.042  0.029] (nominal [0. 0. 0.0285], drift [...], tilt=0.0 deg, drot=84.9 deg)
[diag hold]  gripper_qpos=-1.100 holder_force=(0.12/0.08)N rotator_force=(0.00/0.00)N held(holder)=0.00 grasped(rotator)=0.00
[diag hold]  verdict: cube spun 85 deg in yaw (drot) but FK==sim tcp (|d|=0.00cm) -- gripper ejected it; NOT a waypoint miss. held=0.00, holder_force=0.12/0.08 N (need >=0.5)
```

Читаем так (вердикт в последней строке -- истинный классификатор; строка `!!`
выше -- лишь сигнал к запуску `--diag`, а не приказ пере-решать):

- `grip is ON the handle` (residual < 1 mm) → геометрия правильная, виновата
  физика удержания: смотрите `holder_force` (нужно ≥0.5 Н **обеими** пальцями) и `drot`.
- `cube is BELOW spawn -- grip lost, body fell` → хват держится, но куб упал:
  слабая сила сжима / `BODY_GRASP_OFFSET`.
- `FRAME MISMATCH ... re-check _so100_kinematics vs URDF` → ЕДИНСТВЕННЫЙ случай
  `re-solve`: `sim tcp != fk tcp` (|d| > 3 см). Пере-решайте waypoints ТОЛЬКО здесь.
- `cube spun N deg in yaw (drot) ... gripper ejected it; NOT a waypoint miss` →
  куб **крутится вокруг вертикали** (и поэтому `tilt=0`): это НЕ ошибка фрейма
  (`sim tcp == fk tcp`), а куб не схвачен (`held=0`, `holder_force < 0.5 Н`) и
  отскакивает от закрывающихся пальцев. Править сцену, а не `_so100_kinematics`.
- `residual looks like a geometry/waypoint miss` → координаты сбились с
  сценой: пере-решайте `scripts/solve_waypoints.py` и
  `tests/test_grasp_waypoints.py`.

---

### 1.1 Устранение: когда `success=0` и `held=0`, `drot` растёт

Если §1 заканчивается `success=0` с вердиктом *gripper ejected it*: куб -- свободное
тело на столе, и закрывающие пальцы отгоняют его, а не схватывают. Делайте по
одному пункту -- каждый round-trip на сервер стоит.

**A. Изолируйте держатель** (убираем ротор из уравнения):

```bash
source .callosum-env.sh
uv run python scripts/probe_grasp.py --diag --every 1 --envs 1 --only holder
```

Если держатель **сам по себе** всё ещё крутит и соскальзывает куб
(`held=0`, `drot` растёт, `holder_force < 0.5 Н`) -- виновно схватывание
тела, а не столкновение с ротором. `--only` паркует другую руку на
`READY_QPOS` на всём прогре.

**B. Не перезакрывайте тело.** Держатель закрывает gripper до `-1.1`
(зазор чашек 0.66 см) на тело шириной 3.85–5.7 см -- перезакрытие ~2 см/палец,
от которого пальцы вышибают куб, а сила сжима не накапливается. В
`callosum/envs/_scripted_expert.PHASES` замените `GRIPPER_CLOSED` на
посадочное `qpos` для ширины тела **только в колонках держателя**; ротор
оставайтесь на `SEATING_GRIPPER_QPOS=-0.842` (→ фикс `-1.1` на nub). Точное
посадочное `qpos` для тела = `uv run python scripts/measure_gripper.py`
(раздел `report_clamp` → "the whole bare cube") на сервере: меши `so100`
там есть, а `measure_gripper` работает без симуляции, только с `.ply`.

**C. Сцена: трение и порядок закрытия.** Если (B) всё ещё скользит:
- поднимите `StaticFriction` стола в `callosum/envs/two_so100_base.py`
  (`TableSceneBuilder`) -- куб сейчас крутится на месте, значит статическое
  трение стола слишком мало, чтобы сопротивляться крутящему моменту пальцев;
- сделайте закрытие **адаптивным**: замыкать на `-1.1` ТОЛЬКО после
  `held >= 0.5` И `|hold→body| < WAYPOINT_TOLERANCE`, иначе держать открытым и
  дожидаться; текущие фиксированные бюджеты фаз (hold 30 / lift 40 и т.д.) не
  дожидаются контакта.

> Не пере-решайте waypoints и не трогайте `_so100_kinematics`, пока не увидите
> `FRAME MISMATCH` (|d| > 3 см). `sim tcp == fk tcp` в примере выше доказывает,
> что модель совпадает с симуляцией -- правка там не поможет.

Если визуально удобнее — телепортированные позы + запись:
```bash
source .callosum-env.sh   # Vulkan нужен даже для "none"-рендера, см. runbook
uv run python scripts/render_scene.py --script
```
картинки → `runs/render/`, GIF → `runs/render/expert.gif`.

---

## 2. Smoke-test обе среды (перед обучением)

```bash
uv run python scripts/smoke_env.py          # два SO-100 вокруг стола, reach-награда
uv run python scripts/smoke_face_turn.py     # turntable-cube + articulation + регистратор
```

---

## 3. IPPO (только после `success=1.0` в §1)

Грейд: `probe_grasp.py` кончается `success = 1.00` (и `held`/`grasped`
стабильны) на всех трёх фазах удержания/поворота.

```bash
cd ~/callosum
source .callosum-env.sh
mkdir -p runs
nohup uv run python -m callosum.training.ippo \
    --env-id FaceTurn-v0 --partner-input none --total-timesteps 50000 \
    > runs/ippo_faceturn_none.log 2>&1 &
echo "PID: $!"
tail -f runs/ippo_faceturn_none.log
```

Длинный прогон — detach через `nohup` (ячейка ноутбука умрёт вместе со
вкладкой; в терминале переживёт). Метрики пишутся прямо в лог (см.
JupyterHub runbook §6).

---

## 4. Артефакты

| Что | Где | Как забрать |
|---|---|---|
| Логи тренировки + TensorBoard scalars | `runs/<name>/events.out.tfevents.*` | `tar czf ~/runs.tar.gz --exclude='*.pt' runs/ && Download` |
| Веса | `runs/<name>/agent_*_ckpt_*.pt`, `bijepa_ckpt_*.pt` | отдельным `scp`/Upload (тяжело) |
| Рендер | `runs/render/` | `tar czf ~/render.tar.gz -C runs render` |

`runs/` уведена символической ссылкой на `$HOME/callosum-runs` — выдержит
переразвёртывание архива.
