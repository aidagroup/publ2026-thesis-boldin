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
фрейме куба:

```
[diag hold]  holder TCP vs body_grasp_pos in cube frame (cm): x=+0.12 y=-0.03 z=+0.01
[diag hold]  cube.mean=[ 0.001 -0.003  0.029 ] (nominal [0. 0. 0.0285], drift [...])
[diag hold]  held(holder)=0.42  grasped(rotator)=0.00
[diag hold]  verdict: grip lost, body fell
```

Читаем так:
- `verdict: grip is ON the handle` (residual < 1 mm) → геометрия правильная,
  виновато удержание: поднимайте μ контактов / `FACE_JOINT_DAMPING`.
- `verdict: cube is BELOW spawn -- grip lost, body fell` → хват затянулся, но
  скользит: смотрите `held`/силу сжима и `BODY_GRASP_OFFSET`.
- `verdict: residual looks like a geometry/waypoint miss` → координаты
  сбились с сценой: пере-решайте `scripts/solve_waypoints.py` и
  `tests/test_grasp_waypoints.py`.

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
