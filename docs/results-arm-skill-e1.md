# SO-ARM101 skill-level: один поворот learned-политикой (E1), 2026-11-22

## Что сделано

- **Починены 4 реальных бага skill-линии** (`FaceTurnSkills-v0`): SAPIEN `Pose.q` wxyz vs
  xyzw в obs/контракте; `is_grasping` не мог быть True на 6-слойном кубе (губки жмут РАЗНЫЕ
  слои) → добавлены `_jaw_press`/`_grasp_any`; `torch.flatnonzero` не существует; инвертирован
  знак roll во flip-макросе.
- **Plan `U` (один поворот на 90°) проходит scripted conductor**: success **1.000**, 33 decisions,
  финальный угол **88.5°** (CPU PhysX).
- **Learned BC-политика (E1)**: собраны 24/24 успешных conductor-демо (только success), обучена
  категориальная BC (obs builder + `CategoricalPolicy`), лучший val acc **0.977/0.967**.
- **Learned-демо**: `demo/arms_face_turn/demo_arms_learned_turn.mp4` — success=True, 37 decisions,
  13.5 s, финальный угол **90.0°**, drift **0.26 cm**, HUD `POLICY (learned, BC)` +
  `GOAL: turn face U by 90°`; JSON с checkpoint `runs/skills_bc_U/bc.pt`.

## Честные оговорки

- Aggregate eval BC: **0.719 success на 64 эпизодах** (timeout 0.281, decisions 68.7; в успешных
  эпизодах 36.8 decisions) — честный per-episode-reset протокол на merged `main`. Первый
  «1.000/64» был загрязнён harness-багом (raw ManiSkill episode без auto-reset) и НЕ используется.
  JSON: `results/skills_e1/honest_eval_64.json`.
- Learned-демо: success=True, 37 decisions, 90.0°, drift 0.31 см; HUD с close-up и маркером
  активного слоя. Это один успешный эпизод, не заменяет aggregate 0.719.
- **Sexy move `R U R' U'` conductor-ом заблокирован на post-flip drift**: после фиксов (numpy
  `np.isin` frozenset, proximity-grasp gate для turn_assist, flip-home leg) `R` **завершает
  поворот** (plan_index 1/1, 89.9–90.0°, 1 флип), но env success=False: drift **3.9 см** при
  `body_pos_tol=0.03` (при 0.05 — 5.6 см; уменьшение lift ломает флип). Полный `R U R' U'` —
  2/6 мувов за 600 решений и 3 флипа. Это честный BLOCKED на drift/hand-off, не на plan/quaternion.
  Learned sexy move не делали.
- Bi-JEPA на skill-level **не измерен**: в skills-тренере `predicted` — zero-stub (предиктор не
  подключён). Никаких чисел не выдумывали.
- CPU-симуляция + software Vulkan (llvmpipe) для рендера; GPU-sim путь не проверялся.

## Мост reduced → arms

`docs/results-wrench-bridge.md`, демо `demo/bridge/`: learned CubeWrench policy (success, 91.1°)
→ hold/turn расписание → руки исполняют U-поворот 89.7°; измерены потери (абсолютный тайминг,
reach/grasp-этап, re-grasp). Это intent→skills bridge, не перенос весов.

## Следующие шаги

1. Честный BC eval aggregate на merged tree.
2. Фикс flip hand-off → conductor sexy move → conductor-демо.
3. Learned sexy move (BC/IPPO) после стабильного conductor.
4. Подключить Bi-JEPA `predicted` в skill-тренер и измерить next-primitive/probe.
