# Канал статусов под→агент (без sleep-поллинга)

Проблема: под — отдельная машина, push-уведомлений от неё нет; агент либо спит в цикле, либо
тратит раунды на `sleep && ssh`. Решение — **durable status channel** из двух частей:

- **на поде**: `scripts/ops/jobctl.py` — запускает задачу detached, пишет
  `/workspace/callosum-jobs/<job>.state.json` (status/pid/exit/timestamps), лог
  `/workspace/callosum-jobs/<job>.log` и строку в `events.jsonl`; watchdog фиксирует завершение.
- **на маке**: `scripts/ops/pod_job.sh` — тонкая SSH-обёртка; `wait` держит один поток и
  возвращается ровно по событию `finished`, опционально стримит лог (`--follow`).

## Как пользоваться

```bash
# 0) один раз после поднятия пода: скопировать jobctl
scripts/ops/pod_job.sh install

# 1) запустить задачу на поде (detached, переживает обрыв SSH)
scripts/ops/pod_job.sh start tess-occl-250 -- \
  /root/.callosum/venv/bin/python -m callosum.experiments.cube_wrench.train \
  --task toss --backend warp --device cuda --num-envs 256 --total-updates 250 \
  --partner-mode none --exp-name gpu_occ_toss_none_s0_250

# 2) статус — один SSH round-trip, без циклов
scripts/ops/pod_job.sh status tess-occl-250

# 3) ждать завершения и стримить лог; запускать как DSH background job
scripts/ops/pod_job.sh wait tess-occl-250 --follow
```

`wait` — блокирующий вызов: DSH сам уведомит о завершении фонового job'а, когда задача
закончится. Никаких `sleep` в агенте: максимум один фоновый процесс на ожидание.

Для задач, запущенных не через `jobctl` (например, старый grid-скрипт), есть ожидание по маркеру:

```bash
scripts/ops/pod_job.sh wait-file /root/gpu_toss_phys_grid.log --pattern GPU_PHYS_DONE
```

## Детали

- Состояние и логи лежат на network volume (`/workspace/callosum-jobs`), поэтому переживают
  перезапуск контейнера; `jobctl status` покажет их и после рестарта.
- `jobctl` не требует установки зависимостей (stdlib Python 3).
- `start` выставляет `WRENCH_TORCH_THREADS/OMP/MKL=1` по умолчанию (как у GPU-first прогонов).
- `wait --timeout N` возвращает управление, если задача не уложилась в N секунд.
- Переменные подключения: `POD_HOST`, `POD_PORT`, `POD_KEY` (по умолчанию текущий A40-под).
