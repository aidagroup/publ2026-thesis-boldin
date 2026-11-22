# Cloud GPU (RunPod) — ранбук

> Статус на 2026-11-14: провайдер сменён с Nebius на RunPod. Ранбук — первое
> руководство после создания GPU Pod. Детали установки, диагностики и
> smoke-проверок: [server-runbook.md](server-runbook.md).

## Конфигурация (зафиксировано 2026-11-14)

| Параметр | Значение |
| --- | --- |
| Провайдер | RunPod, **GPU Pod** (не serverless) |
| GPU | NVIDIA **RTX 4090, 24 ГБ** — state-based симуляция + MLP-политики; L40S/A100 — опция на будущее (VLA весной) |
| Тип оплаты | **On-Demand** по умолчанию (стабильность до предзащиты); **Spot** — для дешёвых длинных прогонов (цены проверить в консоли при создании) |
| Container disk | ~30–50 ГБ: репо `~/callosum`, venv и кэши `~/.callosum` |
| Network volume | ~100 ГБ, **тот же регион**, монтируется в `/workspace`: чекпоинты/демо/видео, переживает удаление пода |
| Образ | официальный RunPod PyTorch-шаблон с CUDA 12.x (точный тег выбрать в консоли; драйвер предустановлен) |
| SSH | публичный ключ добавить в Settings → SSH Keys **до** создания пода; команда подключения — в карточке пода (Connect → SSH), вида `ssh root@<host> -p <port>` |

## Первый запуск по SSH

После создания пода вписать хост/порт из карточки Connect в `~/.ssh/config`
(хост `runpod-gpu`) и:

```bash
ssh runpod-gpu
```

0. **Разметка хранилища** (однократно): направить дефолтный путь runs-хранилища
   скриптов на network volume — после этого `CALLOSUM_RUNS` не нужен:

   ```bash
   mkdir -p /workspace/callosum-runs
   ln -sfn /workspace/callosum-runs ~/callosum-runs
   ```

1. **Драйвер.** `nvidia-smi` — ожидаем в выводе `NVIDIA GeForce RTX 4090`.
2. **Клон репозитория** (на container disk):

   ```bash
   git clone -b step/2.3-skill-level-marl https://github.com/aidagroup/publ2026-thesis-boldin.git ~/callosum
   cd ~/callosum
   ```

3. **Установка + серверная верификация storage-скриптов.** Fresh install:

   ```bash
   bash scripts/setup_server.sh --smoke
   ```

   Это одновременно **первая серверная проверка** storage-слоя (до этого
   скрипты проходили только локальный `bash -n`). Повторные синхронизации:

   ```bash
   bash scripts/update_server.sh -- --smoke
   ```

4. **GPU smoke-прогоны** — в порядке возрастания (ожидания и диагностика — в
   [server-runbook.md](server-runbook.md)):

   ```bash
   uv run python scripts/smoke_env.py
   uv run python scripts/smoke_face_turn.py
   uv run python scripts/probe_face_turn.py
   ```

5. **Отчёт домой:** успех/провал каждого шага, полный вывод `nvidia-smi`,
   тайминги установки и прогонов. После этого storage-слой считается
   верифицированным и `main` можно размораживать (по решению основного потока).

## Preemption и персистентность

- **Что переживает что:** network volume (`/workspace`) живёт всегда;
  container disk — пока под не удалён. Поэтому чекпоинты и артефакты — только
  через `~/callosum-runs` (симлинк на volume); репо и venv на container disk —
  их потеря при удалении пода дёшево чинится повторным клоном и setup.
- **Spot:** под может быть остановлен платформой. GPU-часы дешевле, но после
  остановки под надо запустить снова и перезапустить цикл обучения (ниже) —
  `--resume` продолжит точно (`latest.pt` каждые 20 итераций). Для
  дедлайн-критичных прогонов (декабрь) — On-Demand.
  `TODO(review)`: проверить при первом spot-прогоне, как именно приходит
  прерывание и есть ли авто-рестарт пода.
- **Цикл авто-рестарта** (`scripts/cloud_resume_loop.sh`) — для прерываний на
  уровне процесса:

  ```bash
  # 1) запустить прогон обычной командой тренера (см. detached-паттерн в
  #    server-runbook.md), дождаться появления runs/<run_name>/latest.pt
  # 2) держать его под циклом в tmux:
  tmux new -s train
  bash scripts/cloud_resume_loop.sh <run_name> [ippo args...]
  # Ctrl-b d — detach; tmux attach -t train — вернуться
  ```

  Поведение: exit 0 тренера — чистое завершение (total-timesteps), цикл
  выходит; ненулевой код — запись в `runs/<run_name>/resume_loop.log` и
  рестарт через 30 с. Сигналы форвардуются тренеру, чтобы успел дописаться
  `latest.pt`.

## Стоимость и гигиена

- **Останавливать под**, когда обучение не идёт: GPU оплачивается только пока
  под работает; storage (container disk + volume) тарифицируется отдельно —
  сверить в консоли.
- Region volume = region пода (иначе не прицепится / латентность).
- Важное хранить в `~/callosum` (репо; изменения коммитить и пушить!) и на
  `/workspace` (артефакты). Остальное в поде считать эфемерным.
