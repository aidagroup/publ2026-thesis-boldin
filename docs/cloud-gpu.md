# Cloud GPU (Nebius) — ранбук

> Статус на 2026-11-14: конфигурация зафиксирована, VM создаётся в консоли Nebius.
> Этот файл — первое руководство после получения SSH-доступа. Детали установки,
> диагностики и smoke-проверок: [server-runbook.md](server-runbook.md).

## Конфигурация (зафиксировано 2026-11-14)

| Параметр | Значение |
| --- | --- |
| Провайдер | Nebius AI Cloud |
| Регион | `eu-north1` |
| Тип VM | **Preemptible** (spot, биллинг per-second) |
| GPU | NVIDIA **L40S**, 48 ГБ VRAM |
| Пресет | 1 GPU — 16 vCPU — 96 GiB RAM |
| Диск | Network SSD, **200 GiB** (дефолтные 1280 GiB уменьшены: наш data root ≤ 100 ГБ; размер диска можно только увеличивать) |
| Образ | `ubuntu24.04-cuda12` — Ubuntu 24.04 LTS for NVIDIA GPUs (CUDA 12), драйвер предустановлен. Не `driverless`, не `serverless` |
| Запасной вариант | Preemptible H100 (~$2.15/ч по калькулятору) — если L40S не хватит |

Обоснование: обучение state-based (без рендера, `render_backend="none"`), политики — MLP; H100/B200 избыточны. Spot-цена может меняться в течение часа — задайте max price при создании.

## Первый запуск по SSH

```bash
ssh <username>@<VM_PUBLIC_IP>
```

1. **Драйвер.** `nvidia-smi` — ожидаем в выводе `NVIDIA L40S`.
2. **Клон репозитория:**

   ```bash
   git clone -b step/2.3-skill-level-marl https://github.com/aidagroup/publ2026-thesis-boldin.git ~/callosum
   cd ~/callosum
   ```

3. **Установка + серверная верификация storage-скриптов.** Fresh install:

   ```bash
   bash scripts/setup_server.sh --smoke
   ```

   Это одновременно **первая серверная проверка** storage-слоя (до этого скрипты
   проходили только локальный `bash -n`). Повторные синхронизации с веткой:

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

## Preemption и авто-рестарт

Preemptible VM может быть прервана платформой в любой момент. Тренер переживает
это по построению: на SIGTERM он завершает текущую итерацию, делает оценку и
пишет `latest.pt` (чекпоинт каждые 20 итераций); `--resume runs/<name>`
продолжает обучение точно (веса, Adam-моменты, RNG, LR-расписание, счётчики).

Порядок работы с длительным прогоном:

1. Запустить прогон обычной командой тренера (см. detached-паттерн в
   [server-runbook.md](server-runbook.md)) — так появится
   `runs/<run_name>/latest.pt`.
2. Дальше держать его под циклом авто-рестарта в `tmux` (или `screen`):

   ```bash
   tmux new -s train
   bash scripts/cloud_resume_loop.sh <run_name> [ippo args...]
   # Ctrl-b d — detach; tmux attach -t train — вернуться
   ```

Поведение цикла: exit 0 тренера — чистое завершение (total-timesteps
достигнут), цикл выходит; ненулевой код (включая убийство preemption-ом) —
запись в `runs/<run_name>/resume_loop.log` и рестарт через 30 с.

## Стоимость и гигиена

- **Выключайте VM**, когда обучение не идёт: биллинг per-second, простой GPU
  сжигает бюджет. Диск при остановке сохраняется (при 200 GiB — копейки) и
  данные не теряются.
- Следите за spot-ценой и max price в консоли.
- Важное храните только в `~/callosum` (репо) и `~/callosum-runs`
  (чекпоинты/видео за симлинком `runs/`); всё остальное на VM считайте
  эфемерным.
