# 05 – Стек и окружение

## Технологический стек

| Слой            | Инструмент                                 | Роль                                                |
| --------------- | ------------------------------------------ | --------------------------------------------------- |
| Симуляция       | **ManiSkill 3** (SAPIEN, GPU-параллельный) | среда, 2× SO-100, рендер                            |
| MARL            | **TorchRL + BenchMARL**                    | IPPO / MAPPO (+ кастомный Bi-JEPA-модуль)           |
| Представления   | **V-JEPA / V-JEPA 2** (Meta)               | энкодер зрения (фаза 2), предобученный/замороженный |
| Реальное железо | **LeRobot** (HuggingFace)                  | управление SO-100, телеоп, датасеты, деплой         |
| API среды       | **Gymnasium / PettingZoo**                 | стандартные интерфейсы                              |

### Почему не Stable-Baselines3

SB3 отброшен: 
1) он **single-agent** – MARL из коробки нет;
2) numpy-ориентирован и **не использует GPU-векторизацию ManiSkill** (лишние CPU↔GPU трансферы). TorchRL – PyTorch-native, GPU-first, легко встроить кастомный Bi-JEPA-модуль; BenchMARL даёт готовые IPPO/MAPPO и когерентен экосистеме Meta (V-JEPA).

### Как связаны ManiSkill и BenchMARL

ManiSkill 3 даёт нативную мультиагентность (dict action space по агенту, PettingZoo-совместимо), BenchMARL нативно поддерживает PettingZoo. Нюанс: ManiSkill-PettingZoo **GPU-векторизован**, обычная обёртка TorchRL ждёт CPU ParallelEnv → возможен мостик по векторизации; fallback – сырой TorchRL multi-agent (бэкенд BenchMARL), чтобы остаться на GPU.

## Железо

- **1× NVIDIA A100-SXM4-80GB** (Ampere) на сервере лаборатории; драйвер 570.172.08 (CUDA ≤ 12.8). Одна большая карта лучше нескольких мелких: ManiSkill крутится на одном GPU, single-node RL не шардируется тривиально.
- Бюджет VRAM: state-based – 1000+ параллельных сред; vision (RGB-рендер) – тяжелее (~сотни сред), фаза 2. Замороженный V-JEPA-энкодер ~1–2GB на инференс. 80GB хватает с запасом и на vision-фазу.
- Интернет на сервере – по белому списку (GitHub, PyPI, PyTorch, Hugging Face и др.); `wandb.ai` недоступен, поэтому метрики пишутся локально (TensorBoard в `runs/`).

## Окружение (воспроизводимость)

- **Python 3.12** (пин; SAPIEN не имеет колёс под 3.13+).
- Менеджер – **uv**; `uv.lock` и `.python-version` в git.
- PyTorch: **cu128** на Linux (потолок драйвера сервера – CUDA 12.8), CPU/MPS на macOS – авто через `pyproject.toml` (`[tool.uv.sources]`).
- Зависимости по extra-группам: `sim` (ManiSkill, Linux-only) / `train` (torch, tensordict, torchrl, benchmarl) / `dev`.
- Залоченные версии (2026-07): mani-skill 3.0.1 (sapien 3.0.3), torch 2.11.0+cu128, torchrl 0.11.1, benchmarl 1.5.2, tensordict 0.11.0, pettingzoo 1.26.1, gymnasium 1.3.0.

## Машины и деплой

- **Dev (macOS):** правка кода, линт; **без GPU-сима** (ManiSkill GPU – только Linux+CUDA).
- **Train (сервер лаборатории, Linux + A100 80GB):** сим + обучение. Доступ **только через веб-интерфейс JupyterHub и его терминал** (SSH нет, root нет); образ `jupyter/singleuser-gpu_570`, пользователь `jovyan`. Исходящий интернет – белый список (GitHub, PyPI, download.pytorch.org, astral.sh, Hugging Face, LLM-прокси лаборатории; `wandb.ai` закрыт).
- **Диск:** `$HOME` крошечный (4 ГБ, свободно < 1 ГБ), зато оверлей `/tmp` большой (десятки ГБ) и считается очищаемым при перезапуске контейнера. Поэтому клон репозитория, venv и кэши (uv, Hugging Face, ассеты ManiSkill/SAPIEN) лежат в scratch `/tmp/$USER-callosum`, а результаты `runs/` – в `$HOME` (символическая ссылка), чтобы пережить перезапуск; чекпоинты приходится чистить и скачивать.
- **Деплой-петля:** правишь локально → `git push` → в терминале JupyterHub `git clone`/`git pull` + `bash scripts/setup_server.sh` (идемпотентен, после очистки scratch запускается заново) → обучение отсоединённым процессом (`setsid nohup`: ядро ноутбука умирает при закрытии вкладки) → результаты забираются через файловый браузер JupyterHub (ни `ssh`, ни `rsync` нет); TensorBoard – через JupyterHub-прокси, если он есть, иначе локально по скачанным логам.

Подробности сетапа и bootstrap-скрипт сервера – [`../setup.md`](../setup.md).
