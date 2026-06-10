# Распознавание действий человека на видео (Вариант 19)

Система определяет тип действия по короткому видеоролику. Обучаются и сравниваются
**5 архитектур из разных семейств**, лучшая внедряется в демо-приложение (Streamlit).

> **Окружение:** локального GPU нет → тяжёлое обучение выполняется в **Google Colab**
> (ноутбук [`notebooks/train.ipynb`](notebooks/train.ipynb)), а локально на CPU доступны
> инференс, демо и весь сервисный пайплайн. Код един для обоих режимов.

## Датасет
- **HMDB51**, подмножество из **10 классов**: `climb, draw_sword, fencing, golf,
  pullup, ride_bike, run, shoot_bow, walk, wave` (~150 роликов на класс, ~1500 всего).
  В набор намеренно включены **похожие пары** (`run`/`walk`, `fencing`/`draw_sword`),
  чтобы анализ ошибок и confusion matrix были содержательными.
- **Разметка:** один ярлык-действие на ролик (video-level classification).
- **Источник данных:** HuggingFace-зеркало
  [`jili5044/hmdb51`](https://huggingface.co/datasets/jili5044/hmdb51) — клипы уже
  распакованы в `.avi` по папкам-классам (RAR/unrar не нужны). Исходный serre-lab
  (RAR-архивы) оставлен как fallback (`--source serre-lab`).
- **Сплиты:** если присутствуют официальные `testTrainMulti_7030_splits` — берём их;
  иначе строим **свой стратифицированный split с фиксированным seed** (70/30,
  детерминированно). Валидация в обоих случаях выделяется из train (15%,
  стратифицированно), т.к. официального val у HMDB51 нет.
- **Лицензия/ограничения:** HMDB51 предназначен **только для исследовательских/учебных
  целей**; ролики взяты из публичных источников и фильмов. Датасет в репозиторий **не**
  коммитится (см. `.gitignore`). Код проекта — под лицензией MIT ([LICENSE](LICENSE)).

## 5 архитектур (разные семейства)
| # | Модель | Семейство | Источник весов | Реализация |
|---|--------|-----------|----------------|-----------|
| 1 | **TSN** | 2D-CNN + temporal pooling | ResNet-18 / ImageNet | вручную ([src/models/tsn.py](src/models/tsn.py)) |
| 2 | **TSM** | 2D-CNN + temporal shift | ResNet-18 / ImageNet | вручную ([src/models/tsm.py](src/models/tsm.py)) |
| 3 | **I3D** (slow_r50 fallback) | 3D-CNN | Kinetics-400 (pytorchvideo) | [src/models/i3d.py](src/models/i3d.py) |
| 4 | **R(2+1)D** | факторизованный 3D-CNN | Kinetics-400 (torchvision) | [src/models/r2plus1d.py](src/models/r2plus1d.py) |
| 5 | **VideoMAE** (TimeSformer fallback) | трансформер | Kinetics-400 (transformers) | [src/models/videomae.py](src/models/videomae.py) |

Все модели имеют единый контракт: `forward(x[B,T,C,H,W]) -> logits[B,num_classes]`
(см. [src/models/registry.py](src/models/registry.py)), поэтому train/eval/demo не
зависят от конкретной архитектуры.

## Структура
```
src/
  config.py            seed, классы, пути, гиперпараметры
  utils.py             seeding, чекпоинты, метрики, размер модели
  data/                dataset (PyAV + синтетика), splits, download_hmdb
  models/              tsn, tsm, i3d, r2plus1d, videomae, registry
  train.py             единый скрипт обучения (--model ...)
  evaluate.py          метрики, confusion, latency, стабильность
  compare.py           сводная таблица + графики
  inference.py         инференс одного ролика (для демо)
demo/
  app.py               Streamlit-приложение
  history.py           SQLite-история запусков
reports/
  generate_report.py   авто-отчёт Excel + PDF
notebooks/train.ipynb  обучение на GPU в Colab
scripts/smoke_test.py  сквозной дымовой прогон на синтетике
```

## Установка (локально, CPU, WSL/Linux, Python 3.10)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Запуск
**Дымовой прогон** (проверка всего пайплайна на синтетике, без скачивания данных):
```bash
python scripts/smoke_test.py
```
**Данные** (HMDB51 subset, HuggingFace-зеркало — без RAR/unrar):
```bash
python -m src.data.download_hmdb              # источник HF (по умолчанию)
python -m src.data.download_hmdb --check
# fallback на оригинал (нужен unrar):
#   python -m src.data.download_hmdb --source serre-lab
```
**Обучение** (полноценно — в Colab на GPU, см. [notebooks/train.ipynb](notebooks/train.ipynb);
локально — только малые прогоны):
```bash
python -m src.train --model r2plus1d --epochs 15
```
**Оценка и сравнение:**
```bash
python -m src.evaluate --model r2plus1d
python -m src.compare
python -m reports.generate_report
```
**Демо:**
```bash
streamlit run demo/app.py
```

## Воспроизводимость
Seed зафиксирован (`SEED=42`, [src/config.py](src/config.py)), версии пакетов — в
[requirements.txt](requirements.txt). Чекпоинт сохраняется после каждой эпохи
(важно для Colab, который отваливается по таймауту).

## Результаты
Сводная таблица — `results/comparison.md` / `.csv`, графики — `results/comparison.png`,
отчёт — `reports/out/report.pdf` (генерируются после обучения и оценки всех моделей).

_Раздел с финальными числами и выводом о лучшей архитектуре будет заполнен после
полного цикла обучения в Colab._
