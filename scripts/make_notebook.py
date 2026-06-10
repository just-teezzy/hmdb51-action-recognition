"""Generate notebooks/train.ipynb (Colab GPU training) as plain ipynb JSON.

Run:  python scripts/make_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(keepends=True)})


md("""# HMDB51 — распознавание действий: обучение в Colab (GPU)

Обучает **5 архитектур** (TSN, TSM, I3D, R(2+1)D, VideoMAE) на 10-классовом
подмножестве HMDB51. Данные — с **HuggingFace-зеркала** `jili5044/hmdb51`
(клипы уже .avi по папкам, RAR/unrar не нужны). Чекпоинты сохраняются на
**Google Drive каждую эпоху** (Colab отваливается по таймауту).

**Перед запуском:** Runtime → Change runtime type → **GPU**.""")

code("""import torch
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
!nvidia-smi -L""")

md("## 1. Google Drive (чтобы чекпоинты переживали перезапуск)")
code("""from google.colab import drive
drive.mount("/content/drive")""")

md("""## 2. Получить проект
Укажите URL вашего GitHub-репозитория (после git init / push).""")
code("""import os
REPO_URL = "https://github.com/USER/venya.git"  # <-- ЗАМЕНИТЕ
if not os.path.exists("/content/venya"):
    !git clone $REPO_URL /content/venya
%cd /content/venya""")

md("""## 3. Зависимости
Colab уже содержит CUDA-torch; ставим только недостающее.""")
code("""!pip -q install -r requirements-colab.txt
# unrar нужен ТОЛЬКО для запасного источника serre-lab:
!apt-get -qq install -y unrar > /dev/null 2>&1 || true""")

md("""## 4. Данные с HuggingFace-зеркала
Тянет только наши 10 классов и раскладывает в data/raw/<class>/*.avi.""")
code("""!python -m src.data.download_hmdb --source hf
!python -m src.data.download_hmdb --check""")

md("""## 5. Sanity-проверка на РЕАЛЬНЫХ .avi
Закрывает то единственное, что дал бы локальный прогон: реальный PyAV-декодер,
аугментации и сходимость сплитов.""")
code("""from src.data.splits import build_splits
from src.data.dataset import decode_video, sample_indices, transform_clip

sp = build_splits()
print("split source:", sp["_source"])
for k in ("train", "val", "test"):
    print(f"  {k:5s}: {len(sp[k])} clips")

for path, label in sp["test"][:3]:
    frames = decode_video(path)
    idx = sample_indices(len(frames), 8, train=False)
    clip = transform_clip(frames[idx], 112, train=True)
    print(os.path.basename(path), "| decoded", len(frames), "-> clip", tuple(clip.shape))
print("PyAV decode + augment + splits: OK")""")

md("""## 6. Чекпоинты и результаты → на Drive""")
code("""import pathlib, shutil
DRIVE = "/content/drive/MyDrive/venya"
for name in ("checkpoints", "results"):
    os.makedirs(f"{DRIVE}/{name}", exist_ok=True)
    p = pathlib.Path(name)
    if p.is_symlink():
        p.unlink()
    elif p.exists():
        shutil.rmtree(p)
    os.symlink(f"{DRIVE}/{name}", name)
print("checkpoints/ и results/ -> Drive")""")

md("""## 7. Обучение всех 5 моделей
По умолчанию backbone заморожен (учится только голова) — быстро и помещается в
память; чекпоинт пишется каждую эпоху. Если рантайм умрёт — перезапустите
ячейки 1-6 и снова эту (веса на Drive целы).

**VideoMAE** тяжёлый: при нехватке памяти добавьте `--tiny` или уменьшите batch.
Для полного дообучения backbone добавьте `--no-freeze`.""")
code("""EPOCHS = 15
for m in ["tsn", "tsm", "r2plus1d", "i3d", "videomae"]:
    print("=" * 70, m)
    !python -m src.train --model {m} --epochs {EPOCHS}""")

md("## 8. Оценка, сравнение, отчёт")
code("""for m in ["tsn", "tsm", "r2plus1d", "i3d", "videomae"]:
    !python -m src.evaluate --model {m}
!python -m src.compare
!python -m reports.generate_report
print("comparison + report готовы")""")

md("""## 9. Скачать отчёт
```python
from google.colab import files
files.download("reports/out/report.pdf")
```
Чекпоинты и метрики уже на Drive (MyDrive/venya/).""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "notebooks" / "train.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out} with {len(cells)} cells")
