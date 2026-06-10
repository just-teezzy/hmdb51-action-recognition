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

md("""## 7. ФАЗА 1 — мини-прогон на реальных данных (страховка ~минуты)
Убеждаемся, что данные распакованы в `<class>/*.avi`, PyAV декодит реальные
ролики, сплиты сходятся и train-цикл крутится на GPU. **Если выше `--check`
показал 0 clips у каких-то классов — остановитесь и пришлите вывод (поправим
фильтр), не запускайте полный прогон.**""")
code("""!python -m src.train --model tsn --epochs 1 --limit 40 --batch-size 4 --no-freeze
print("\\nФАЗА 1 ок -> можно запускать полный прогон (ячейка ниже).")""")

md("""## 8. ФАЗА 2 — полное обучение 5 моделей
Порядок **от надёжных к рискованным**: TSN → TSM → R(2+1)D → I3D → VideoMAE.
К моменту возможного OOM VideoMAE четыре модели уже обучены и сохранены.
Если VideoMAE падает — автоматически обучается запасная 5-я (**TimeSformer**),
так что в сравнении всегда ≥5 архитектур.

После каждой модели сразу считаются метрики (`evaluate`) → и чекпоинт, и
`results/` лежат на Drive. Если рантайм умрёт — перезапустите ячейки 1-6 и
эту: готовые модели не переобучаются с нуля (чекпоинты на Drive).

По умолчанию backbone заморожен (учится голова) — быстро и экономит память;
для полного дообучения добавьте `--no-freeze`.""")
code("""import subprocess, sys
EPOCHS = 15
ORDER = ["tsn", "tsm", "r2plus1d", "i3d", "videomae"]

def train_then_eval(model, train_extra=None):
    r = subprocess.run([sys.executable, "-m", "src.train", "--model", model,
                        "--epochs", str(EPOCHS)] + (train_extra or []))
    if r.returncode != 0:
        return False
    subprocess.run([sys.executable, "-m", "src.evaluate", "--model", model])
    return True

for m in ORDER:
    print("=" * 70, m)
    ok = train_then_eval(m)
    if not ok and m == "videomae":
        print("VideoMAE упал (вероятно OOM) -> запасная 5-я: TimeSformer")
        if not train_then_eval("timesformer"):
            print("TimeSformer тоже упал -> пробуем VideoMAE в tiny-режиме")
            train_then_eval("videomae", ["--tiny"])
print("\\nФАЗА 2 завершена.")""")

md("## 9. Сравнение и отчёт")
code("""!python -m src.compare
!python -m reports.generate_report
print("comparison + report готовы (results/, reports/out/ на Drive)")""")

md("""## 10. Скачать отчёт
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
