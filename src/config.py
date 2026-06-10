"""Central project configuration: paths, class list, seed, default hyperparameters.

Everything that the rest of the codebase needs to agree on lives here so that
training, evaluation, the demo and the report all use the same definitions.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Paths (all relative to the repository root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # extracted HMDB51 .avi files, one folder per class
SPLIT_DIR = DATA_DIR / "splits"      # official HMDB51 split .txt files
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"       # per-model metrics .json + figures
REPORT_DIR = ROOT / "reports" / "out"
DEMO_DB = ROOT / "demo" / "history.db"

for _d in (DATA_DIR, RAW_DIR, SPLIT_DIR, CHECKPOINT_DIR, RESULTS_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset: 10-class subset of HMDB51.
# Mostly distinct actions, but two deliberately CONFUSABLE pairs are included so
# the error analysis / confusion matrix is meaningful (course requirement):
#   * run  <-> walk            (both locomotion, differ mainly in speed)
#   * fencing <-> draw_sword   (both sword-in-hand arm motions)
# ---------------------------------------------------------------------------
CLASSES = [
    "climb",
    "draw_sword",
    "fencing",
    "golf",
    "pullup",
    "ride_bike",
    "run",
    "shoot_bow",
    "walk",
    "wave",
]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------------------
# Input / preprocessing defaults
# ---------------------------------------------------------------------------
NUM_FRAMES = 16          # frames sampled per clip (training default)
IMG_SIZE = 112           # spatial resolution after crop
# ImageNet / Kinetics normalisation (shared by all backbones)
MEAN = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)

# ---------------------------------------------------------------------------
# Training defaults (per-model overrides live in MODEL_CONFIGS)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    epochs=15,
    batch_size=8,
    lr=1e-4,
    weight_decay=1e-4,
    num_frames=NUM_FRAMES,
    img_size=IMG_SIZE,
    freeze_backbone=True,
    num_workers=2,
)

# Registered model keys (the 5 families). Each entry can override DEFAULTS.
MODEL_CONFIGS = {
    "tsn":      dict(img_size=112, num_frames=8),
    "tsm":      dict(img_size=112, num_frames=8),
    "i3d":      dict(img_size=112, num_frames=16),
    "r2plus1d": dict(img_size=112, num_frames=16),
    "videomae": dict(img_size=224, num_frames=16, batch_size=4, lr=5e-5),
}
MODEL_NAMES = list(MODEL_CONFIGS.keys())


def get_config(model_name: str) -> dict:
    """Return effective hyper-parameters for a model (DEFAULTS + overrides)."""
    cfg = dict(DEFAULTS)
    cfg.update(MODEL_CONFIGS.get(model_name, {}))
    cfg["model"] = model_name
    return cfg
