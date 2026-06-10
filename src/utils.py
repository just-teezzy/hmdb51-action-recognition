"""Shared utilities: seeding, logging, checkpoints, metrics, model size."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility & device
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "venya") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Running average
# ---------------------------------------------------------------------------
class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(path, model, optimizer=None, epoch=0, extra=None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optim_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt


# ---------------------------------------------------------------------------
# Model size
# ---------------------------------------------------------------------------
def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model) -> float:
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return n_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                    class_names: Sequence[str]) -> Dict:
    """Accuracy / precision / recall / F1 (macro + per-class) + confusion matrix."""
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                  confusion_matrix)
    labels = list(range(len(class_names)))
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pp, pr, pf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(acc),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "per_class": {
            class_names[i]: {
                "precision": float(pp[i]),
                "recall": float(pr[i]),
                "f1": float(pf[i]),
            } for i in labels
        },
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def save_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


class Timer:
    """Context manager measuring wall-clock seconds."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
