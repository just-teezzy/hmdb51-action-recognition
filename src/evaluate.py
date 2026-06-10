"""Evaluate a trained checkpoint on the test split.

Produces:
  results/<model>_metrics.json   accuracy / precision / recall / F1 (+per-class),
                                 latency, throughput, stability, model size
  results/<model>_confusion.png  confusion-matrix heatmap

Examples
--------
    python -m src.evaluate --model tsn --synthetic --tiny      # smoke
    python -m src.evaluate --model r2plus1d                     # real test split
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.data.build import get_datasets
from src.models.registry import build_model
from src.utils import (compute_metrics, count_parameters, get_device, get_logger,
                       load_checkpoint, model_size_mb, save_json)

log = get_logger()


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    y_true, y_pred, confidences = [], [], []
    for clips, labels in loader:
        clips = clips.to(device)
        probs = torch.softmax(model(clips), dim=1)
        conf, pred = probs.max(dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(pred.cpu().tolist())
        confidences.extend(conf.cpu().tolist())
    return y_true, y_pred, confidences


@torch.no_grad()
def measure_latency(model, dataset, device, n: int = 20):
    """Mean wall-clock seconds to classify one clip (batch size 1, CPU-realistic)."""
    model.eval()
    n = min(n, len(dataset))
    # warm-up
    clip0, _ = dataset[0]
    model(clip0.unsqueeze(0).to(device))
    t0 = time.perf_counter()
    for i in range(n):
        clip, _ = dataset[i]
        model(clip.unsqueeze(0).to(device))
    dt = (time.perf_counter() - t0) / n
    return dt


@torch.no_grad()
def measure_stability(model, loader, device, sigma: float = 0.05, max_batches: int = 10):
    """Prediction agreement when small Gaussian noise perturbs the input — a proxy
    for temporal/visual stability of the model's decision."""
    model.eval()
    agree = total = 0
    for i, (clips, _) in enumerate(loader):
        if i >= max_batches:
            break
        clips = clips.to(device)
        p0 = model(clips).argmax(1)
        p1 = model(clips + sigma * torch.randn_like(clips)).argmax(1)
        agree += (p0 == p1).sum().item()
        total += p0.numel()
    return agree / max(1, total)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=config.MODEL_NAMES)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    mcfg = config.get_config(args.model)
    frames = args.frames or mcfg["num_frames"]
    img_size = args.img_size or mcfg["img_size"]
    batch_size = args.batch_size or mcfg["batch_size"]
    device = torch.device(args.device) if args.device else get_device()

    _, _, test_ds = get_datasets(frames, img_size, synthetic=args.synthetic,
                                 per_class=args.per_class, limit=args.limit,
                                 seed=config.SEED)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    log.info(f"test clips: {len(test_ds)}")

    # build architecture (pretrained=False — we load our own weights) and load ckpt
    model = build_model(args.model, config.NUM_CLASSES, num_frames=frames,
                        img_size=img_size, pretrained=False, tiny=args.tiny).to(device)
    ckpt_path = Path(args.ckpt) if args.ckpt else (config.CHECKPOINT_DIR / f"{args.model}_best.pt")
    if not ckpt_path.exists():
        ckpt_path = config.CHECKPOINT_DIR / f"{args.model}.pt"
    ckpt = load_checkpoint(ckpt_path, model, map_location=str(device))
    log.info(f"loaded {ckpt_path} (epoch {ckpt.get('epoch')})")

    y_true, y_pred, conf = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, config.CLASSES)
    metrics["latency_s_per_clip"] = measure_latency(model, test_ds, device)
    metrics["fps"] = frames / metrics["latency_s_per_clip"]
    metrics["stability"] = measure_stability(model, test_loader, device)
    metrics["mean_confidence"] = float(np.mean(conf)) if conf else 0.0
    metrics["params"] = count_parameters(model)
    metrics["size_mb"] = model_size_mb(model)
    metrics["model"] = args.model
    metrics["config"] = {"frames": frames, "img_size": img_size}

    save_json(config.RESULTS_DIR / f"{args.model}_metrics.json", metrics)
    _plot_confusion(metrics["confusion_matrix"], config.CLASSES,
                    config.RESULTS_DIR / f"{args.model}_confusion.png", args.model)
    log.info(f"acc={metrics['accuracy']:.3f} f1={metrics['f1_macro']:.3f} "
             f"latency={metrics['latency_s_per_clip']*1000:.0f}ms "
             f"stability={metrics['stability']:.2f}")


def _plot_confusion(cm, classes, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes,
                yticklabels=classes, ax=ax, cbar=False)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix — {title}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
