"""Study the effect of clip length (number of frames) on accuracy.

Re-evaluates a trained checkpoint at several ``num_frames`` values on the test
split. Models with a fixed temporal input are skipped:
  * TSM           — TemporalShift has a fixed n_segment
  * VideoMAE/TimeSformer — fixed positional embeddings
Sweepable (variable-T): TSN (temporal pooling), I3D, R(2+1)D (3D convs + adaptive
pooling).

Usage:
    python -m src.frame_sweep                      # default models & frame counts
    python -m src.frame_sweep --models r2plus1d i3d --frames 4 8 16 32
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src import config  # noqa: E402
from src.data.build import get_datasets  # noqa: E402
from src.evaluate import collect_predictions  # noqa: E402
from src.models.registry import build_model  # noqa: E402
from src.utils import (compute_metrics, get_device, get_logger, load_checkpoint,  # noqa: E402
                       save_json)

log = get_logger()
SWEEPABLE = ["tsn", "i3d", "r2plus1d"]
DEFAULT_FRAMES = [4, 8, 16, 32]


def sweep_model(name: str, frames_list, device) -> dict:
    mcfg = config.get_config(name)
    img = mcfg["img_size"]
    model = build_model(name, config.NUM_CLASSES, num_frames=mcfg["num_frames"],
                        img_size=img, pretrained=False).to(device)
    ckpt = config.CHECKPOINT_DIR / f"{name}_best.pt"
    if not ckpt.exists():
        ckpt = config.CHECKPOINT_DIR / f"{name}.pt"
    if not ckpt.exists():
        log.info(f"{name}: no checkpoint, skip")
        return {}
    load_checkpoint(ckpt, model, map_location=str(device))
    model.eval()

    out = {}
    for F in frames_list:
        try:
            _, _, test = get_datasets(F, img, synthetic=False)
            loader = DataLoader(test, batch_size=mcfg["batch_size"])
            yt, yp, _ = collect_predictions(model, loader, device)
            m = compute_metrics(yt, yp, config.CLASSES)
            out[F] = {"accuracy": m["accuracy"], "f1_macro": m["f1_macro"]}
            log.info(f"{name} frames={F:2d}: acc={m['accuracy']:.3f} f1={m['f1_macro']:.3f}")
        except Exception as e:  # noqa: BLE001 — some frame counts may be invalid
            log.info(f"{name} frames={F}: skipped ({e})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=SWEEPABLE)
    ap.add_argument("--frames", nargs="+", type=int, default=DEFAULT_FRAMES)
    args = ap.parse_args()
    device = get_device()

    results = {}
    for name in args.models:
        r = sweep_model(name, args.frames, device)
        if r:
            results[name] = r
    save_json(config.RESULTS_DIR / "frame_sweep.json", results)

    # plot accuracy vs num_frames
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, r in results.items():
        xs = sorted(r.keys())
        ax.plot(xs, [r[x]["accuracy"] for x in xs], marker="o", label=name)
    ax.set_xlabel("число кадров (clip length)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Влияние длины фрагмента на качество")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(config.RESULTS_DIR / "frame_sweep.png", dpi=120, bbox_inches="tight")
    log.info(f"wrote {config.RESULTS_DIR/'frame_sweep.json'} and frame_sweep.png")


if __name__ == "__main__":
    main()
