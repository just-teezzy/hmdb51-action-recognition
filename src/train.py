"""Unified training script for all 5 architectures.

Examples
--------
Smoke (synthetic, CPU, 1 epoch):
    python -m src.train --model tsn --synthetic --epochs 1 --per-class 6 \
        --frames 8 --img-size 64 --no-pretrained --batch-size 4

Real HMDB51 (Colab GPU):
    python -m src.train --model r2plus1d --epochs 15

A checkpoint is written to checkpoints/<model>.pt after every epoch (Colab can die
at any time) and the best-val checkpoint is kept as checkpoints/<model>_best.pt.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config
from src.data.build import get_datasets
from src.models.registry import build_model
from src.utils import (AverageMeter, count_parameters, get_device, get_logger,
                       model_size_mb, save_checkpoint, save_json, seed_everything)

log = get_logger()

HEAD_KEYWORDS = ("fc", "classifier", "proj", "head")


def freeze_backbone(model: nn.Module) -> None:
    """Freeze everything, then re-enable only classifier-head parameters."""
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if any(k in name.lower() for k in HEAD_KEYWORDS):
            p.requires_grad = True


@torch.no_grad()
def evaluate_loader(model, loader, device, criterion):
    model.eval()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        logits = model(clips)
        loss = criterion(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean().item()
        loss_m.update(loss.item(), clips.size(0))
        acc_m.update(acc, clips.size(0))
    return loss_m.avg, acc_m.avg


def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(1) == labels).float().mean().item()
        loss_m.update(loss.item(), clips.size(0))
        acc_m.update(acc, clips.size(0))
    return loss_m.avg, acc_m.avg


def parse_args():
    cfg = config.DEFAULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=config.MODEL_NAMES)
    ap.add_argument("--epochs", type=int, default=cfg["epochs"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=cfg["weight_decay"])
    ap.add_argument("--workers", type=int, default=cfg["num_workers"])
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-freeze", action="store_true", help="full fine-tune")
    ap.add_argument("--tiny", action="store_true", help="tiny transformer (smoke)")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic smoke data")
    ap.add_argument("--per-class", type=int, default=12, help="synthetic clips/class")
    ap.add_argument("--limit", type=int, default=None, help="cap samples per split")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(config.SEED)
    mcfg = config.get_config(args.model)
    frames = args.frames or mcfg["num_frames"]
    img_size = args.img_size or mcfg["img_size"]
    batch_size = args.batch_size or mcfg["batch_size"]
    lr = args.lr or mcfg["lr"]
    device = torch.device(args.device) if args.device else get_device()

    log.info(f"model={args.model} device={device} frames={frames} img={img_size} "
             f"bs={batch_size} lr={lr} epochs={args.epochs}")

    train_ds, val_ds, _ = get_datasets(frames, img_size, synthetic=args.synthetic,
                                       per_class=args.per_class, limit=args.limit,
                                       seed=config.SEED)
    log.info(f"datasets: train={len(train_ds)} val={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=args.workers)

    model = build_model(args.model, config.NUM_CLASSES, num_frames=frames,
                        img_size=img_size, pretrained=not args.no_pretrained,
                        tiny=args.tiny).to(device)
    if not args.no_freeze:
        freeze_backbone(model)
    n_train = sum(p.requires_grad for p in model.parameters())
    log.info(f"params total={count_parameters(model):,} size={model_size_mb(model):.1f}MB "
             f"trainable_tensors={n_train}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=args.weight_decay)

    history = {"model": args.model, "config": {
        "frames": frames, "img_size": img_size, "batch_size": batch_size,
        "lr": lr, "epochs": args.epochs, "pretrained": not args.no_pretrained,
        "frozen": not args.no_freeze}, "epochs": []}

    best_val = -1.0
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc = evaluate_loader(model, val_loader, device, criterion)
        dt = time.perf_counter() - t0
        log.info(f"epoch {epoch}/{args.epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f}"
                 f" | val loss {va_loss:.3f} acc {va_acc:.3f} | {dt:.1f}s")
        history["epochs"].append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                                      val_loss=va_loss, val_acc=va_acc, seconds=dt))
        # survivability: checkpoint every epoch
        save_checkpoint(config.CHECKPOINT_DIR / f"{args.model}.pt", model, optimizer,
                        epoch, extra={"val_acc": va_acc})
        if va_acc > best_val:
            best_val = va_acc
            save_checkpoint(config.CHECKPOINT_DIR / f"{args.model}_best.pt", model,
                            optimizer, epoch, extra={"val_acc": va_acc})

    history["best_val_acc"] = best_val
    save_json(config.RESULTS_DIR / f"{args.model}_history.json", history)
    log.info(f"done. best val acc={best_val:.3f} -> {config.RESULTS_DIR / (args.model + '_history.json')}")


if __name__ == "__main__":
    main()
