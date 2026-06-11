"""Collect successful and failed predictions of the best model for the report.

Runs the chosen model over the test split, then saves >=3 correct (high
confidence) and >=3 incorrect examples as frame montages + a JSON description.
Incorrect examples from the deliberately confusable pairs (run/walk,
fencing/draw_sword) are preferred, since those are the interesting failures.

Usage:
    python -m src.find_examples --model videomae
    python -m src.find_examples --model r2plus1d --n 3
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402
from src.data.splits import build_splits  # noqa: E402
from src.inference import predict_video  # noqa: E402
from src.utils import get_logger, save_json  # noqa: E402

log = get_logger()

CONFUSABLE = {("run", "walk"), ("walk", "run"),
              ("fencing", "draw_sword"), ("draw_sword", "fencing")}


def _montage(frames, title, out_path):
    n = min(8, len(frames))
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.6))
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].imshow(frames[i])
        axes[i].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="videomae", choices=config.MODEL_NAMES)
    ap.add_argument("--n", type=int, default=3, help="examples per category")
    args = ap.parse_args()

    test = build_splits()["test"]
    log.info(f"scanning {len(test)} test clips with '{args.model}'…")
    records = []
    for path, label in test:
        res = predict_video(path, model_name=args.model)
        true = config.IDX_TO_CLASS[label]
        records.append({"path": path, "true": true, "pred": res["pred"],
                        "conf": res["confidence"], "correct": res["pred"] == true,
                        "top": res["top"]})

    correct = sorted([r for r in records if r["correct"]], key=lambda r: -r["conf"])
    wrong = sorted([r for r in records if not r["correct"]],
                   key=lambda r: (0 if (r["true"], r["pred"]) in CONFUSABLE else 1,
                                  -r["conf"]))
    acc = len(correct) / max(1, len(records))
    log.info(f"accuracy={acc:.3f} | correct={len(correct)} wrong={len(wrong)}")

    ex_dir = config.ROOT / "reports" / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for tag, group in (("good", correct[:args.n]), ("bad", wrong[:args.n])):
        for i, r in enumerate(group, 1):
            res = predict_video(r["path"], model_name=args.model)  # frames for montage
            title = f"{tag.upper()} | true={r['true']}  pred={r['pred']}  ({r['conf']*100:.0f}%)"
            out = ex_dir / f"{tag}_{i}_{r['true']}_as_{r['pred']}.png"
            _montage(res["preview_frames"], title, out)
            saved.append({"category": tag, "true": r["true"], "pred": r["pred"],
                          "confidence": round(r["conf"], 4),
                          "top3": [[c, round(p, 4)] for c, p in r["top"]],
                          "video": r["path"].split("/")[-1],
                          "image": str(out.relative_to(config.ROOT))})
            log.info(f"  {tag}: true={r['true']:11s} pred={r['pred']:11s} "
                     f"conf={r['conf']:.2f} -> {out.name}")

    save_json(ex_dir / "examples.json",
              {"model": args.model, "test_accuracy": acc,
               "n_correct": len(correct), "n_wrong": len(wrong),
               "n_test": len(records), "examples": saved})
    log.info(f"wrote {ex_dir/'examples.json'} and {2*args.n} montages")


if __name__ == "__main__":
    main()
