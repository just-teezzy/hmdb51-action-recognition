"""Aggregate per-model metrics into a comparison table + figures.

Reads every results/<model>_metrics.json produced by evaluate.py and writes:
  results/comparison.csv      one row per architecture
  results/comparison.md       markdown table (for the README / report)
  results/comparison.png      accuracy / F1 / latency / size bar charts
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.utils import get_logger, load_json  # noqa: E402

log = get_logger()

COLUMNS = ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro",
           "latency_s_per_clip", "fps", "stability", "mean_confidence",
           "params", "size_mb"]


def build_table() -> pd.DataFrame:
    rows = []
    for name in config.MODEL_NAMES:
        p = config.RESULTS_DIR / f"{name}_metrics.json"
        if not p.exists():
            log.info(f"skip {name}: no metrics yet")
            continue
        m = load_json(p)
        rows.append({k: m.get(k) for k in COLUMNS})
    if not rows:
        raise SystemExit("No *_metrics.json found in results/. Run evaluate first.")
    df = pd.DataFrame(rows)[COLUMNS].sort_values("accuracy", ascending=False)
    return df.reset_index(drop=True)


def save_outputs(df: pd.DataFrame) -> None:
    csv = config.RESULTS_DIR / "comparison.csv"
    df.to_csv(csv, index=False)

    show = df.copy()
    show["params"] = (show["params"] / 1e6).round(2).astype(str) + "M"
    show["latency_ms"] = (show["latency_s_per_clip"] * 1000).round(1)
    show = show.drop(columns=["latency_s_per_clip"])
    for c in ["accuracy", "precision_macro", "recall_macro", "f1_macro",
              "stability", "mean_confidence", "fps", "size_mb"]:
        show[c] = show[c].round(3)
    md = config.RESULTS_DIR / "comparison.md"
    md.write_text(show.to_markdown(index=False))
    log.info(f"wrote {csv} and {md}")
    _plot(df)


def _plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    m = df["model"]
    axes[0, 0].bar(m, df["accuracy"], color="steelblue")
    axes[0, 0].set_title("Accuracy"); axes[0, 0].set_ylim(0, 1)
    axes[0, 1].bar(m, df["f1_macro"], color="seagreen")
    axes[0, 1].set_title("F1 (macro)"); axes[0, 1].set_ylim(0, 1)
    axes[1, 0].bar(m, df["latency_s_per_clip"] * 1000, color="indianred")
    axes[1, 0].set_title("Latency per clip (ms, CPU)")
    axes[1, 1].scatter(df["latency_s_per_clip"] * 1000, df["accuracy"], s=80)
    for _, r in df.iterrows():
        axes[1, 1].annotate(r["model"], (r["latency_s_per_clip"] * 1000, r["accuracy"]))
    axes[1, 1].set_xlabel("latency ms"); axes[1, 1].set_ylabel("accuracy")
    axes[1, 1].set_title("Accuracy vs latency")
    for ax in axes.flat:
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)
    plt.tight_layout()
    out = config.RESULTS_DIR / "comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info(f"wrote {out}")


def main():
    df = build_table()
    print(df.to_string(index=False))
    save_outputs(df)


if __name__ == "__main__":
    main()
