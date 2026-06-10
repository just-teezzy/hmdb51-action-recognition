"""Generate an auto-report (Excel + PDF) from the comparison results.

Reads results/comparison.csv and per-model metrics, then writes:
    reports/out/report.xlsx
    reports/out/report.pdf      (comparison table + charts + confusion matrices)

Usage:
    python -m reports.generate_report
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.utils import get_logger, load_json

log = get_logger()


def _load_df() -> pd.DataFrame:
    csv = config.RESULTS_DIR / "comparison.csv"
    if csv.exists():
        return pd.read_csv(csv)
    # fall back to building from per-model metrics
    rows = []
    for name in config.MODEL_NAMES:
        p = config.RESULTS_DIR / f"{name}_metrics.json"
        if p.exists():
            m = load_json(p)
            rows.append({k: m.get(k) for k in
                         ["model", "accuracy", "f1_macro", "latency_s_per_clip",
                          "params", "size_mb"]})
    return pd.DataFrame(rows)


def generate_excel(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="comparison", index=False)
        # per-class F1 for each model, if metrics exist
        for name in config.MODEL_NAMES:
            p = config.RESULTS_DIR / f"{name}_metrics.json"
            if not p.exists():
                continue
            m = load_json(p)
            pc = m.get("per_class", {})
            if pc:
                pcd = pd.DataFrame(pc).T.reset_index().rename(columns={"index": "class"})
                pcd.to_excel(xl, sheet_name=f"{name}_perclass"[:31], index=False)
    log.info(f"wrote {out}")


def generate_pdf(df: pd.DataFrame, out: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    out.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = [Paragraph("Action Recognition — Model Comparison Report", styles["Title"]),
             Spacer(1, 0.4 * cm),
             Paragraph(f"Dataset: HMDB51 subset ({config.NUM_CLASSES} classes). "
                       f"Classes: {', '.join(config.CLASSES)}.", styles["Normal"]),
             Spacer(1, 0.4 * cm)]

    # comparison table
    show = df.copy()
    for c in show.columns:
        if show[c].dtype.kind in "fc":
            show[c] = show[c].round(3)
    data = [list(show.columns)] + show.values.tolist()
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#E9EEF7")]),
    ]))
    story += [tbl, Spacer(1, 0.6 * cm)]

    # comparison chart
    chart = config.RESULTS_DIR / "comparison.png"
    if chart.exists():
        story += [Paragraph("Comparison charts", styles["Heading2"]),
                  Image(str(chart), width=16 * cm, height=12 * cm), Spacer(1, 0.4 * cm)]

    # confusion matrices
    for name in config.MODEL_NAMES:
        cm_png = config.RESULTS_DIR / f"{name}_confusion.png"
        if cm_png.exists():
            story += [Paragraph(f"Confusion matrix — {name}", styles["Heading3"]),
                      Image(str(cm_png), width=11 * cm, height=9 * cm),
                      Spacer(1, 0.3 * cm)]

    SimpleDocTemplate(str(out), pagesize=A4).build(story)
    log.info(f"wrote {out}")


def main() -> None:
    df = _load_df()
    if df.empty:
        raise SystemExit("No results to report. Run evaluate + compare first.")
    generate_excel(df, config.REPORT_DIR / "report.xlsx")
    generate_pdf(df, config.REPORT_DIR / "report.pdf")


if __name__ == "__main__":
    main()
