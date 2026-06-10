"""SQLite history of demo inference runs (demo/history.db)."""
from __future__ import annotations

import json
import sqlite3
import time
from typing import List

from src.config import DEMO_DB


def _conn():
    DEMO_DB.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(DEMO_DB))


def init_db() -> None:
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, filename TEXT, model TEXT,
                pred TEXT, confidence REAL, latency_s REAL, top3 TEXT)""")


def log_run(filename: str, model: str, pred: str, confidence: float,
            latency_s: float, top3: list) -> None:
    init_db()
    with _conn() as c:
        c.execute(
            "INSERT INTO runs (ts, filename, model, pred, confidence, latency_s, top3)"
            " VALUES (?,?,?,?,?,?,?)",
            (time.time(), filename, model, pred, float(confidence),
             float(latency_s), json.dumps(top3)))


def fetch_all() -> List[dict]:
    init_db()
    with _conn() as c:
        rows = c.execute(
            "SELECT ts, filename, model, pred, confidence, latency_s FROM runs"
            " ORDER BY ts DESC").fetchall()
    cols = ["ts", "filename", "model", "pred", "confidence", "latency_s"]
    return [dict(zip(cols, r)) for r in rows]


def summary() -> dict:
    rows = fetch_all()
    if not rows:
        return {"total": 0}
    per_class: dict = {}
    for r in rows:
        per_class[r["pred"]] = per_class.get(r["pred"], 0) + 1
    return {
        "total": len(rows),
        "avg_confidence": sum(r["confidence"] for r in rows) / len(rows),
        "avg_latency_s": sum(r["latency_s"] for r in rows) / len(rows),
        "per_class": per_class,
    }
