"""
Minimal trace storage. Every evaluation run gets a row in `runs`; every
timed step inside it (retrieval, generation, judging) gets a row in
`spans`. SQLite because this needs to work with zero setup — no Docker,
no external service, just a file on disk.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    pipeline_version TEXT,
    query TEXT,
    generated_answer TEXT,
    retrieval_json TEXT,
    judge_json TEXT,
    retrieved_json TEXT,
    reference_json TEXT,
    latency_ms REAL,
    cost_usd REAL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    name TEXT,
    duration_ms REAL,
    metadata_json TEXT,
    created_at TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
"""

# Columns added after v0.1. Older trace databases are upgraded in place on open.
_ADDED_RUN_COLUMNS = {"retrieved_json": "TEXT", "reference_json": "TEXT"}


class Tracer:
    def __init__(self, db_path: str = "anchor_traces.db"):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(runs)")}
        for col, col_type in _ADDED_RUN_COLUMNS.items():
            if col not in existing:
                self._conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {col_type}")
        self._conn.commit()

    def span(self, run_id: str, name: str):
        return Span(self, run_id, name)

    def _record_span(self, run_id: str, name: str, duration_ms: float, metadata: Optional[Dict] = None):
        self._conn.execute(
            "INSERT INTO spans (run_id, name, duration_ms, metadata_json, created_at) VALUES (?, ?, ?, ?, datetime('now'))",
            (run_id, name, duration_ms, json.dumps(metadata or {})),
        )
        self._conn.commit()

    def record_run(self, run_result) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, pipeline_version, query, generated_answer, retrieval_json,
                judge_json, retrieved_json, reference_json, latency_ms, cost_usd, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_result.run_id,
                run_result.pipeline_version,
                run_result.query,
                run_result.generated_answer,
                json.dumps(run_result.retrieval.model_dump() if run_result.retrieval else None),
                json.dumps(run_result.judge.model_dump()),
                json.dumps([c.model_dump() for c in run_result.retrieved_chunks]),
                json.dumps(run_result.reference_chunk_ids),
                run_result.latency_ms,
                run_result.cost_usd,
                run_result.created_at,
            ),
        )
        self._conn.commit()

    def fetch_runs(self, pipeline_version: Optional[str] = None):
        cur = self._conn.cursor()
        if pipeline_version:
            cur.execute("SELECT * FROM runs WHERE pipeline_version = ? ORDER BY created_at", (pipeline_version,))
        else:
            cur.execute("SELECT * FROM runs ORDER BY created_at")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def delete_version(self, pipeline_version: str) -> int:
        """Remove every run (and its spans) for one pipeline_version. Returns runs deleted."""
        self._conn.execute(
            "DELETE FROM spans WHERE run_id IN (SELECT run_id FROM runs WHERE pipeline_version = ?)",
            (pipeline_version,),
        )
        cur = self._conn.execute("DELETE FROM runs WHERE pipeline_version = ?", (pipeline_version,))
        self._conn.commit()
        return cur.rowcount

    def list_versions(self):
        cur = self._conn.execute("SELECT DISTINCT pipeline_version FROM runs ORDER BY pipeline_version")
        return [row[0] for row in cur.fetchall()]

    def fetch_spans(self, run_id: str):
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM spans WHERE run_id = ? ORDER BY id", (run_id,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self):
        self._conn.close()


class Span:
    """Context manager: `with tracer.span(run_id, "retrieval") as s: ...`
    Records wall-clock duration automatically on exit."""

    def __init__(self, tracer: Tracer, run_id: str, name: str):
        self.tracer = tracer
        self.run_id = run_id
        self.name = name
        self.metadata: Dict = {}
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        duration_ms = (time.perf_counter() - self._start) * 1000
        if exc_type is not None:
            self.metadata["error"] = str(exc)
        self.tracer._record_span(self.run_id, self.name, duration_ms, self.metadata)
        return False  # don't swallow exceptions
