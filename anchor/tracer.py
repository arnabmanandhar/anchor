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


class Tracer:
    def __init__(self, db_path: str = "anchor_traces.db"):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
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
                judge_json, latency_ms, cost_usd, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_result.run_id,
                run_result.pipeline_version,
                run_result.query,
                "",  # filled by runner via update_answer if needed
                json.dumps(run_result.retrieval.model_dump() if run_result.retrieval else None),
                json.dumps(run_result.judge.model_dump()),
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
