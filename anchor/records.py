"""
Load and validate pre-computed pipeline outputs ("records"), so a pipeline
can be evaluated from its logs without Anchor ever calling it — from the
CLI (`anchor score`) or by uploading a file in the dashboard.

One record per query:

    {
      "query": "How many provinces does Nepal have?",
      "retrieved_chunks": [                       # ranked, best first
        {"id": "c2", "text": "Nepal ... seven provinces.", "score": 0.91},
        {"id": "c1", "text": "The Constitution ..."}
      ],
      "answer": "Nepal has seven provinces.",
      "reference_chunk_ids": ["c2"]               # optional ground truth
    }

Accepted aliases (common names in other tools' logs):
    retrieved_chunks: chunks, contexts, retrieved, retrieved_contexts
    answer:           generated_answer, response, output
    reference_chunk_ids: relevant_ids, ground_truth_ids, reference_ids
A chunk may also be a bare string; it then gets the id "r1", "r2", ... by rank.
"""

from __future__ import annotations

import csv
import io
import json

from .sandbox import parse_reference_ids

_CHUNK_KEYS = ("retrieved_chunks", "chunks", "contexts", "retrieved", "retrieved_contexts")
_ANSWER_KEYS = ("answer", "generated_answer", "response", "output")
_REF_KEYS = ("reference_chunk_ids", "relevant_ids", "ground_truth_ids", "reference_ids")


def _first(record: dict, keys, default=None):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def normalize_record(record: dict, index: int = 1) -> dict:
    """Return {query, retrieved_chunks, answer, reference_chunk_ids} or raise
    ValueError naming the record number and what is wrong with it."""
    where = f"record {index}"
    query = str(record.get("query", "")).strip()
    if not query:
        raise ValueError(f"{where}: missing 'query'")

    raw_chunks = _first(record, _CHUNK_KEYS, [])
    if isinstance(raw_chunks, str):
        raw_chunks = json.loads(raw_chunks)
    chunks = []
    for rank, c in enumerate(raw_chunks, start=1):
        if isinstance(c, str):
            chunks.append({"id": f"r{rank}", "text": c})
        elif isinstance(c, dict) and "text" in c:
            chunks.append({"id": str(c.get("id") or f"r{rank}"), "text": str(c["text"]), "score": c.get("score")})
        else:
            raise ValueError(f"{where}: chunk {rank} needs a 'text' field")

    answer = _first(record, _ANSWER_KEYS)
    if answer is None:
        raise ValueError(f"{where}: missing 'answer'")

    return {
        "query": query,
        "retrieved_chunks": chunks,
        "answer": str(answer),
        "reference_chunk_ids": parse_reference_ids(_first(record, _REF_KEYS)),
    }


def load_records(data: bytes | str, filename: str = "records.jsonl") -> list[dict]:
    """Parse a JSON list, JSONL, or CSV file of records and validate every row.
    In CSV, retrieved_chunks must be a JSON array string in its column."""
    text = data.decode("utf-8-sig") if isinstance(data, bytes) else data
    name = filename.lower()
    if name.endswith(".csv"):
        rows = list(csv.DictReader(io.StringIO(text)))
    elif name.endswith(".jsonl"):
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = rows.get("records") or []
    if not rows:
        raise ValueError("No records found.")
    return [normalize_record(r, i) for i, r in enumerate(rows, start=1)]
