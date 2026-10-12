import json
import sqlite3

import pytest

from anchor import compare_versions, evaluate_pipeline, run_gate, summarize
from anchor.cli import main, run_demo
from anchor.judge import FaithfulnessJudge
from anchor.schema import Chunk
from anchor.tracer import Tracer

HEURISTIC = FaithfulnessJudge(backend="heuristic")


def make_pipeline(ranking):
    def pipeline(query):
        chunks = [Chunk(id=i, text=f"text about {query} number {i}") for i in ranking]
        return chunks, f"text about {query} number {ranking[0]}."

    return pipeline


DATASET = [{"query": "alpha", "reference_chunk_ids": ["a"]}, {"query": "beta", "reference_chunk_ids": ["a"]}]


def test_runs_store_answer_and_chunks(tmp_path):
    db = str(tmp_path / "t.db")
    results = evaluate_pipeline(make_pipeline(["a", "b"]), DATASET, pipeline_version="v1", judge=HEURISTIC, db_path=db)
    assert summarize(results)["mean_recall_at_k"] == 1.0
    tracer = Tracer(db)
    run = tracer.fetch_runs("v1")[0]
    assert run["generated_answer"].startswith("text about")
    assert [c["id"] for c in json.loads(run["retrieved_json"])] == ["a", "b"]
    assert json.loads(run["reference_json"]) == ["a"]
    assert {s["name"] for s in tracer.fetch_spans(run["run_id"])} == {
        "retrieval+generation",
        "retrieval_metrics",
        "faithfulness_judge",
    }


def test_pipeline_may_return_plain_dicts(tmp_path):
    def pipeline(q):
        return [{"id": "a", "text": "alpha"}], "alpha."

    results = evaluate_pipeline(pipeline, DATASET[:1], judge=HEURISTIC, db_path=str(tmp_path / "t.db"))
    assert results[0].retrieval.recall_at_k == 1.0


def test_old_v01_database_is_upgraded_in_place(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE runs (run_id TEXT PRIMARY KEY, pipeline_version TEXT, query TEXT, generated_answer TEXT,"
        " retrieval_json TEXT, judge_json TEXT, latency_ms REAL, cost_usd REAL, created_at TEXT)"
    )
    conn.commit()
    conn.close()
    evaluate_pipeline(make_pipeline(["a"]), DATASET, judge=HEURISTIC, db_path=str(db))
    assert len(Tracer(str(db)).fetch_runs()) == 2


def test_compare_flags_the_regressed_query(tmp_path):
    db = str(tmp_path / "t.db")
    evaluate_pipeline(make_pipeline(["a", "b"]), DATASET, pipeline_version="good", judge=HEURISTIC, db_path=db)
    evaluate_pipeline(
        lambda q: make_pipeline(["b", "a"] if q == "beta" else ["a", "b"])(q),
        DATASET,
        pipeline_version="worse",
        judge=HEURISTIC,
        db_path=db,
    )
    result = compare_versions(db, "good", "worse")
    assert result["diff (b - a)"]["mean_mrr"] == pytest.approx(-0.25)
    assert result["regressed_queries"] == ["beta"]


def test_gate_floors_and_baseline(tmp_path):
    db = str(tmp_path / "t.db")
    evaluate_pipeline(make_pipeline(["a", "b"]), DATASET, pipeline_version="base", judge=HEURISTIC, db_path=db)
    evaluate_pipeline(make_pipeline(["b", "c"]), DATASET, pipeline_version="broken", judge=HEURISTIC, db_path=db)

    assert run_gate(db, "base", floors={"recall": 1.0}).passed
    failing = run_gate(db, "broken", floors={"recall": 0.5}, baseline="base", max_drop={"mrr": 0.1})
    assert not failing.passed
    assert all(not c["passed"] for c in failing.checks)
    assert not run_gate(db, "missing", floors={"recall": 0.0}).passed
    with pytest.raises(ValueError):
        run_gate(db, "base", floors={"vibes": 1.0})


def test_cli_gate_exit_code(tmp_path):
    db = str(tmp_path / "t.db")
    evaluate_pipeline(make_pipeline(["b"]), DATASET, pipeline_version="v", judge=HEURISTIC, db_path=db)
    assert main(["gate", "v", "--db", db, "--min-recall", "0.5"]) == 1
    assert main(["gate", "v", "--db", db, "--min-faithfulness", "0.5"]) == 0


def test_demo_is_rerunnable_and_shows_a_regression(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db = str(tmp_path / "demo.db")
    run_demo(db, quiet=True)
    reports = run_demo(db, quiet=True)  # second run replaces, doesn't duplicate
    assert reports["demo-tfidf"]["n_runs"] == len(Tracer(db).fetch_runs("demo-tfidf"))
    assert reports["demo-tfidf"]["mean_recall_at_k"] == 1.0
    assert reports["demo-keyword"]["mean_recall_at_k"] < 1.0
    assert compare_versions(db, "demo-tfidf", "demo-keyword")["regressed_queries"]


def test_parallel_evaluation_keeps_order_and_writes_every_run(tmp_path):
    db = str(tmp_path / "t.db")
    dataset = [{"query": f"q{i}", "reference_chunk_ids": ["a"]} for i in range(20)]
    results = evaluate_pipeline(make_pipeline(["a", "b"]), dataset, pipeline_version="par", judge=HEURISTIC,
                                db_path=db, max_workers=6)
    assert [r.query for r in results] == [d["query"] for d in dataset]
    assert len(Tracer(db).fetch_runs("par")) == 20
    assert len({r.run_id for r in results}) == 20


def test_gate_supports_precision_and_hit(tmp_path):
    db = str(tmp_path / "t.db")
    evaluate_pipeline(make_pipeline(["a", "b"]), DATASET, pipeline_version="v", judge=HEURISTIC, db_path=db, k=2)
    assert run_gate(db, "v", floors={"hit": 1.0, "precision": 0.5}).passed
    assert not run_gate(db, "v", floors={"precision": 0.6}).passed
