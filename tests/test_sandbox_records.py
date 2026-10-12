import json
import math
from pathlib import Path

import pytest

from anchor import evaluate_records, load_records, summarize
from anchor.judge import FaithfulnessJudge
from anchor.sandbox import extractive_answer, make_pipeline, make_retriever, parse_corpus, parse_dataset
from anchor.schema import Chunk

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
HEURISTIC = FaithfulnessJudge(backend="heuristic")
CHUNKS = [
    Chunk(id="a", text="cats purr when they are happy"),
    Chunk(id="b", text="dogs bark at strangers and cats"),
    Chunk(id="c", text="birds sing in the morning"),
]


def test_bm25_matches_hand_computation():
    # Content tokens: a = cats purr when they happy (5), b = dogs bark strangers cats (4),
    # c = birds sing morning (3) -> avgdl = 4. Query "purr": only a, tf = 1, df = 1, N = 3.
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 1 * (1.5 + 1) / (1 + 1.5 * (1 - 0.75 + 0.75 * 5 / 4))
    top = make_retriever(CHUNKS, "bm25", top_k=3)("purr")
    assert [c.id for c in top] == ["a", "b", "c"]
    assert top[0].score == pytest.approx(expected, abs=1e-4)
    assert top[1].score == top[2].score == 0


@pytest.mark.parametrize("method", ["bm25", "tfidf", "keyword", "hybrid"])
def test_every_retriever_ranks_the_obvious_chunk_first(method):
    assert make_retriever(CHUNKS, method, top_k=2)("why do birds sing")[0].id == "c"


def test_ties_keep_corpus_order():
    ids = [c.id for c in make_retriever(CHUNKS, "keyword", top_k=3)("zebra")]
    assert ids == ["a", "b", "c"]


def test_extractive_answer_only_uses_retrieved_text():
    answer = extractive_answer("do cats purr", CHUNKS[:2])
    assert answer.startswith("cats purr")
    verdict = HEURISTIC.score("q", CHUNKS[:2], answer)
    assert verdict.faithfulness_score == 1.0


def test_llm_generator_requires_key():
    with pytest.raises(ValueError):
        make_pipeline(CHUNKS, generator="llm")


@pytest.mark.parametrize(
    "name,data",
    [
        ("c.csv", "id,text\nx1,hello world\nx2,second chunk\n"),
        ("c.json", json.dumps([{"id": "x1", "text": "hello world"}, "second chunk"])),
        ("c.jsonl", '{"id": "x1", "text": "hello world"}\n{"id": "x2", "text": "second chunk"}\n'),
        ("c.txt", "hello world\n\nsecond chunk\n"),
    ],
)
def test_parse_corpus_formats(name, data):
    chunks = parse_corpus(data, name)
    assert len(chunks) == 2 and chunks[0].text == "hello world"


def test_parse_corpus_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        parse_corpus("id,text\na,x\na,y\n", "c.csv")


def test_parse_dataset_csv_reference_ids():
    rows = parse_dataset("query,reference_chunk_ids\nq1,\"a, b\"\nq2,\n", "d.csv")
    assert rows == [{"query": "q1", "reference_chunk_ids": ["a", "b"]}, {"query": "q2", "reference_chunk_ids": None}]


def test_sample_files_load():
    assert len(parse_corpus((EXAMPLES / "sample_corpus.json").read_text(), "sample_corpus.json")) == 20
    questions = json.loads((EXAMPLES / "sample_questions.json").read_text())
    corpus_ids = {c.id for c in parse_corpus((EXAMPLES / "sample_corpus.json").read_text(), "x.json")}
    assert all(set(q["reference_chunk_ids"]) <= corpus_ids for q in questions)


def test_records_aliases_and_bare_string_chunks():
    recs = load_records(
        json.dumps([{"query": "q", "contexts": ["alpha text", "beta text"], "response": "alpha text.", "relevant_ids": "r1"}]),
        "r.json",
    )
    assert recs[0]["retrieved_chunks"][1]["id"] == "r2"
    assert recs[0]["reference_chunk_ids"] == ["r1"]


def test_records_errors_name_the_row():
    with pytest.raises(ValueError, match="record 2: missing 'answer'"):
        load_records('{"query": "a", "answer": "x"}\n{"query": "b"}\n', "r.jsonl")


def test_sample_records_score_as_documented(tmp_path):
    records = load_records((EXAMPLES / "sample_records.jsonl").read_bytes(), "sample_records.jsonl")
    s = summarize(evaluate_records(records, pipeline_version="logs", judge=HEURISTIC, db_path=str(tmp_path / "r.db")))
    assert s["n_runs"] == 4
    assert s["mean_recall_at_k"] == 0.75  # one miss out of four
    assert s["mean_faithfulness"] < 1.0  # hallucinated answers are caught
