import json
from pathlib import Path

import pytest

import anchor.synth as synth
from anchor.cli import main
from anchor.sandbox import parse_corpus, parse_dataset, tokenize
from anchor.synth import cloze_questions, difficulty, generate_questions

CORPUS = Path(__file__).resolve().parent.parent / "examples" / "sample_corpus.json"


@pytest.fixture(scope="module")
def chunks():
    return parse_corpus(CORPUS.read_text(), CORPUS.name)


def test_cloze_rules():
    by_rule = {q["rule"]: q for q in cloze_questions("Nepal is divided into seven provinces, each with its own assembly.")}
    assert by_rule["count"]["question"] == "Nepal is divided into how many provinces?"
    assert by_rule["count"]["answer"] == "seven"
    date = cloze_questions("The Constitution was promulgated on 20 September 2015 by the Assembly.")[0]
    assert date["rule"] == "date" and date["answer"] == "20 September 2015" and "what date" in date["question"]
    numbered = cloze_questions("Article 7 makes Nepali the official language of Nepal.")[0]
    assert numbered["question"].startswith("Which article makes") and numbered["answer"] == "Article 7"
    # "84 sets" is an article number followed by a verb, not a count.
    count = [q for q in cloze_questions("Article 84 sets the House at 275 members.") if q["rule"] == "count"][0]
    assert count["answer"] == "275"
    assert cloze_questions("Hello there") == []


def test_answers_never_leak_into_questions(chunks):
    for c in chunks:
        for q in cloze_questions(c.text):
            assert q["answer"].lower() not in q["question"].lower()


def test_generated_set_shape_and_ground_truth(chunks):
    qs = generate_questions(chunks, n=10)
    assert len(qs) == 10
    assert [q["type"] for q in qs].count("single_hop") == 6
    by_id = {c.id: c.text for c in chunks}
    for q in qs:
        if q["type"] == "single_hop":
            (ref,) = q["reference_chunk_ids"]
            assert q["reference_answer"] in by_id[ref]
        elif q["type"] == "multi_hop":
            assert len(set(q["reference_chunk_ids"])) == 2
        else:
            assert q["reference_chunk_ids"] == [] and q["reference_answer"] is None
            assert q["difficulty"] == "n/a"


def test_unanswerable_topics_are_absent_from_corpus(chunks):
    vocab = {t for c in chunks for t in tokenize(c.text)}
    qs = generate_questions(chunks, n=8, mix={"unanswerable": 1})
    assert len(qs) == 8 and all("Nepal" in q["query"] for q in qs)
    for q in qs:
        keys = next(k for t, k in synth._UNANSWERABLE_TEMPLATES if t.format(e="Nepal") == q["query"])
        assert not vocab & set(keys)


def test_deterministic_and_seeded(chunks):
    assert generate_questions(chunks, n=10, seed=1) == generate_questions(chunks, n=10, seed=1)
    assert generate_questions(chunks, n=10, seed=1) != generate_questions(chunks, n=10, seed=2)


def test_no_near_duplicates(chunks):
    qs = generate_questions(chunks, n=60)
    for i, a in enumerate(qs):
        for b in qs[:i]:
            assert synth._jaccard(a["query"], b["query"]) < synth.DUPLICATE_JACCARD


def test_difficulty_labels_and_drop_easy(chunks):
    index = synth._Index(chunks)
    assert difficulty("Nepal is divided into how many provinces?", ["prov-007"], index) == "easy"
    assert difficulty("unrelated words entirely", ["prov-007"], index) == "hard"
    assert difficulty("anything", [], index) == "n/a"
    assert all(q["difficulty"] != "easy" for q in generate_questions(chunks, n=30, drop_easy=True))


def test_llm_generator_flow_without_network(chunks, monkeypatch):
    calls = []

    def fake(kind, context, connection):
        calls.append(kind)
        return {"query": f"natural {kind} question {len(calls)} about {context[0].id}",
                "reference_answer": None if kind == "unanswerable" else "x", "rule": "llm"}

    monkeypatch.setattr(synth, "_llm_question", fake)
    from anchor.llm import Connection, Secret

    qs = generate_questions(chunks, n=5, generator="llm", connection=Connection("groq", "m", Secret("gsk_test-key-0000")))
    assert len(qs) == 5 and all(q["generator"] == "llm" for q in qs)
    assert len(calls) <= 10  # attempts are capped at twice the quota
    with pytest.raises(ValueError):
        generate_questions(chunks, generator="llm")
    with pytest.raises(ValueError):
        generate_questions(chunks, mix={"trick": 1})


def test_generated_file_round_trips_into_a_dataset(chunks, tmp_path):
    out = tmp_path / "q.json"
    assert main(["synth", str(CORPUS), "-n", "6", "-o", str(out), "--generator", "template"]) == 0
    rows = parse_dataset(out.read_text(), "q.json")
    assert len(rows) == 6 and {"type", "reference_chunk_ids"} <= set(rows[0])
    assert json.loads(out.read_text())[0]["generator"] == "template"
