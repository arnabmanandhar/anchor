import json

import pytest

import anchor.llm as llm
from anchor.judge import FaithfulnessJudge, claim_overlap, is_abstention, numbers, stem
from anchor.llm import Connection, LLMError, Secret
from anchor.schema import Chunk
from anchor.tracer import Tracer

CHUNKS = [
    Chunk(id="c1", text="Nepal's federal structure divides the country into seven provinces."),
    Chunk(id="c2", text="The Supreme Court is the highest judicial authority."),
]
ANSWER = "Nepal has seven provinces. Each province elects a governor."
FAKE_KEY = "gsk_test-0000-do-not-leak-0000"
KEY_VARS = [v for p in llm.PROVIDERS.values() for v in p.env_keys] + ["ANCHOR_LLM_PROVIDER", "ANCHOR_LLM_MODEL"]


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("this path must not make a network call")

    monkeypatch.setattr(llm, "_post", boom)


def fake_provider(monkeypatch, replies):
    """Replace the HTTP layer. Each reply is a dict (the provider's JSON
    response) or an exception to raise. Returns the list of calls made."""
    calls, queue = [], list(replies)

    def _post(url, headers, payload, secret):
        calls.append({"url": url, "headers": headers, "payload": payload})
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm, "_post", _post)
    return calls


def openai_reply(content, prompt_tokens=1000, completion_tokens=200, finish="stop"):
    return {"choices": [{"message": {"content": content if isinstance(content, str) else json.dumps(content)},
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}


AUDIT = {
    "claims": [
        {"claim": "Nepal has seven provinces", "verdict": "supported", "chunk_id": "c1", "note": None},
        {"claim": "Each province elects a governor", "verdict": "unsupported", "chunk_id": None, "note": "not stated"},
    ],
    "reasoning": "One of two claims grounded.",
}


# ---------------------------------------------------------------- heuristic


def test_no_key_uses_heuristic_without_network(no_network):
    judge = FaithfulnessJudge()
    assert judge.active_backend == "heuristic"
    verdict = judge.score("How many provinces?", CHUNKS, ANSWER)
    assert verdict.judge_backend == "heuristic"
    assert [c.verdict for c in verdict.claims] == ["supported", "unsupported"]
    assert verdict.claims[0].chunk_id == "c1"
    assert verdict.faithfulness_score == 0.5


def test_empty_key_counts_as_no_key(monkeypatch, no_network):
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert FaithfulnessJudge().active_backend == "heuristic"


def test_forced_heuristic_ignores_key(no_network):
    judge = FaithfulnessJudge(provider="groq", api_key=FAKE_KEY, backend="heuristic")
    assert judge.score("q", CHUNKS, ANSWER).judge_backend == "heuristic"


def test_overlap_uses_light_stemming():
    assert claim_overlap("Nepal has seven provinces.", CHUNKS[0].text) == 1.0
    assert claim_overlap("Nepal has a province.", CHUNKS[0].text) == 1.0  # province ~ provinces
    assert claim_overlap("Nepal has seven governors.", CHUNKS[0].text) == pytest.approx(2 / 3)
    assert claim_overlap("", CHUNKS[0].text) == 0.0
    assert stem("cities") == "city" and stem("class") == "class"


def test_wrong_number_is_contradicted():
    verdict = FaithfulnessJudge(backend="heuristic").score("q", CHUNKS, "Nepal has nine provinces.")
    assert verdict.claims[0].verdict == "contradicted"
    assert "9" in verdict.claims[0].note and "7" in verdict.claims[0].note
    assert verdict.contradicted == 1 and verdict.faithfulness_score == 0.0
    assert numbers("seven provinces, 1,000 units") == {"7", "1000"}


def test_abstention_is_faithful_but_empty_answer_is_not():
    judge = FaithfulnessJudge(backend="heuristic")
    assert is_abstention("I don't know based on the context.")
    assert judge.score("q", CHUNKS, "").faithfulness_score == 0.0


# --------------------------------------------------------------------- LLM


def test_llm_verdict_is_derived_from_claims(monkeypatch):
    calls = fake_provider(monkeypatch, [openai_reply(AUDIT)])
    judge = FaithfulnessJudge(provider="groq", api_key=FAKE_KEY)
    verdict = judge.score("How many provinces?", CHUNKS, ANSWER)
    assert verdict.judge_backend == "llm" and verdict.judge_model == "groq/openai/gpt-oss-120b"
    assert verdict.faithfulness_score == 0.5
    assert verdict.input_tokens == 1000 and verdict.output_tokens == 200
    assert calls[0]["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert calls[0]["payload"]["response_format"]["type"] == "json_schema"


def test_every_provider_can_back_the_judge(monkeypatch):
    for pid, spec in llm.PROVIDERS.items():
        if spec.protocol == "anthropic":
            reply = {"content": [{"type": "text", "text": json.dumps(AUDIT)}], "stop_reason": "end_turn",
                     "usage": {"input_tokens": 10, "output_tokens": 5}}
        else:
            reply = openai_reply(AUDIT)
        calls = fake_provider(monkeypatch, [reply])
        verdict = FaithfulnessJudge(provider=pid, api_key=FAKE_KEY).score("q", CHUNKS, ANSWER)
        assert verdict.judge_backend == "llm" and verdict.faithfulness_score == 0.5, pid
        assert calls[0]["url"].startswith(spec.base_url), pid


def test_schema_mode_unsupported_falls_back_to_json_mode(monkeypatch):
    calls = fake_provider(monkeypatch, [LLMError("bad_request", "response_format not supported", 400),
                                        openai_reply("```json\n" + json.dumps(AUDIT) + "\n```")])
    verdict = FaithfulnessJudge(provider="huggingface", api_key=FAKE_KEY).score("q", CHUNKS, ANSWER)
    assert verdict.judge_backend == "llm"
    assert calls[1]["payload"]["response_format"] == {"type": "json_object"}


def test_invalid_reply_gets_one_repair_attempt(monkeypatch):
    calls = fake_provider(monkeypatch, [openai_reply({"claims": "nope"}), openai_reply(AUDIT)])
    verdict = FaithfulnessJudge(provider="groq", api_key=FAKE_KEY).score("q", CHUNKS, ANSWER)
    assert verdict.judge_backend == "llm" and len(calls) == 2
    assert verdict.input_tokens == 2000  # both calls are counted


def test_citation_of_unretrieved_chunk_is_downgraded(monkeypatch):
    fake_provider(monkeypatch, [openai_reply(
        {"claims": [{"claim": "x", "verdict": "supported", "chunk_id": "c99", "note": None}], "reasoning": "r"})])
    verdict = FaithfulnessJudge(provider="groq", api_key=FAKE_KEY).score("q", CHUNKS, "x")
    assert verdict.claims[0].verdict == "unsupported" and verdict.claims[0].supported is False
    assert verdict.faithfulness_score == 0.0


def test_failure_falls_back_without_leaking_key(monkeypatch):
    fake_provider(monkeypatch, [LLMError("auth", llm.redact(f"invalid key {FAKE_KEY}", Secret(FAKE_KEY)), 401)])
    judge = FaithfulnessJudge(provider="groq", api_key=FAKE_KEY)
    verdict = judge.score("q", CHUNKS, ANSWER)
    assert verdict.judge_backend == "heuristic" and verdict.fallback_reason == "auth"
    assert FAKE_KEY not in verdict.model_dump_json()
    assert FAKE_KEY not in repr(judge) and FAKE_KEY not in repr(judge.connection)


def test_verdicts_are_cached_per_model(monkeypatch, tmp_path):
    tracer = Tracer(str(tmp_path / "t.db"))
    calls = fake_provider(monkeypatch, [openai_reply(AUDIT), openai_reply(AUDIT)])
    judge = FaithfulnessJudge(provider="groq", api_key=FAKE_KEY, cache=tracer)
    first, second = judge.score("q", CHUNKS, ANSWER), judge.score("q", CHUNKS, ANSWER)
    assert len(calls) == 1 and second.cached and second.input_tokens == 0
    assert first.faithfulness_score == second.faithfulness_score
    FaithfulnessJudge(provider="groq", model="llama-3.3-70b-versatile", api_key=FAKE_KEY, cache=tracer).score(
        "q", CHUNKS, ANSWER)
    assert len(calls) == 2  # a different model is judged afresh
    raw = (tmp_path / "t.db").read_bytes()
    assert FAKE_KEY.encode() not in raw


def test_environment_picks_the_provider(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-abcdefghijklmnop")
    judge = FaithfulnessJudge()
    assert judge.active_backend == "llm" and judge.connection.provider == "openrouter"
    monkeypatch.setenv("ANCHOR_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setenv("ANCHOR_LLM_MODEL", "llama-3.1-8b-instant")
    assert FaithfulnessJudge().model == "groq/llama-3.1-8b-instant"


def test_bad_backend_name_rejected():
    with pytest.raises(ValueError):
        FaithfulnessJudge(backend="vibes")
    with pytest.raises(ValueError):
        Connection("nope", "m")
