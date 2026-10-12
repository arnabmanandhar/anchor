import json
import sys
import types

import pytest

from anchor.judge import FaithfulnessJudge, claim_overlap
from anchor.schema import Chunk

CHUNKS = [
    Chunk(id="c1", text="Nepal's federal structure divides the country into seven provinces."),
    Chunk(id="c2", text="The Supreme Court is the highest judicial authority."),
]
ANSWER = "Nepal has seven provinces. Each province elects a governor."
FAKE_KEY = "sk-ant-test-0000-do-not-leak"


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def block_anthropic(monkeypatch):
    """Make any attempt to import the SDK (i.e. any network path) fail loudly."""

    class _Blocker(types.ModuleType):
        def __getattr__(self, name):
            raise AssertionError("the heuristic path must not touch the anthropic SDK")

    monkeypatch.setitem(sys.modules, "anthropic", _Blocker("anthropic"))


def _fake_sdk(monkeypatch, payload=None, exc=None, stop_reason="end_turn"):
    """Install a stand-in `anthropic` module whose client returns `payload`."""
    calls = []

    class _Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            if exc:
                raise exc
            return types.SimpleNamespace(
                stop_reason=stop_reason,
                content=[types.SimpleNamespace(type="text", text=json.dumps(payload))],
                usage=types.SimpleNamespace(input_tokens=1000, output_tokens=200),
            )

    class _Client:
        def __init__(self, api_key=None):
            calls.append({"api_key": api_key})
            self.messages = _Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Client))
    return calls


def test_no_key_uses_heuristic_without_network(no_key, block_anthropic):
    judge = FaithfulnessJudge()
    assert judge.active_backend == "heuristic"
    verdict = judge.score("How many provinces?", CHUNKS, ANSWER)
    assert verdict.judge_backend == "heuristic"
    assert [c.supported for c in verdict.claims] == [True, False]
    assert verdict.claims[0].chunk_id == "c1"
    assert verdict.faithfulness_score == 0.5


def test_empty_string_key_counts_as_no_key(monkeypatch, block_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert FaithfulnessJudge().active_backend == "heuristic"


def test_forced_heuristic_ignores_key(block_anthropic):
    judge = FaithfulnessJudge(api_key=FAKE_KEY, backend="heuristic")
    assert judge.score("q", CHUNKS, ANSWER).judge_backend == "heuristic"


def test_overlap_coefficient_is_share_of_claim_words():
    # content words {nepal, seven, provinces} -> all 3 in the chunk
    assert claim_overlap("Nepal has seven provinces.", CHUNKS[0].text) == 1.0
    # {nepal, seven, governors} -> 2 of 3 in the chunk
    assert claim_overlap("Nepal has seven governors.", CHUNKS[0].text) == pytest.approx(2 / 3)
    assert claim_overlap("", CHUNKS[0].text) == 0.0


def test_llm_verdict_is_derived_from_claims(no_key, monkeypatch):
    calls = _fake_sdk(
        monkeypatch,
        {
            "claims": [
                {"claim": "Nepal has seven provinces", "supported": True, "chunk_id": "c1", "note": None},
                {"claim": "Each province elects a governor", "supported": False, "chunk_id": None, "note": "not in any chunk"},
            ],
            "reasoning": "One of two claims grounded.",
        },
    )
    judge = FaithfulnessJudge(api_key=FAKE_KEY)
    verdict = judge.score("How many provinces?", CHUNKS, ANSWER)
    assert verdict.judge_backend == "llm"
    assert verdict.faithfulness_score == 0.5
    assert verdict.input_tokens == 1000 and verdict.output_tokens == 200
    assert judge.estimate_cost(verdict) > 0
    assert calls[0]["api_key"] == FAKE_KEY
    assert calls[1]["output_config"]["format"]["type"] == "json_schema"


def test_llm_citation_of_unretrieved_chunk_is_downgraded(no_key, monkeypatch):
    _fake_sdk(
        monkeypatch,
        {"claims": [{"claim": "x", "supported": True, "chunk_id": "c99", "note": None}], "reasoning": "r"},
    )
    verdict = FaithfulnessJudge(api_key=FAKE_KEY).score("q", CHUNKS, "x")
    assert verdict.claims[0].supported is False
    assert verdict.faithfulness_score == 0.0


def test_llm_failure_falls_back_without_leaking_key(no_key, monkeypatch):
    _fake_sdk(monkeypatch, exc=RuntimeError(f"401 invalid x-api-key {FAKE_KEY}"))
    judge = FaithfulnessJudge(api_key=FAKE_KEY)
    verdict = judge.score("q", CHUNKS, ANSWER)
    assert verdict.judge_backend == "heuristic"
    assert verdict.fallback_reason == "RuntimeError"
    assert FAKE_KEY not in verdict.model_dump_json()
    assert FAKE_KEY not in repr(judge)


def test_refusal_falls_back(no_key, monkeypatch):
    _fake_sdk(monkeypatch, {"claims": [], "reasoning": ""}, stop_reason="refusal")
    assert FaithfulnessJudge(api_key=FAKE_KEY).score("q", CHUNKS, ANSWER).judge_backend == "heuristic"


def test_bad_backend_name_rejected():
    with pytest.raises(ValueError):
        FaithfulnessJudge(backend="vibes")
