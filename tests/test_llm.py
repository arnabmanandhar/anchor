import io
import json
import pickle
import urllib.error

import pytest

import anchor.llm as llm
from anchor.llm import Connection, LLMError, Secret

KEY = "sk-test-abcdefghijklmnopqrstuvwxyz"


def test_secret_never_shows_itself():
    s = Secret(KEY)
    assert KEY not in repr(s) and KEY not in str(s) and KEY not in f"{s}"
    assert s.hint == "••••wxyz" and s.reveal() == KEY
    with pytest.raises(TypeError):
        pickle.dumps(s)
    with pytest.raises(AttributeError):
        s._value = "x"
    with pytest.raises(ValueError):
        Secret("   ")
    conn = Connection("openai", "gpt-6-luna", s)
    assert KEY not in repr(conn) and KEY not in conn.name


def test_redact_removes_keys_and_bearer_tokens():
    s = Secret(KEY)
    assert KEY not in llm.redact(f"bad key {KEY}", s)
    assert "abcdefghijklmnop" not in llm.redact("Authorization: Bearer abcdefghijklmnopqrstu")
    assert "[redacted]" in llm.redact('api_key="zzzzzzzzzzzzzzzzzz"')


def test_extract_json_tolerates_fences_thinking_and_chatter():
    assert llm.extract_json('<think>hmm {"no": 1}</think>Sure! {"a": {"b": "}"}} done') == {"a": {"b": "}"}}
    assert llm.extract_json('```json\n{"x": [1, 2]}\n```') == {"x": [1, 2]}
    with pytest.raises(LLMError):
        llm.extract_json("no json here")


def test_validate_reports_problems():
    schema = {"type": "object", "required": ["v"], "properties": {"v": {"type": "string", "enum": ["a", "b"]},
                                                                  "n": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}
    assert llm.validate({"v": "a", "n": None}, schema) == []
    assert llm.validate({"v": "c"}, schema)
    assert llm.validate({"n": 3}, schema)
    assert llm.validate({"v": "a", "n": 3}, schema)


def _http_error(status, body, headers=None):
    return urllib.error.HTTPError("https://x", status, "err", headers or {}, io.BytesIO(json.dumps(body).encode()))


def test_http_errors_are_classified_and_scrubbed(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    secret = Secret(KEY)

    def opener(status, body):
        def _open(req, timeout):
            raise _http_error(status, body)
        return _open

    monkeypatch.setattr(llm.urllib.request, "urlopen", opener(401, {"error": {"message": f"Incorrect key {KEY}"}}))
    with pytest.raises(LLMError) as e:
        llm._post("https://api.example.com/v1/x", {}, {}, secret)
    assert e.value.kind == "auth" and e.value.status == 401 and KEY not in str(e.value)

    monkeypatch.setattr(llm.urllib.request, "urlopen", opener(404, {"error": "model not found"}))
    with pytest.raises(LLMError) as e:
        llm._post("https://api.example.com/v1/x", {}, {}, secret)
    assert e.value.kind == "not_found"


def test_rate_limits_are_retried(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    attempts = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _open(req, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise _http_error(429, {"error": {"message": "slow down"}}, {"Retry-After": "1"})
        return _Resp(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", _open)
    assert llm._post("https://api.example.com/v1/x", {}, {}, None) == {"ok": True}
    assert len(attempts) == 3


def test_openai_uses_max_completion_tokens_and_others_max_tokens(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "_post", lambda url, h, payload, s: seen.append(payload) or
                        {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}], "usage": {}})
    llm.complete_text(Connection("openai", "gpt-6-luna", Secret(KEY)), "s", "u")
    llm.complete_text(Connection("groq", "llama-3.1-8b-instant", Secret(KEY)), "s", "u")
    assert "max_completion_tokens" in seen[0] and "max_tokens" in seen[1]


def test_anthropic_protocol_shape(monkeypatch):
    seen = []

    def _post(url, headers, payload, secret):
        seen.append((url, headers, payload))
        return {"content": [{"type": "text", "text": "OK"}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 1}}

    monkeypatch.setattr(llm, "_post", _post)
    out = llm.complete_text(Connection("anthropic", "claude-haiku-5-5", Secret(KEY)), "sys", "hi")
    url, headers, payload = seen[0]
    assert url.endswith("/v1/messages") and headers["x-api-key"] == KEY and payload["system"] == "sys"
    assert out.text == "OK" and out.input_tokens == 3


def test_base_url_guard_blocks_private_and_plain_http():
    llm.check_base_url("http://localhost:11434/v1")  # fine for local use
    for bad in ("http://example.com/v1", "https://127.0.0.1/v1", "https://169.254.169.254/latest",
                "https://user:pw@api.example.com", "file:///etc/passwd"):
        with pytest.raises(LLMError):
            llm.check_base_url(bad, allow_private=False)


def test_connection_from_env_order_and_overrides():
    assert llm.connection_from_env({}) is None
    env = {"OPENAI_API_KEY": "sk-aaaaaaaaaaaaaaaa", "GROQ_API_KEY": "gsk_bbbbbbbbbbbbbbbb"}
    assert llm.connection_from_env(env).provider == "groq"  # free tiers come first
    env["ANCHOR_LLM_PROVIDER"], env["ANCHOR_LLM_MODEL"] = "openai", "gpt-6.1-sol"
    conn = llm.connection_from_env(env)
    assert (conn.provider, conn.model) == ("openai", "gpt-6.1-sol")


def test_key_format_hint():
    assert llm.key_format_warning("groq", "sk-ant-xyz") is not None
    assert llm.key_format_warning("groq", "gsk_xyz") is None
    assert llm.key_format_warning("mistral", "anything") is None


def test_provider_registry_is_neutral_and_complete():
    ids = list(llm.PROVIDERS)
    assert ids[0] == llm.DEFAULT_PROVIDER and llm.PROVIDERS[ids[0]].free_tier
    assert {"gemini", "groq", "openrouter", "huggingface", "openai", "xai", "anthropic", "custom"} <= set(ids)
    for p in llm.PROVIDERS.values():
        assert p.models and p.env_keys and p.base_url.startswith(("https://", "http://localhost"))


def test_bad_key_reported_as_400_is_auth(monkeypatch):
    def _open(req, timeout):
        raise _http_error(400, {"error": {"message": "API key not valid. Please pass a valid API key."}})

    monkeypatch.setattr(llm.urllib.request, "urlopen", _open)
    with pytest.raises(LLMError) as e:
        llm._post("https://api.example.com/v1/x", {}, {}, None)
    assert e.value.kind == "auth"
