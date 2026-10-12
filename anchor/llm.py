"""
Provider-neutral access to language models.

Anchor uses a model in three optional places: the faithfulness judge, the
sandbox answer generator, and question generation. All three go through
this module, so any provider works for any of them.

Providers
  Most providers speak the OpenAI chat-completions protocol, so one client
  covers Google Gemini, Groq, OpenRouter, Hugging Face, OpenAI, xAI (Grok),
  Mistral, and any compatible server you run yourself (Ollama, vLLM,
  LM Studio). Anthropic has its own protocol and its own small adapter.
  Only the standard library is used: no SDK is required for any provider.

Structured output
  Judge and question-generation calls need JSON in a fixed shape. Support
  differs by provider and model, so complete_json() negotiates: it asks for
  a strict JSON schema first, falls back to plain JSON mode, then to a
  prompt-only request, and validates the result against the schema in every
  case. One repair attempt is made when a reply doesn't validate.

Keys
  A key is wrapped in Secret, whose repr and str are masked and which
  refuses to be pickled, so it can't leak through a log line, a traceback,
  a cached object or session serialisation. It is revealed only to build
  the Authorization header of a request to that connection's own endpoint.
  Errors carry a category and HTTP status, plus the provider's message with
  the key scrubbed out; never the request headers.
"""

from __future__ import annotations

import ipaddress
import json
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

# ----------------------------------------------------------------- secrets


class Secret:
    """An API key that never shows itself by accident."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        value = (value or "").strip()
        if not value:
            raise ValueError("empty API key")
        object.__setattr__(self, "_value", value)

    def __setattr__(self, *_):
        raise AttributeError("Secret is immutable")

    def reveal(self) -> str:
        return self._value

    @property
    def hint(self) -> str:
        """Last four characters, for telling keys apart in the UI."""
        return "••••" + self._value[-4:] if len(self._value) > 8 else "••••"

    def __repr__(self) -> str:
        return f"Secret({self.hint})"

    __str__ = __repr__

    def __eq__(self, other) -> bool:
        return isinstance(other, Secret) and other._value == self._value

    def __hash__(self) -> int:
        return hash(("Secret", self._value))

    def __reduce__(self):
        raise TypeError("API keys can't be pickled or serialised")


def redact(text: str, *secrets: Optional[Secret]) -> str:
    """Remove secrets (and anything shaped like a bearer token) from text."""
    for s in secrets:
        if s is not None:
            text = text.replace(s.reveal(), s.hint)
    return re.sub(r"(?i)(bearer\s+|api[_-]?key[\"'=:\s]+)[A-Za-z0-9_\-\.]{12,}", r"\1[redacted]", text)


# --------------------------------------------------------------- providers


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    protocol: str  # "openai" or "anthropic"
    base_url: str
    models: tuple[str, ...]  # suggestions; the first is the default
    env_keys: tuple[str, ...]
    key_url: str
    free_tier: str = ""  # short note, empty if paid only
    key_prefixes: tuple[str, ...] = ()


# Ordered by how far a free account gets you, then alphabetically. Model
# lists are suggestions: any model id the provider serves can be typed in.
PROVIDERS: dict[str, Provider] = {p.id: p for p in (
    Provider("gemini", "Google Gemini", "openai", "https://generativelanguage.googleapis.com/v1beta/openai",
             ("gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"),
             ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "https://aistudio.google.com/apikey",
             "Free tier with daily request limits; no card needed", ("AIza",)),
    Provider("groq", "Groq", "openai", "https://api.groq.com/openai/v1",
             ("openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "llama-3.1-8b-instant"),
             ("GROQ_API_KEY",), "https://console.groq.com/keys",
             "Free tier, about 1,000 requests a day on most models", ("gsk_",)),
    Provider("openrouter", "OpenRouter", "openai", "https://openrouter.ai/api/v1",
             ("openai/gpt-oss-120b:free", "meta-llama/llama-3.3-70b-instruct:free", "qwen/qwen3-coder:free"),
             ("OPENROUTER_API_KEY",), "https://openrouter.ai/keys",
             "Free ':free' models, 50 requests a day", ("sk-or-",)),
    Provider("huggingface", "Hugging Face", "openai", "https://router.huggingface.co/v1",
             ("openai/gpt-oss-120b", "deepseek-ai/DeepSeek-R1", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
             ("HF_TOKEN", "HUGGINGFACE_API_KEY"), "https://huggingface.co/settings/tokens",
             "Small monthly credit on free accounts", ("hf_",)),
    Provider("mistral", "Mistral", "openai", "https://api.mistral.ai/v1",
             ("mistral-small-latest", "mistral-large-latest"),
             ("MISTRAL_API_KEY",), "https://console.mistral.ai/api-keys",
             "Free experiment tier (opt-in to data use)"),
    Provider("openai", "OpenAI", "openai", "https://api.openai.com/v1",
             ("gpt-6-luna", "gpt-6.1-sol", "gpt-6-astra"),
             ("OPENAI_API_KEY",), "https://platform.openai.com/api-keys", "", ("sk-",)),
    Provider("xai", "xAI Grok", "openai", "https://api.x.ai/v1",
             ("grok-4.7",), ("XAI_API_KEY",), "https://console.x.ai", "", ("xai-",)),
    Provider("anthropic", "Anthropic", "anthropic", "https://api.anthropic.com/v1",
             ("claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"),
             ("ANTHROPIC_API_KEY",), "https://console.anthropic.com/settings/keys", "", ("sk-ant-",)),
    Provider("custom", "Custom (OpenAI-compatible)", "openai", "http://localhost:11434/v1",
             ("llama3.2",), ("ANCHOR_LLM_API_KEY",), "",
             "Local servers such as Ollama, vLLM or LM Studio cost nothing"),
)}
DEFAULT_PROVIDER = "gemini"


def key_format_warning(provider_id: str, key: str) -> Optional[str]:
    """A gentle hint when a key doesn't look like the provider's keys."""
    p = PROVIDERS[provider_id]
    if p.key_prefixes and not key.strip().startswith(p.key_prefixes):
        return f"{p.label} keys usually start with {' or '.join(p.key_prefixes)}. Check you picked the right provider."
    return None


# -------------------------------------------------------------- connection


class LLMError(RuntimeError):
    """A failed model call. kind is one of auth, rate_limit, bad_request,
    not_found, server, network, refused, truncated, parse, blocked."""

    def __init__(self, kind: str, message: str = "", status: Optional[int] = None):
        self.kind, self.status = kind, status
        super().__init__(f"{kind}{f' (HTTP {status})' if status else ''}{': ' + message if message else ''}")


@dataclass(frozen=True)
class Connection:
    provider: str
    model: str
    key: Optional[Secret] = field(default=None, repr=False)
    base_url: Optional[str] = None
    label: Optional[str] = None
    price_per_mtok: Optional[tuple[float, float]] = None  # (input, output) USD, if known

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}; choose from {sorted(PROVIDERS)}")
        if not self.model.strip():
            raise ValueError("model is required")

    @property
    def spec(self) -> Provider:
        return PROVIDERS[self.provider]

    @property
    def url(self) -> str:
        return (self.base_url or self.spec.base_url).rstrip("/")

    @property
    def name(self) -> str:
        """Safe display name: provider and model, never the key."""
        return self.label or f"{self.spec.label} · {self.model}"

    def __repr__(self) -> str:
        return f"Connection({self.provider}/{self.model}{', ' + self.key.hint if self.key else ''})"

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        if not self.price_per_mtok:
            return 0.0
        return (input_tokens * self.price_per_mtok[0] + output_tokens * self.price_per_mtok[1]) / 1_000_000


def connection_from_env(environ=None) -> Optional[Connection]:
    """ANCHOR_LLM_PROVIDER / ANCHOR_LLM_MODEL / ANCHOR_LLM_API_KEY /
    ANCHOR_LLM_BASE_URL if set; otherwise the first provider (in PROVIDERS
    order) whose own key variable is set. None when no key is available."""
    env = os.environ if environ is None else environ
    model = env.get("ANCHOR_LLM_MODEL") or env.get("ANCHOR_JUDGE_MODEL")
    chosen = env.get("ANCHOR_LLM_PROVIDER")
    if chosen:
        p = PROVIDERS[chosen]
        key = env.get("ANCHOR_LLM_API_KEY") or next((env[k] for k in p.env_keys if env.get(k)), None)
        if not key and p.id != "custom":
            return None
        return Connection(p.id, model or p.models[0], Secret(key) if key else None, env.get("ANCHOR_LLM_BASE_URL"))
    for p in PROVIDERS.values():
        if p.id == "custom":
            continue
        key = next((env[k] for k in p.env_keys if env.get(k)), None)
        if key:
            return Connection(p.id, model or p.models[0], Secret(key))
    return None


# ------------------------------------------------------------- URL safety


def check_base_url(url: str, allow_private: bool = True) -> None:
    """Reject URLs that would send a key somewhere unexpected. Public
    deployments call this with allow_private=False, so a visitor can't point
    the server at its own network (SSRF) or downgrade to plain http."""
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        raise LLMError("blocked", "base URL must be http(s)://host/…")
    if parsed.username or parsed.password:
        raise LLMError("blocked", "base URL must not contain credentials")
    if allow_private:
        return
    if parsed.scheme != "https":
        raise LLMError("blocked", "only https endpoints are allowed here")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise LLMError("blocked", "host does not resolve") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise LLMError("blocked", "private or local addresses are not allowed here")


# ------------------------------------------------------------------- HTTP

TIMEOUT_S = float(os.environ.get("ANCHOR_LLM_TIMEOUT", "60"))
MAX_RETRIES = 3


def _post(url: str, headers: dict[str, str], payload: dict, secret: Optional[Secret]) -> dict:
    if not url.startswith(("https://", "http://")):
        raise LLMError("blocked", "only http(s) endpoints are supported")
    body = json.dumps(payload).encode()
    for attempt in range(MAX_RETRIES + 1):
        req = urllib.request.Request(url, data=body, method="POST",  # noqa: S310 (scheme checked above)
                                     headers={"Content-Type": "application/json", "User-Agent": "anchor-rag-eval",
                                              **headers})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # noqa: S310
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            status = e.code
            detail = ""
            try:
                detail = _error_message(json.loads(e.read().decode() or "{}"))
            except Exception:  # noqa: S110 (an unreadable error body just means no detail)
                pass
            detail = redact(str(detail or ""), secret)[:300]
            retryable = status == 429 or status >= 500
            if retryable and attempt < MAX_RETRIES:
                wait = e.headers.get("Retry-After") if e.headers else None
                time.sleep(min(float(wait), 30.0) if wait and wait.replace(".", "").isdigit() else 2 ** attempt + random.random())
                continue
            kind = {400: "bad_request", 401: "auth", 403: "auth", 404: "not_found", 422: "bad_request",
                    429: "rate_limit"}.get(status, "server" if status >= 500 else "bad_request")
            if kind == "bad_request" and re.search(r"api[ _-]?key|unauthori[sz]ed|authenticat|credential", detail, re.I):
                kind = "auth"  # some providers (Gemini) answer a bad key with 400
            raise LLMError(kind, detail, status) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt + random.random())
                continue
            raise LLMError("network", type(e).__name__) from None
    raise LLMError("network", "retries exhausted")


def _error_message(body) -> str:
    """The human-readable message from a provider's error body, whatever its shape."""
    if isinstance(body, list):
        body = body[0] if body else {}
    if not isinstance(body, dict):
        return ""
    err = body.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or "")
    return str(err or body.get("message") or "")


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


def _openai_chat(conn: Connection, system: str, user: str, max_tokens: int, response_format: Optional[dict]) -> Completion:
    headers = {"Authorization": f"Bearer {conn.key.reveal()}"} if conn.key else {}
    # OpenAI's current models take max_completion_tokens; compatible servers still expect max_tokens.
    limit = "max_completion_tokens" if conn.provider == "openai" else "max_tokens"
    payload = {"model": conn.model, limit: max_tokens,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if response_format:
        payload["response_format"] = response_format
    data = _post(f"{conn.url}/chat/completions", headers, payload, conn.key)
    try:
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        raise LLMError("parse", "unexpected response shape") from None
    if choice.get("finish_reason") == "length" and not text.strip():
        raise LLMError("truncated")
    if choice.get("finish_reason") == "content_filter":
        raise LLMError("refused")
    usage = data.get("usage") or {}
    return Completion(text, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0))


def _anthropic_messages(conn: Connection, system: str, user: str, max_tokens: int, schema: Optional[dict]) -> Completion:
    headers = {"x-api-key": conn.key.reveal() if conn.key else "", "anthropic-version": "2023-06-01"}
    payload = {"model": conn.model, "max_tokens": max_tokens, "system": system,
               "messages": [{"role": "user", "content": user}]}
    if schema:
        payload["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    data = _post(f"{conn.url}/messages", headers, payload, conn.key)
    if data.get("stop_reason") == "refusal":
        raise LLMError("refused")
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    if data.get("stop_reason") == "max_tokens" and not text.strip():
        raise LLMError("truncated")
    usage = data.get("usage") or {}
    return Completion(text, int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0))


def complete_text(conn: Connection, system: str, user: str, max_tokens: int = 2048) -> Completion:
    """Plain text completion."""
    if conn.spec.protocol == "anthropic":
        c = _anthropic_messages(conn, system, user, max_tokens, None)
    else:
        c = _openai_chat(conn, system, user, max_tokens, None)
    c.text = strip_reasoning(c.text).strip()
    return c


# ------------------------------------------------------------ JSON output


def strip_reasoning(text: str) -> str:
    """Drop <think>…</think> blocks some open reasoning models emit inline."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def extract_json(text: str):
    """Parse the first JSON object in text, tolerating code fences and chatter."""
    text = strip_reasoning(text).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise LLMError("parse", "no JSON object in the reply")


def validate(value, schema: dict, path: str = "$") -> list[str]:
    """Minimal JSON-schema check covering what Anchor's schemas use:
    type, properties, required, items, enum, anyOf."""
    errors: list[str] = []
    if "anyOf" in schema:
        if all(validate(value, s, path) for s in schema["anyOf"]):
            errors.append(f"{path}: matches none of the allowed shapes")
        return errors
    t = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None),
             "integer": int, "number": (int, float)}
    if t and not isinstance(value, types[t]) or (t in ("integer", "number") and isinstance(value, bool)):
        return [f"{path}: expected {t}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: must be one of {schema['enum']}")
    if t == "object":
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}.{name}: missing")
        for name, sub in schema.get("properties", {}).items():
            if name in value:
                errors += validate(value[name], sub, f"{path}.{name}")
    if t == "array" and "items" in schema:
        for i, item in enumerate(value):
            errors += validate(item, schema["items"], f"{path}[{i}]")
    return errors


def complete_json(conn: Connection, system: str, user: str, schema: dict, max_tokens: int = 4096,
                  name: str = "result") -> tuple[dict, Completion]:
    """A reply that parses and validates against schema, negotiating the
    strongest structured-output mode the provider accepts."""
    schema_hint = (f"{system}\n\nReply with a single JSON object and nothing else. It must match this JSON "
                   f"schema:\n{json.dumps(schema)}")
    if conn.spec.protocol == "anthropic":
        modes = [("schema", lambda: _anthropic_messages(conn, system, user, max_tokens, schema)),
                 ("prompt", lambda: _anthropic_messages(conn, schema_hint, user, max_tokens, None))]
    else:
        strict = {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}
        modes = [("schema", lambda: _openai_chat(conn, system, user, max_tokens, strict)),
                 ("json", lambda: _openai_chat(conn, schema_hint, user, max_tokens, {"type": "json_object"})),
                 ("prompt", lambda: _openai_chat(conn, schema_hint, user, max_tokens, None))]
    last: Optional[LLMError] = None
    for _mode, call in modes:
        try:
            completion = call()
        except LLMError as e:
            if e.kind == "bad_request":  # mode unsupported by this provider/model: try the next one
                last = e
                continue
            raise
        for repair in (False, True):
            try:
                data = extract_json(completion.text)
                problems = validate(data, schema)
                if not problems:
                    return data, completion
                last = LLMError("parse", "; ".join(problems[:3]))
            except LLMError as e:
                last = e
            if repair:
                break
            fix = f"{user}\n\nYour previous reply was not valid ({last}). Reply again with only the JSON object."
            prev = completion
            completion = (_anthropic_messages(conn, schema_hint, fix, max_tokens, None)
                          if conn.spec.protocol == "anthropic" else _openai_chat(conn, schema_hint, fix, max_tokens, None))
            completion.input_tokens += prev.input_tokens
            completion.output_tokens += prev.output_tokens
        break
    raise last or LLMError("parse")


def ping(conn: Connection) -> str:
    """Cheapest possible call, to check a key and model work. Returns the
    model's reply (a few characters) or raises LLMError."""
    # Generous limit: reasoning models spend tokens thinking before they answer.
    return complete_text(conn, "Reply with the single word OK.", "Ping", max_tokens=512).text[:40]
