"""
Model connections for the dashboard: which provider, model and key each
browser session can use. All key handling lives here so it can be audited
in one place.

Rules
  - A visitor's key lives only in this server process's memory, inside that
    visitor's own Streamlit session, wrapped in anchor.llm.Secret. It is
    never written to disk, the trace database, the judge cache, logs, URLs
    or the page, and it is never sent anywhere except the provider endpoint
    the visitor chose.
  - The key field is in a form that clears on submit, so the raw value
    doesn't linger in widget state; the browser is asked not to autofill it.
  - Keys are forgotten after ANCHOR_KEY_TTL_MINUTES of inactivity (default
    60), when the session ends, or when the visitor clicks "Forget all keys".
  - Each session can hold several connections (say, a free Gemini key for
    judging and an OpenAI key for generation) and picks one per task.
    Sessions never see each other's connections.
  - The owner's key (server secrets) is offered as a "built-in" connection
    that is never shown: capped per session, capped per day across all
    sessions, and only ever used with its configured provider and model.
  - On a public deployment (ANCHOR_PUBLIC=1) custom endpoints must be public
    https hosts, so nobody can aim the server at its own network.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import date
from typing import Optional

import streamlit as st

from anchor import llm
from anchor.llm import Connection, LLMError, Secret

STORE = "_anchor_connections"
SERVER_ID = "server"


def _setting(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        try:
            value = st.secrets.get(name)
        except Exception:  # no secrets.toml locally is fine
            value = None
    return str(value) if value is not None else default


def public_mode() -> bool:
    return _setting("ANCHOR_PUBLIC").lower() in ("1", "true", "yes")


def session_limit() -> int:
    return int(_setting("ANCHOR_SERVER_KEY_SESSION_LIMIT", "25"))


def daily_limit() -> int:
    return int(_setting("ANCHOR_SERVER_KEY_DAILY_LIMIT", "500"))


def ttl_seconds() -> float:
    return float(_setting("ANCHOR_KEY_TTL_MINUTES", "60")) * 60


# ------------------------------------------------- the owner's (server) key

_daily = {"day": None, "calls": 0}
_daily_lock = threading.Lock()


def _server_connection() -> Optional[Connection]:
    """The owner's connection from secrets/env, or None. Provider and model
    are fixed by the owner (ANCHOR_SERVER_PROVIDER / ANCHOR_SERVER_MODEL);
    by default the first free-tier provider with a key set."""
    wanted = _setting("ANCHOR_SERVER_PROVIDER")
    candidates = [llm.PROVIDERS[wanted]] if wanted in llm.PROVIDERS else list(llm.PROVIDERS.values())
    for spec in candidates:
        if spec.id == "custom":
            continue
        key = next((_setting(k) for k in spec.env_keys if _setting(k)), None)
        if key:
            return Connection(spec.id, _setting("ANCHOR_SERVER_MODEL") or spec.models[0], Secret(key),
                              label=f"Built-in · {spec.label} {_setting('ANCHOR_SERVER_MODEL') or spec.models[0]}")
    return None


def server_calls_left() -> int:
    with _daily_lock:
        today = date.today()
        if _daily["day"] != today:
            _daily.update(day=today, calls=0)
        day_left = daily_limit() - _daily["calls"]
    return max(0, min(session_limit() - st.session_state.get("server_calls", 0), day_left))


def charge(conn_id: Optional[str], calls: int) -> None:
    """Count calls made on the built-in key (other connections are the visitor's own)."""
    if conn_id != SERVER_ID or calls <= 0:
        return
    st.session_state.server_calls = st.session_state.get("server_calls", 0) + calls
    with _daily_lock:
        _daily["calls"] += calls


# ------------------------------------------------------- session store


def _store() -> dict[str, dict]:
    store = st.session_state.setdefault(STORE, {})
    now = time.time()
    last = st.session_state.get("_anchor_conn_seen", now)
    if store and now - last > ttl_seconds():
        store.clear()
        st.session_state["_anchor_keys_expired"] = True
    st.session_state["_anchor_conn_seen"] = now
    return store


def add(provider: str, model: str, key: str, base_url: str = "", label: str = "") -> str:
    """Validate and store a connection; returns its id. Raises ValueError/LLMError."""
    spec = llm.PROVIDERS[provider]
    model = (model or spec.models[0]).strip()
    base = base_url.strip() or None
    if provider == "custom":
        if not base:
            raise ValueError("a custom provider needs a base URL")
        llm.check_base_url(base, allow_private=not public_mode())
    elif base:
        raise ValueError("only the custom provider takes a base URL")
    secret = Secret(key) if key.strip() else None
    if secret is None and provider != "custom":
        raise ValueError(f"{spec.label} needs an API key")
    conn = Connection(provider, model, secret, base, label.strip() or None)
    conn_id = uuid.uuid4().hex[:8]
    _store()[conn_id] = {"conn": conn, "added": time.time()}
    return conn_id


def remove(conn_id: str) -> None:
    _store().pop(conn_id, None)


def forget_all() -> None:
    _store().clear()
    for k in list(st.session_state.keys()):
        if str(k).startswith("conn_form"):
            del st.session_state[k]


def available() -> list[tuple[str, Connection]]:
    """(id, connection) pairs usable right now: the visitor's own, then the built-in one."""
    out = [(cid, item["conn"]) for cid, item in _store().items()]
    server = _server_connection()
    if server and server_calls_left() > 0:
        out.append((SERVER_ID, server))
    return out


def get(conn_id: Optional[str]) -> Optional[Connection]:
    return dict(available()).get(conn_id) if conn_id else None


def name(conn_id: Optional[str]) -> str:
    conn = get(conn_id)
    if conn is None:
        return "—"
    hint = f" ({conn.key.hint})" if conn.key and conn_id != SERVER_ID else ""
    return conn.name + hint


def budget_problem(conn_ids: list[Optional[str]], calls_each: int) -> Optional[str]:
    """A reason the planned calls can't run on the built-in key, or None."""
    needed = calls_each * sum(1 for c in conn_ids if c == SERVER_ID)
    if needed and needed > server_calls_left():
        return (f"the built-in model has {server_calls_left()} calls left for this session; "
                f"this needs {needed}. Add your own key in the sidebar (free tiers work)")
    return None


def test(conn_id: str) -> tuple[bool, str]:
    conn = get(conn_id)
    if conn is None:
        return False, "not found"
    try:
        llm.ping(conn)
        charge(conn_id, 1)
        return True, "works"
    except LLMError as e:
        return False, {"auth": "the key was rejected", "not_found": "the model id wasn't found",
                       "rate_limit": "rate limited; try again shortly", "network": "couldn't reach the provider",
                       "blocked": str(e)}.get(e.kind, f"the provider said: {str(e).split(': ', 1)[-1]}")


# ------------------------------------------------------------------- UI


def _provider_label(pid: str) -> str:
    spec = llm.PROVIDERS[pid]
    return f"{spec.label} · free tier" if spec.free_tier else spec.label


def render_sidebar() -> None:
    """The 'Models' section of the sidebar."""
    store = _store()
    st.markdown("**Models**")
    if st.session_state.pop("_anchor_keys_expired", False):
        st.caption(":material/schedule: Your keys were forgotten after a period of inactivity.")
    conns = available()
    if not conns:
        st.caption("Everything runs offline. Add a model to judge claims with an LLM, generate answers or "
                   "write test questions. Free tiers from Google Gemini, Groq or OpenRouter are enough.")
    for cid, conn in conns:
        with st.container(border=True):
            st.markdown(f"**{conn.spec.label}** · `{conn.model}`")
            if cid == SERVER_ID:
                st.caption(f"Built in, shared · {server_calls_left()} calls left this session")
                continue
            st.caption(f"Key {conn.key.hint if conn.key else 'none'} · this session only")
            b1, b2 = st.columns(2)
            if b1.button("Test", key=f"test::{cid}", width="stretch", help="Send one tiny request to check the key and model"):
                ok, msg = test(cid)
                (st.success if ok else st.error)(msg.capitalize() + ".")
            if b2.button("Remove", key=f"rm::{cid}", width="stretch", help="Forget this key now"):
                remove(cid)
                st.rerun()

    with st.expander("Add a model", icon=":material/add:", expanded=not store):
        provider = st.selectbox("Provider", list(llm.PROVIDERS), format_func=_provider_label, key="conn_form_provider")
        spec = llm.PROVIDERS[provider]
        if spec.free_tier:
            st.caption(spec.free_tier + ".")
        with st.form("conn_form", clear_on_submit=True, border=False):
            model = st.selectbox("Model", spec.models, accept_new_options=True, key=f"conn_form_model::{provider}",
                                 help="Pick a suggestion or type any model id this provider serves.")
            key = st.text_input("API key", type="password", autocomplete="off", key="conn_form_key",
                                placeholder="Optional for local servers" if provider == "custom" else "Paste your key",
                                help="Kept in server memory for this browser session only. Never saved, logged or "
                                     "shown again, and only sent to the provider you picked.")
            base = st.text_input("Base URL", key="conn_form_base", placeholder=spec.base_url) if provider == "custom" else ""
            submitted = st.form_submit_button("Add model", icon=":material/add_link:", width="stretch")
        if spec.key_url:
            st.caption(f"[Get a {spec.label} key]({spec.key_url})")
        if submitted:
            # Drop the raw key from widget state right away, whatever happens next.
            st.session_state.pop("conn_form_key", None)
            try:
                warning = llm.key_format_warning(provider, key) if key else None
                add(provider, model or "", key or "", base or "")
                if warning:
                    st.session_state["_anchor_key_warning"] = warning
                st.rerun()
            except (ValueError, LLMError) as e:
                st.error(str(e))
        if "_anchor_key_warning" in st.session_state:
            st.warning(st.session_state.pop("_anchor_key_warning"))
    if store:
        if st.button("Forget all keys", icon=":material/key_off:", width="stretch"):
            forget_all()
            st.rerun()
    st.caption(":material/shield_lock: Keys stay in this browser session's server memory. "
               f"They're forgotten after {int(ttl_seconds() // 60)} idle minutes or when you close the tab.")


def picker(label: str, offline_label: str, key: str, help_text: str = "", container=st,
           prefer_model: bool = True) -> Optional[str]:
    """Selectbox of the offline option plus every connection. Returns a
    connection id, or None for the offline option. With prefer_model, a
    connection is preselected when one exists (used for the judge); without
    it the offline option stays the default (used where each case costs a call)."""
    options = [None] + [cid for cid, _ in available()]
    seen = st.session_state.get(f"{key}::seen", [])
    added = [c for c in options[1:] if c not in seen]
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = options[1] if prefer_model and len(options) > 1 else None
    elif prefer_model and added and st.session_state[key] is None:
        st.session_state[key] = added[0]  # a model was just added: use it rather than staying offline
    st.session_state[f"{key}::seen"] = options[1:]
    labels = {None: offline_label, **{c: name(c) for c in options[1:]}}
    return container.selectbox(label, options, key=key, help=help_text, format_func=labels.get)
