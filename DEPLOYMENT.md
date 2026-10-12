# Deploying the Anchor dashboard

The dashboard (`dashboard/app.py`) is designed to be hosted publicly on
**Streamlit Community Cloud** without ever exposing an API key, and to work
fully with **no key at all**.

## Who pays for what

| Situation | What runs | Who pays |
|---|---|---|
| Visitor adds their own model in the sidebar (any provider) | LLM judge, generator and question writer on that model | The visitor |
| Owner set a provider key in Secrets | A "built-in" model, capped per session and per day | The owner |
| No key anywhere | Offline heuristic judge, extractive generator, template questions | Nobody |

The recommended public setup is either **no server key** (zero cost) or a
**free-tier key** (Google Gemini by default) with the caps below. Visitors
who want more can add their own key from any provider; several have free
tiers, and the sidebar links to each provider's key page.

## How keys are kept safe

All key handling lives in two files, so it can be audited in one sitting:
`anchor/llm.py` (the client) and `dashboard/connections.py` (the session
store).

**The owner's key**

- Read from Streamlit Secrets or the environment on the server. It is never
  rendered, never put in widget state, and the built-in connection only
  ever shows its provider and model.
- Only used with the provider and model the owner configured
  (`ANCHOR_SERVER_PROVIDER`, `ANCHOR_SERVER_MODEL`). A visitor cannot point
  it at another model or endpoint.
- Capped per browser session (`ANCHOR_SERVER_KEY_SESSION_LIMIT`, default 25
  calls) and across all sessions per day (`ANCHOR_SERVER_KEY_DAILY_LIMIT`,
  default 500). When the cap is reached the built-in model disappears from
  the pickers and the app keeps working offline.

**Visitors' keys**

- Entered in a password field inside a form that clears on submit; the raw
  value is removed from widget state immediately and the browser is asked
  not to autofill it.
- Wrapped in `anchor.llm.Secret`, which masks itself in `repr`, `str` and
  f-strings, can't be pickled, and is immutable. The UI only ever shows the
  last four characters.
- Kept in server memory for that browser session only. Sessions never see
  each other's connections. Keys are forgotten after
  `ANCHOR_KEY_TTL_MINUTES` of inactivity (default 60), when the tab closes,
  or when the visitor clicks **Forget all keys**.
- Sent only to the endpoint of the provider the visitor picked, over HTTPS
  with certificate verification. Nothing in the browser talks to a provider
  directly.

**Everywhere**

- Keys are never written to the trace database, the judge cache, logs or a
  URL. The judge cache key is a hash of the prompt version, model, query,
  chunks and answer; it never includes the key.
- Provider error messages are scrubbed (`llm.redact`) of the key and of any
  bearer token before they can reach a page or a verdict. Verdicts record
  only an error kind such as `auth` or `rate_limit`.
- `.streamlit/config.toml` sets `showErrorDetails = "type"`, so an
  unexpected exception shows its type, not a traceback with local values.
- With `ANCHOR_PUBLIC=1`, a custom base URL must be a public `https` host
  with no credentials in it. Private, loopback and link-local addresses
  (including cloud metadata at 169.254.169.254) are refused, so nobody can
  aim the server at its own network.
- With `ANCHOR_PUBLIC=1`, each browser session also gets its own temporary
  trace database. The database path is fixed on the server (`ANCHOR_DB`),
  so visitors can't point the app at other files.
- LLM work per click is capped (`ANCHOR_MAX_LLM_CASES`, default 25).
- `tests/test_llm.py`, `tests/test_judge.py`, `tests/test_dashboard.py` and
  `tests/test_repo_hygiene.py` check all of this, including that a pasted
  key never appears in the rendered page or in session state.

## Deploy on Streamlit Community Cloud

1. Push the repo to GitHub (public or private).
2. Go to https://share.streamlit.io, choose **Create app**, then **Deploy a public app from GitHub**.
3. Repository `arnabmanandhar/anchor`, branch `main`, main file path `dashboard/app.py`.
4. Under **Advanced settings → Secrets**, add `ANCHOR_PUBLIC = "1"`. This
   gives every visitor a private workspace. Optionally add one provider key
   from `.streamlit/secrets.toml.example`, for example
   `GEMINI_API_KEY = "..."`, plus the caps you want. Skip the key for a
   zero-cost demo.
5. Click **Deploy**. Each visitor's workspace starts seeded with the
   bundled demo, two retriever variants ready to compare.

Streamlit Cloud installs `requirements.txt` automatically. The app's disk
is ephemeral, so a restart re-seeds the demo, which is deterministic.

## Settings

| Setting | Default | Purpose |
|---|---|---|
| `ANCHOR_PUBLIC` | unset | `1` for per-session databases and the endpoint guard |
| `GEMINI_API_KEY`, `GROQ_API_KEY`, … | unset | Owner key for the built-in model (any one provider) |
| `ANCHOR_SERVER_PROVIDER` | first provider with a key | Which provider the built-in model uses |
| `ANCHOR_SERVER_MODEL` | provider's default | Which model the built-in model uses |
| `ANCHOR_SERVER_KEY_SESSION_LIMIT` | `25` | Built-in calls per browser session |
| `ANCHOR_SERVER_KEY_DAILY_LIMIT` | `500` | Built-in calls per day, all sessions |
| `ANCHOR_KEY_TTL_MINUTES` | `60` | Idle minutes before visitors' keys are forgotten |
| `ANCHOR_MAX_LLM_CASES` | `25` | LLM cases per click |
| `ANCHOR_LLM_WORKERS` | `4` | Parallel LLM calls per run |
| `ANCHOR_LLM_TIMEOUT` | `60` | Seconds per provider request |

## Pre-deploy security checklist

Run these from the repo root before every public deploy:

```bash
# 1. No real keys in tracked files or anywhere in git history
KEYS='(sk-ant-|sk-proj-|sk-or-v1-|gsk_|xai-|hf_)[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{35}'
git grep -nE "$KEYS" || echo "clean: working tree"
git log --all -p | grep -nE "$KEYS" || echo "clean: history"

# 2. Secrets files are ignored and not tracked
git check-ignore -v .env .streamlit/secrets.toml
git ls-files | grep -E "(^|/)(\.env|secrets\.toml)$" && echo "TRACKED SECRET FILE!" || echo "clean: none tracked"

# 3. Everything works with no key (tests clear provider keys themselves)
pytest -q
env -u GEMINI_API_KEY -u GROQ_API_KEY -u OPENAI_API_KEY -u ANTHROPIC_API_KEY anchor demo
```

`tests/test_repo_hygiene.py` runs check 1 on the working tree as part of
the suite, so CI fails if a key-shaped string is committed.

If a real key ever appears in git history, even in a deleted commit,
**revoke it at the provider first**. Rewriting history doesn't un-leak a
key that has already been pushed.

After deploying:

- Open the app in a private window. The sidebar should say everything runs
  offline, or show only the built-in model if you set a server key.
- Open the browser's Network tab and run an evaluation. No request from
  your browser should go to a model provider, because calls happen
  server-side, and no response should contain a key.
- Add a model with a deliberately wrong key and click **Test**. You should
  see "The key was rejected." and the key should appear nowhere on the page.

## What visitors can do

- **Evaluate:** build a pipeline from the sample corpus or their own file,
  upload their own pipeline's logged outputs, or score a single answer.
- **Results and Compare:** fixed 0–1 charts, per-question regressions,
  significance tests, and an interactive regression gate.
- **Inspect:** the claim-by-claim audit, ranked chunks marked relevant or
  cited, and trace timings.
- **Learn the metrics:** the formulas, with worked examples.

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for a full walkthrough.
