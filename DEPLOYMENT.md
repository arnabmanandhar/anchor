# Deploying the Anchor dashboard

The dashboard (`dashboard/app.py`) is designed to be hosted publicly on
**Streamlit Community Cloud** without ever exposing an API key, and to work
fully with **no key at all**.

## API key policy

| Situation | Judge used in the playground | Who pays |
|---|---|---|
| Visitor pastes their own key in the sidebar | LLM judge (Claude) | The visitor |
| No visitor key, owner set `ANTHROPIC_API_KEY` in Secrets | LLM judge, capped at `ANCHOR_SERVER_KEY_SESSION_LIMIT` calls per session (default 5) | The owner |
| No key anywhere | Offline heuristic judge | Nobody |

The recommended public setup is to **not set a server key**. Visitors can
still paste their own key to see the LLM judge, and everyone else gets the
offline judge, so the demo costs nothing.

How keys are handled in code:

- The sidebar field is a `type="password"` input. The value lives only in
  that visitor's Streamlit session, and nothing writes it to disk.
- `FaithfulnessJudge` keeps the key in a private attribute. Its `repr` never
  shows the key, and the key never goes into a verdict or the trace DB.
- If the LLM call fails (bad key, network, refusal), the judge falls back to
  the heuristic. The verdict records only the exception *type*, for example
  `AuthenticationError`. The provider's error message is dropped, so it
  can't echo a key back onto the page.
- The trace DB path is fixed on the server (`ANCHOR_DB` env var), so visitors
  can't point the app at other files on the host. With `ANCHOR_PUBLIC=1`,
  each browser session gets its own temporary database instead.
- LLM work per click is capped (`ANCHOR_MAX_LLM_CASES`, default 25). Use of
  the server's key is capped per session (`ANCHOR_SERVER_KEY_SESSION_LIMIT`).
- `tests/test_judge.py` and `tests/test_dashboard.py` check all of this.

## Deploy on Streamlit Community Cloud

1. Push the repo to GitHub (public or private).
2. Go to https://share.streamlit.io, choose **Create app**, then **Deploy a public app from GitHub**.
3. Repository `arnabmanandhar/anchor`, branch `main`, main file path `dashboard/app.py`.
4. Under **Advanced settings → Secrets**, add `ANCHOR_PUBLIC = "1"`. This
   gives every visitor a private workspace, so one visitor's runs and
   uploads are never visible to another. Optionally, also paste the key
   line from `.streamlit/secrets.toml.example` with a real key. Skip the
   key for a zero-cost demo.
5. Click **Deploy**. Each visitor's workspace starts seeded with the
   bundled demo, two retriever variants ready to compare, and they can
   run their own evaluations from the **Evaluate** page.

Streamlit Cloud installs `requirements.txt` automatically. The app's disk
is ephemeral, so a restart re-seeds the demo. That's fine, because the demo
is deterministic.

## Pre-deploy security checklist

Run these from the repo root before every public deploy:

```bash
# 1. No real keys in tracked files or anywhere in git history
git grep -nE "sk-ant-[A-Za-z0-9_-]{20,}" || echo "clean: working tree"
git log --all -p | grep -nE "sk-ant-[A-Za-z0-9_-]{20,}" || echo "clean: history"

# 2. Secrets files are ignored and not tracked
git check-ignore -v .env .streamlit/secrets.toml
git ls-files | grep -E "(^|/)(\.env|secrets\.toml)$" && echo "TRACKED SECRET FILE!" || echo "clean: none tracked"

# 3. Everything works with no key
env -u ANTHROPIC_API_KEY pytest -q
env -u ANTHROPIC_API_KEY anchor demo
```

On Windows PowerShell, run step 3 with:

```powershell
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
pytest -q
anchor demo
```

If a real key ever appears in git history, even in a deleted commit,
**rotate it in the Anthropic Console first**. Rewriting history doesn't
un-leak a key that has already been pushed.

After deploying:

- Open the app in a private window. Check that the sidebar says the
  **offline heuristic judge** is in use, unless you set a server key.
- Open the browser's Network tab and run the playground. No request from
  your browser should go to `api.anthropic.com`, because calls happen
  server-side. No response should contain a key.
- Paste a deliberately wrong key, such as `sk-ant-wrong`, and run the
  playground. You should see `LLM judge failed (AuthenticationError); used
  heuristic fallback.` and the key should appear nowhere on the page.

## What visitors can do

- **Evaluate:** build a pipeline from the sample corpus or their own file,
  upload their own pipeline's logged outputs, or score a single answer.
- **Results and Compare:** fixed 0–1 charts, per-question regressions, and
  an interactive regression gate.
- **Inspect:** the claim-by-claim audit, ranked chunks marked relevant or
  cited, and trace timings.
- **Learn the metrics:** the formulas, with worked examples.

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for a full walkthrough.
