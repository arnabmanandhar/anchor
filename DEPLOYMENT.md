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
  can't point the app at other files on the host.
- `tests/test_judge.py` and `tests/test_dashboard.py` check all of this.

## Deploy on Streamlit Community Cloud

1. Push the repo to GitHub (public or private).
2. Go to https://share.streamlit.io, choose **Create app**, then **Deploy a public app from GitHub**.
3. Repository `arnabmanandhar/anchor`, branch `main`, main file path `dashboard/app.py`.
4. Optional: under **Advanced settings → Secrets**, paste the contents of
   `.streamlit/secrets.toml.example` with a real key. Skip this for a
   zero-cost demo.
5. Click **Deploy**. On first load the trace database is empty, so the app
   seeds itself by running the bundled demo with the offline judge. Two
   retriever variants appear, ready to compare.

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

## What visitors can verify

The dashboard is meant to show depth, not only that a demo works:

- **Overview:** mean metrics per pipeline version.
- **Compare versions:** aggregate deltas, plus which individual queries
  regressed. The keyword retriever loses recall on the multi-relevant
  question even though its MRR is unchanged.
- **Inspect a run:** the generated answer, the claim-by-claim audit, the
  ranked chunks marked relevant or cited, and the timed trace spans.
- **Judge playground:** paste any chunks and answer to watch the judge audit
  them live.
