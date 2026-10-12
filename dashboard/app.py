"""
Anchor dashboard.  Run with:  streamlit run dashboard/app.py

Pages
  Home      what Anchor is, how it works, where to start
  Evaluate  run evaluations from the browser, three ways:
              1. build a pipeline here (corpus + questions + retriever)
              2. upload outputs your own pipeline already produced
              3. score a single answer by hand
  Results   per-version scores on a fixed 0–1 scale, every run, downloads
  Compare   two versions side by side, per-query changes, regression gate
  Inspect   one run in full: answer, claim audit, ranked chunks, trace
  Learn     the metrics, derived (renders docs/METRICS.md)

Storage: by default runs go to anchor_traces.db in the repo root, shared
with the CLI. With ANCHOR_PUBLIC=1 (public deploys) every browser session
gets its own private database, seeded with the demo.

Models (see connections.py and DEPLOYMENT.md): each browser session can add
several provider/model/key connections in the sidebar; they live only in
that session's memory and expire when idle. The owner's key, if configured,
is offered as a capped built-in model. With no model, everything runs
offline. Keys never reach the database, logs or page.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import views  # noqa: E402

views.setup()

views.PAGES.update(
    {
        "home": st.Page(views.page_home, title="Home", icon=":material/anchor:", default=True),
        "evaluate": st.Page(views.page_evaluate, title="Evaluate", icon=":material/play_circle:", url_path="evaluate"),
        "results": st.Page(views.page_results, title="Results", icon=":material/bar_chart:", url_path="results"),
        "compare": st.Page(views.page_compare, title="Compare", icon=":material/compare_arrows:", url_path="compare"),
        "inspect": st.Page(views.page_inspect, title="Inspect", icon=":material/manage_search:", url_path="inspect"),
        "learn": st.Page(views.page_learn, title="Learn the metrics", icon=":material/menu_book:", url_path="learn"),
    }
)

views.sidebar()
st.navigation(list(views.PAGES.values())).run()
