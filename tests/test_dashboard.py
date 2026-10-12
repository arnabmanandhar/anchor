"""Headless tests for the Streamlit dashboard: every page renders from an
empty workspace with no API key, the Evaluate flows run end to end, charts
use a fixed 0–1 scale, and a pasted key never appears on the page."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASH = Path(__file__).resolve().parent.parent / "dashboard"
PAGES = ["home", "evaluate", "results", "compare", "inspect", "learn"]


def page(name: str) -> AppTest:
    script = f"import sys; sys.path.insert(0, {str(DASH)!r}); import views; views.render({name!r})"
    return AppTest.from_string(script, default_timeout=90)


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANCHOR_PUBLIC", raising=False)
    monkeypatch.setenv("ANCHOR_DB", str(tmp_path / "dash.db"))
    return tmp_path / "dash.db"


@pytest.mark.parametrize("name", PAGES)
def test_every_page_renders_offline(name):
    at = page(name).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("offline" in i.value for i in at.sidebar.info)


def test_empty_workspace_is_seeded_with_demo(workspace):
    from anchor.tracer import Tracer

    page("home").run()
    assert len(Tracer(str(workspace)).list_versions()) == 2


def test_build_and_run_all_retrievers():
    at = page("evaluate").run()
    at.checkbox[0].check().run()
    run = next(b for b in at.button if b.label == "Run evaluation")
    run.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("across 4 version(s)" in s.value for s in at.success)


def test_upload_tab_scores_sample_records():
    at = page("evaluate").run()
    next(b for b in at.button if b.label == "Score these outputs").click().run()
    assert not at.exception
    assert any("1 version(s): sample-logged-pipeline" in s.value for s in at.success)


def test_single_answer_catches_the_hallucinated_claim():
    at = page("evaluate").run()
    next(b for b in at.button if b.label == "Score").click().run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "0.50" in html and "Unsupported" in html


def test_score_charts_use_fixed_unit_domain():
    import pandas as pd

    import ui

    means = pd.DataFrame({"faithfulness": [0.6], "recall_at_k": [0.6]}, index=pd.Index(["v1"], name="pipeline_version"))
    spec = ui.metric_bars(means, ["v1"]).to_dict()
    assert spec["encoding"]["y"]["scale"]["domain"] == [0, 1]


def test_pasted_key_is_never_rendered(monkeypatch):
    import anchor.judge as judge_mod

    secret = "sk-ant-dashboard-test-key"

    def boom(self, *a, **k):
        raise RuntimeError(f"auth failed for {secret}")

    monkeypatch.setattr(judge_mod.FaithfulnessJudge, "_score_llm", boom)
    at = page("evaluate").run()
    at.sidebar.text_input[0].input(secret).run()
    at.selectbox(key="single_judge").select("LLM (Claude)").run()
    next(b for b in at.button if b.label == "Score").click().run()
    rendered = " ".join(
        str(el.value)
        for group in (at.markdown, at.caption, at.sidebar.caption, at.info, at.success, at.sidebar.success)
        for el in group
    )
    assert secret not in rendered
    assert "LLM judge failed (RuntimeError)" in rendered


def test_public_mode_gives_each_session_a_private_workspace(workspace, monkeypatch):
    monkeypatch.setenv("ANCHOR_PUBLIC", "1")
    a = page("results").run()
    b = page("results").run()
    assert not a.exception and not b.exception
    assert a.session_state["db_path"] != b.session_state["db_path"]
    assert not workspace.exists()  # the shared DB was never touched
    assert any("Private workspace" in c.value for c in a.sidebar.caption)


def test_compare_shows_significance_and_gate_toggle():
    at = page("compare").run()
    assert not at.exception
    assert any("Is the difference real?" in m.value for m in at.markdown)
    assert any("p-value" in df.value.columns for df in at.dataframe)
    at.toggle[0].set_value(True).run()
    assert not at.exception
    assert any("--require-significant" in c.value for c in at.code)
