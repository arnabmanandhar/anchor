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
    assert any("runs offline" in c.value for c in at.sidebar.caption)


def test_empty_workspace_is_seeded_with_demo(workspace):
    from anchor.tracer import Tracer

    page("home").run()
    assert len(Tracer(str(workspace)).list_versions()) == 2


def test_build_and_run_all_retrievers():
    at = page("evaluate").run()
    next(c for c in at.checkbox if c.label.startswith("Also run every other retriever")).check().run()
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


def add_model(at, provider="groq", key="gsk_dashboard-test-key-000"):
    at.sidebar.selectbox(key="conn_form_provider").select(provider).run()
    at.sidebar.text_input(key="conn_form_key").input(key)
    next(b for b in at.sidebar.button if b.label == "Add model").click().run()
    return at


def test_pasted_key_is_never_rendered_or_stored(monkeypatch, workspace):
    import anchor.llm as llm

    secret = "gsk_dashboard-test-key-000"

    def reject(url, headers, payload, s):
        raise llm.LLMError("auth", llm.redact(f"invalid api key {secret}", s), 401)

    monkeypatch.setattr(llm, "_post", reject)
    at = add_model(page("evaluate").run(), key=secret)
    assert not at.exception
    assert at.session_state["conn_form_key"] in ("", None)  # the form cleared the raw key
    assert any("Groq" in m.value for m in at.sidebar.markdown)
    next(b for b in at.button if b.label == "Score").click().run()
    rendered = " ".join(
        str(el.value)
        for group in (at.markdown, at.caption, at.sidebar.caption, at.sidebar.markdown, at.info, at.success,
                      at.error, at.warning, at.sidebar.success, at.sidebar.error)
        for el in group
    )
    assert secret not in rendered
    assert "LLM judge failed (auth)" in rendered
    assert secret.encode() not in workspace.read_bytes()


def test_connections_are_per_session_and_can_be_forgotten():
    at = add_model(page("evaluate").run())
    assert at.selectbox(key="build_judge").value is not None  # a new connection becomes the default
    other = page("evaluate").run()  # a different browser session
    assert any("runs offline" in c.value for c in other.sidebar.caption)
    next(b for b in at.sidebar.button if b.label == "Forget all keys").click().run()
    assert any("runs offline" in c.value for c in at.sidebar.caption)


def test_public_mode_blocks_private_custom_endpoints(monkeypatch):
    monkeypatch.setenv("ANCHOR_PUBLIC", "1")
    at = page("evaluate").run()
    at.sidebar.selectbox(key="conn_form_provider").select("custom").run()
    at.sidebar.text_input(key="conn_form_base").input("http://169.254.169.254/latest")
    next(b for b in at.sidebar.button if b.label == "Add model").click().run()
    assert any("blocked" in e.value for e in at.sidebar.error)


def test_owner_key_is_capped_and_never_shown(monkeypatch):
    owner = "gsk_owner-secret-key-0000"
    monkeypatch.setenv("GROQ_API_KEY", owner)
    monkeypatch.setenv("ANCHOR_SERVER_KEY_SESSION_LIMIT", "3")
    at = page("evaluate").run()
    text = " ".join(str(el.value) for el in (*at.sidebar.markdown, *at.sidebar.caption))
    assert "Built in" in text and "3 calls left" in text and owner not in text and "0000" not in text


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


def test_generate_questions_and_replace_the_table():
    at = page("evaluate").run()
    next(b for b in at.button if b.label == "Generate questions").click().run()
    assert not at.exception
    assert any("12 questions" in m.value for m in at.markdown)
    next(b for b in at.button if b.label.startswith("Replace questions with these")).click().run()
    assert not at.exception
    assert len(at.session_state["questions::Sample: Constitution of Nepal (20 chunks)"]) == 12
