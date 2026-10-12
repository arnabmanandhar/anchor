"""Smoke test the Streamlit dashboard headlessly: it must seed itself from
an empty database, render with no API key, and never echo a pasted key."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANCHOR_DB", str(tmp_path / "dash.db"))
    return AppTest.from_file(APP, default_timeout=60)


def test_seeds_and_renders_without_key(app):
    app.run()
    assert not app.exception
    assert any("seeded" in i.value for i in app.info)
    assert app.metric[0].value == "16"
    assert "offline heuristic judge" in app.sidebar.caption[0].value


def test_playground_runs_heuristic_judge(app):
    app.run()
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[-1].label == "Faithfulness"


def test_pasted_key_is_never_rendered(app, monkeypatch):
    import anchor.judge as judge_mod

    secret = "sk-ant-dashboard-test-key"

    def boom(self, *a, **k):
        raise RuntimeError(f"auth failed for {secret}")

    monkeypatch.setattr(judge_mod.FaithfulnessJudge, "_score_llm", boom)
    app.run()
    app.sidebar.text_input[0].input(secret).run()
    app.button[0].click().run()
    rendered = " ".join(
        str(el.value) for group in (app.markdown, app.caption, app.sidebar.caption, app.info, app.metric) for el in group
    )
    assert secret not in rendered
    assert "LLM judge failed (RuntimeError)" in rendered
