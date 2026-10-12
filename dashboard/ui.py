"""Presentation helpers for the dashboard: styles, score cards, and charts.

Every score Anchor produces lives on [0, 1], so every score visual here is
drawn on a fixed 0–1 scale: a 0.6 is a bar 60% of the way across, never a
full-height bar on an axis that happens to stop at 0.6.
"""

from __future__ import annotations

import html
import json
from typing import Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

# Categorical palette, assigned to versions in a fixed order (validated
# for colour-vision deficiency on light surfaces as adjacent pairs).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

METRICS = [
    # key in a run row, summary key, label, one-line meaning
    ("faithfulness", "mean_faithfulness", "Faithfulness", "Share of answer claims grounded in a retrieved chunk"),
    ("recall_at_k", "mean_recall_at_k", "Recall@k", "Share of relevant chunks found in the top k"),
    ("mrr", "mean_mrr", "MRR", "1 / rank of the first relevant chunk"),
    ("ndcg_at_k", "mean_ndcg_at_k", "nDCG@k", "Ranking quality, rewarding relevant chunks near the top"),
    ("map_score", "mean_map", "MAP", "Precision averaged at every relevant hit"),
]
METRIC_LABELS = {m[0]: m[2] for m in METRICS}

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1200px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.anchor-hero {padding: 1.6rem 1.8rem; border-radius: 16px;
  background: linear-gradient(135deg, rgba(42,120,214,0.12), rgba(27,175,122,0.10));
  border: 1px solid rgba(42,120,214,0.25); margin-bottom: 1.2rem;}
.anchor-hero h1 {margin: 0 0 .3rem 0; font-size: 2.3rem;}
.anchor-hero p {margin: 0; font-size: 1.08rem; opacity: .85;}
.ws-value {font-size: 1.05rem; font-weight: 600;}
.anchor-card {border: 1px solid rgba(128,128,128,0.25); border-radius: 12px;
  padding: 1rem 1.1rem; height: 100%; background: rgba(128,128,128,0.04);}
.anchor-card h4 {margin: 0 0 .35rem 0; font-size: 1rem;}
.anchor-card p {margin: 0; font-size: .92rem; opacity: .85;}
.score-card {border: 1px solid rgba(128,128,128,0.25); border-radius: 12px;
  padding: .85rem 1rem .9rem; background: rgba(128,128,128,0.04);}
.score-card .label {font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; opacity: .7;}
.score-card .value {font-size: 1.7rem; font-weight: 650; line-height: 1.25;}
.score-card .value span {font-size: .9rem; font-weight: 400; opacity: .6;}
.score-card .track {height: 8px; border-radius: 4px; background: rgba(128,128,128,0.18); margin-top: .45rem; overflow: hidden;}
.score-card .fill {height: 100%; border-radius: 4px;}
.score-card .hint {font-size: .78rem; opacity: .65; margin-top: .4rem; min-height: 3.6em;}
.claim {border-radius: 10px; padding: .6rem .85rem; margin: .35rem 0; border: 1px solid;}
.claim.ok {border-color: rgba(0,131,0,.35); background: rgba(0,131,0,.07);}
.claim.bad {border-color: rgba(227,73,72,.45); background: rgba(227,73,72,.08);}
.claim .pill {display: inline-block; font-size: .7rem; font-weight: 600; letter-spacing: .03em; text-transform: uppercase;
  padding: .05rem .5rem; border-radius: 999px; margin-right: .45rem; vertical-align: 1px;}
.claim.ok .pill {background: rgba(0,131,0,.14); color: #008300;}
.claim.bad .pill {background: rgba(227,73,72,.14); color: #e34948;}
.claim .meta {font-size: .8rem; opacity: .75; margin-top: .2rem;}
.chunk {border: 1px solid rgba(128,128,128,0.25); border-radius: 10px; padding: .55rem .8rem; margin: .3rem 0;}
.badge {display: inline-block; font-size: .72rem; padding: .05rem .45rem; border-radius: 999px; margin-right: .3rem;
  border: 1px solid rgba(128,128,128,.4);}
.badge.rel {border-color: rgba(42,120,214,.6); color: #2a78d6;}
.badge.cited {border-color: rgba(0,131,0,.6); color: #008300;}
.badge.pass {border-color: rgba(0,131,0,.6); color: #008300;}
.badge.fail {border-color: rgba(227,73,72,.7); color: #e34948;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(str(text))


def _score_color(value: float) -> str:
    # Neutral blue fill; the number, not the colour, carries the meaning.
    return SERIES[0]


def score_card(label: str, value: Optional[float], hint: str = "", fmt: str = "{:.2f}") -> str:
    if value is None or pd.isna(value):
        shown, width = "—", 0
    else:
        shown, width = fmt.format(value), max(0.0, min(1.0, float(value))) * 100
    return (
        f'<div class="score-card"><div class="label">{esc(label)}</div>'
        f'<div class="value">{shown}<span> / 1</span></div>'
        f'<div class="track"><div class="fill" style="width:{width:.1f}%;background:{_score_color(value or 0)}"></div></div>'
        f'<div class="hint">{esc(hint)}</div></div>'
    )


def score_cards(values: Dict[str, Optional[float]]) -> None:
    """values: {metric_key: value} for the metrics in METRICS order."""
    cols = st.columns(len(METRICS))
    for col, (key, _, label, hint) in zip(cols, METRICS):
        col.markdown(score_card(label, values.get(key), hint), unsafe_allow_html=True)


def card(title: str, body: str) -> str:
    return f'<div class="anchor-card"><h4>{esc(title)}</h4><p>{body}</p></div>'


def version_colors(shown: List[str], all_versions: List[str]) -> alt.Scale:
    # Colour follows the version's position in the full sorted list, so
    # filtering never repaints the remaining versions; the legend lists only
    # the versions on screen.
    shown = [v for v in all_versions if v in set(shown)]
    return alt.Scale(domain=shown, range=[SERIES[all_versions.index(v) % len(SERIES)] for v in shown])


def metric_bars(means: pd.DataFrame, all_versions: List[str]) -> alt.Chart:
    """means: one row per version, columns = metric keys. Grouped bars on a fixed 0–1 axis."""
    long = means.reset_index().melt(id_vars="pipeline_version", var_name="metric", value_name="score").dropna()
    long["metric"] = long["metric"].map(METRIC_LABELS)
    order = [m[2] for m in METRICS]
    return (
        alt.Chart(long)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("metric:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("pipeline_version:N", sort=all_versions, scale=alt.Scale(paddingInner=0.08)),
            y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1], nice=False), title="Score (0–1)",
                    axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1], format=".2f")),
            color=alt.Color("pipeline_version:N", scale=version_colors(list(means.index), all_versions), title="Version",
                            legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("pipeline_version:N", title="Version"), alt.Tooltip("metric:N"),
                     alt.Tooltip("score:Q", format=".3f")],
        )
        .properties(height=320)
    )


def per_query_bars(df: pd.DataFrame, metric: str, all_versions: List[str]) -> alt.Chart:
    """Horizontal bars per query (and version) for one metric, on a fixed 0–1 axis."""
    data = df[["query", "pipeline_version", metric]].dropna()
    n_rows = data["query"].nunique()
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("query:N", title=None, sort=None, axis=alt.Axis(labelLimit=380)),
            yOffset=alt.YOffset("pipeline_version:N", sort=all_versions),
            x=alt.X(f"{metric}:Q", scale=alt.Scale(domain=[0, 1], nice=False), title=f"{METRIC_LABELS[metric]} (0–1)",
                    axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1], format=".2f")),
            color=alt.Color("pipeline_version:N", scale=version_colors(list(data["pipeline_version"].unique()), all_versions),
                            title="Version", legend=alt.Legend(orient="top")),
            tooltip=["query", "pipeline_version", alt.Tooltip(f"{metric}:Q", format=".3f")],
        )
        .properties(height=max(160, 26 * n_rows * max(1, data["pipeline_version"].nunique()) ** 0.6))
    )


def delta_ci_chart(tests: Dict[str, Dict]) -> alt.Chart:
    """Mean paired difference per metric with its confidence interval, on a
    fixed −1 to +1 axis (every score difference lives there)."""
    order = [m[2] for m in METRICS]
    data = pd.DataFrame([
        {"metric": METRIC_LABELS[k], "delta": t["mean_delta"], "low": t["ci_low"], "high": t["ci_high"],
         "p": t["p_value"], "verdict": ("improved" if t["mean_delta"] > 0 else "regressed") if t["significant"] else "not significant"}
        for k, t in tests.items()
    ])
    x = alt.Scale(domain=[-1, 1], nice=False)
    colour = alt.Color("verdict:N", title=None, legend=alt.Legend(orient="top"),
                       scale=alt.Scale(domain=["improved", "regressed", "not significant"],
                                       range=[SERIES[5], SERIES[7], "#8a8f98"]))
    y = alt.Y("metric:N", sort=order, title=None)
    tooltip = [alt.Tooltip("metric:N", title="Metric"), alt.Tooltip("delta:Q", title="Δ (B − A)", format="+.3f"),
               alt.Tooltip("low:Q", title="CI low", format="+.3f"), alt.Tooltip("high:Q", title="CI high", format="+.3f"),
               alt.Tooltip("p:Q", title="p-value", format=".3f")]
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8a8f98", strokeDash=[4, 3]).encode(x=alt.X("x:Q", scale=x))
    interval = alt.Chart(data).mark_rule(strokeWidth=3).encode(
        y=y, x=alt.X("low:Q", scale=x, title="Change B − A (−1 to +1)", axis=alt.Axis(values=[-1, -0.5, 0, 0.5, 1], format="+.1f")),
        x2="high:Q", color=colour, tooltip=tooltip)
    point = alt.Chart(data).mark_point(filled=True, size=90).encode(y=y, x=alt.X("delta:Q", scale=x), color=colour, tooltip=tooltip)
    return (zero + interval + point).properties(height=44 * len(data) + 40)


def span_chart(spans: List[Dict]) -> alt.Chart:
    data = pd.DataFrame(spans)[["name", "duration_ms"]]
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color=SERIES[0])
        .encode(
            y=alt.Y("name:N", sort=None, title=None),
            x=alt.X("duration_ms:Q", title="Duration (ms)"),
            tooltip=["name", alt.Tooltip("duration_ms:Q", format=".2f")],
        )
        .properties(height=36 * len(data) + 20)
    )


def score_column(label: str, help_text: str = "") -> st.column_config.ProgressColumn:
    return st.column_config.ProgressColumn(label, help=help_text, min_value=0.0, max_value=1.0, format="%.2f")


SCORE_COLUMNS = {key: score_column(label, hint) for key, _, label, hint in METRICS}


def claim_cards(claims: List[Dict]) -> None:
    for c in claims:
        ok = c.get("supported")
        meta = f"Supported by chunk <b>{esc(c.get('chunk_id'))}</b>" if ok else f"Unsupported. {esc(c.get('note') or '')}"
        st.markdown(
            f'<div class="claim {"ok" if ok else "bad"}"><span class="pill">{"Supported" if ok else "Unsupported"}</span>{esc(c.get("claim"))}'
            f'<div class="meta">{meta}</div></div>',
            unsafe_allow_html=True,
        )


def chunk_cards(chunks: List[Dict], reference: set, cited: set) -> None:
    for rank, c in enumerate(chunks, start=1):
        badges = ""
        if c["id"] in reference:
            badges += '<span class="badge rel">relevant</span>'
        if c["id"] in cited:
            badges += '<span class="badge cited">cited by judge</span>'
        score = f" · score {c['score']:.3f}" if c.get("score") is not None else ""
        st.markdown(
            f'<div class="chunk"><b>#{rank} · {esc(c["id"])}</b>{score} {badges}<br>{esc(c["text"])}</div>',
            unsafe_allow_html=True,
        )


def runs_frame(runs: List[Dict]) -> pd.DataFrame:
    """Flatten trace-DB run rows into a DataFrame with one column per metric."""
    df = pd.DataFrame(runs)
    if df.empty:
        return df
    judge = df["judge_json"].apply(json.loads)
    retrieval = df["retrieval_json"].apply(lambda r: json.loads(r) if r else None)
    df["faithfulness"] = judge.apply(lambda j: j["faithfulness_score"])
    df["judge_backend"] = judge.apply(lambda j: j["judge_backend"])
    df["claims"] = judge.apply(lambda j: len(j["claims"]))
    for key in ("recall_at_k", "mrr", "ndcg_at_k", "map_score"):
        df[key] = retrieval.apply(lambda r, k=key: r[k] if r else None)
    df["k"] = retrieval.apply(lambda r: r["k"] if r else None)
    return df
