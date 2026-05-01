"""
ui/components/confidence_chart.py — Confidence score bar chart component.

Renders a Plotly horizontal bar chart showing the top-N class probabilities,
with the winning class highlighted in a distinct accent colour.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import config
from tradevision.inference.predictor import PredictionResult


def render_confidence_chart(result: PredictionResult) -> None:
    """Render a horizontal bar chart of the top-N class probabilities.

    Args:
        result: Dict returned by :func:`Predictor.predict`.
    """
    all_scores: dict[str, float] = result["all_scores"]
    top_pattern: str = result.get("predicted_class", result["pattern"])

    # Sort descending and take top-N
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    top_n = sorted_scores[: config.TOP_N_CLASSES]

    labels = [cls for cls, _ in top_n]
    values = [score * 100 for _, score in top_n]
    colors = [
        "#6366f1" if cls == top_pattern else "#334155"
        for cls in labels
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(0,0,0,0)", width=0),
            ),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=12),
            hovertemplate="<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Top {len(top_n)} Pattern Scores",
            font=dict(color="#f1f5f9", size=16),
            x=0,
        ),
        xaxis=dict(
            title="Confidence (%)",
            range=[0, 110],
            tickfont=dict(color="#64748b"),
            gridcolor="#1e293b",
            zerolinecolor="#1e293b",
        ),
        yaxis=dict(
            tickfont=dict(color="#94a3b8", size=13),
            autorange="reversed",
        ),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        margin=dict(l=10, r=40, t=50, b=40),
        height=max(260, len(top_n) * 52),
        font=dict(family="Inter, sans-serif"),
    )

    st.plotly_chart(fig, use_container_width=True)
