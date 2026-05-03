"""
ui/components/confidence_chart.py -- Ranked probability card.
"""

from __future__ import annotations

from html import escape

import streamlit as st

import config
from tradevision.inference.predictor import PredictionResult


def render_confidence_chart(result: PredictionResult) -> None:
    """Render the single secondary probability view."""
    all_scores: dict[str, float] = result["all_scores"]
    top_pattern: str = result.get("predicted_class", result["pattern"])

    sorted_scores = sorted(all_scores.items(), key=lambda item: item[1], reverse=True)
    top_n = sorted_scores[: config.TOP_N_CLASSES]

    probability_rows: list[str] = []
    for index, (label, score) in enumerate(top_n, start=1):
        pct = score * 100
        emphasized = label == top_pattern
        fill_color = "#1D4ED8" if emphasized else "#94A3B8"
        row_class = "tv-probability-row tv-probability-row-top" if emphasized else "tv-probability-row"
        probability_rows.append(
            (
                f'<div class="{row_class}">'
                f'<div class="tv-probability-meta">'
                f'<span class="tv-probability-rank">{index:02d}</span>'
                f'<span class="tv-probability-label">{escape(label)}</span>'
                f'<span class="tv-probability-value">{pct:.1f}%</span>'
                f"</div>"
                f'<div class="tv-probability-track">'
                f'<div class="tv-probability-fill" style="width:{pct:.1f}%; background:{fill_color};"></div>'
                f"</div>"
                f"</div>"
            )
        )

    st.markdown(
        f"""
        <section class="tv-surface tv-probability-card">
            <p class="tv-section-label">Top {len(top_n)} class probabilities</p>
            <div class="tv-probability-list">
                {''.join(probability_rows)}
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
