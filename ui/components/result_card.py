"""
ui/components/result_card.py -- KPI-style prediction summary card.
"""

from __future__ import annotations

from html import escape
import streamlit as st

import config
from tradevision.inference.predictor import PredictionResult


def _confidence_tone(confidence: float) -> tuple[str, str, str]:
    """Return semantic colors and label for the confidence score."""
    if confidence >= 0.80:
        return "#ECFDF3", "#047857", "High confidence"
    if confidence >= config.CONFIDENCE_THRESHOLD:
        return "#FFF7ED", "#B45309", "Moderate confidence"
    return "#FEF2F2", "#B91C1C", "Low confidence"


def _confidence_explainer(confidence: float, margin: float, low_confidence: bool) -> str:
    """Generate the result interpretation copy."""
    if low_confidence:
        return (
            "The leading class is visible, but the score or score margin is too weak to treat the image "
            "as a confident pattern match."
        )
    if confidence >= 0.80 and margin >= config.CONFIDENCE_MARGIN_THRESHOLD:
        return (
            "The model found a strong leading class with enough separation from the next candidate to display "
            "the recognized pattern directly."
        )
    return (
        "The top class is acceptable, but the probability spread remains worth reviewing before relying on the output."
    )


def render_result_card(result: PredictionResult) -> None:
    """Render the main decision card and supporting probability list."""
    pattern: str = result["pattern"]
    predicted_class: str = result.get("predicted_class", pattern)
    confidence: float = result["confidence"]
    low_confidence: bool = result["low_confidence"]
    confidence_margin: float = result.get("confidence_margin", 0.0)

    background, foreground, label = _confidence_tone(confidence)
    pct = confidence * 100
    margin_pct = confidence_margin * 100

    st.markdown(
        f"""
        <section class="tv-kpi-card">
            <p class="tv-kpi-label">Primary classification</p>
            <h2 class="tv-kpi-title">{escape(pattern)}</h2>
            <div style="margin-top:0.95rem;">
                <span class="tv-status-pill"
                      style="background:{background}; color:{foreground}; border-color:{foreground}22;">
                    <span class="tv-status-dot"></span>
                    {escape(label)} · {pct:.1f}%
                </span>
            </div>
            <p class="tv-kpi-note">{escape(_confidence_explainer(confidence, confidence_margin, low_confidence))}</p>
            <div class="tv-metric-row">
                <div class="tv-metric">
                    <span class="tv-metric-label">Top candidate</span>
                    <span class="tv-metric-value">{escape(predicted_class)}</span>
                </div>
                <div class="tv-metric">
                    <span class="tv-metric-label">Confidence margin</span>
                    <span class="tv-metric-value">{margin_pct:.1f}%</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if low_confidence:
        st.markdown(
            """
            <div class="tv-note-callout">
                This output is below the decision threshold or too close to the runner-up class.
                Review the wider score distribution before treating the image as a recognized pattern.
            </div>
            """,
            unsafe_allow_html=True,
        )
