"""
ui/components/result_card.py — Prediction result display component.

Renders:
- Pattern name (large, styled header)
- Confidence badge (color-coded)
- Low-confidence warning callout
- Expandable "See all pattern scores" section
"""

from __future__ import annotations

import streamlit as st

from tradevision.inference.predictor import PredictionResult
import config


def _confidence_color(confidence: float) -> str:
    """Return a hex colour string based on confidence level.

    Args:
        confidence: Probability in [0, 1].

    Returns:
        Hex colour string.
    """
    if confidence >= 0.80:
        return "#22c55e"   # green-500
    elif confidence >= config.CONFIDENCE_THRESHOLD:
        return "#f59e0b"   # amber-500
    else:
        return "#ef4444"   # red-500


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.80:
        return "High"
    elif confidence >= config.CONFIDENCE_THRESHOLD:
        return "Moderate"
    else:
        return "Low"


def render_result_card(result: PredictionResult) -> None:
    """Render the prediction result card.

    Args:
        result: Dict returned by :func:`Predictor.predict`.
    """
    pattern: str = result["pattern"]
    predicted_class: str = result.get("predicted_class", pattern)
    confidence: float = result["confidence"]
    all_scores: dict[str, float] = result["all_scores"]
    low_confidence: bool = result["low_confidence"]

    color = _confidence_color(confidence)
    label = _confidence_label(confidence)
    pct = confidence * 100

    # ── Pattern name ──────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid {color}44;
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-bottom: 1rem;
        ">
            <p style="color:#94a3b8; font-size:0.85rem; margin:0 0 0.25rem 0;
                      text-transform:uppercase; letter-spacing:0.08em;">
                Detected Pattern
            </p>
            <h2 style="color:#f1f5f9; font-size:2rem; font-weight:700;
                       margin:0 0 1rem 0; line-height:1.2;">
                {pattern}
            </h2>
            <span style="
                display:inline-block;
                background:{color}22;
                color:{color};
                border:1px solid {color}55;
                border-radius:999px;
                padding:0.35rem 1.1rem;
                font-size:0.95rem;
                font-weight:600;
            ">
                {label} Confidence — {pct:.1f}%
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Low confidence warning ─────────────────────────────────────────────
    if low_confidence:
        st.warning(
            "⚠️ **Low confidence result** — The uploaded image may not contain a supported "
            "chart pattern, or the pattern may be too ambiguous for a reliable prediction. "
            f"Top candidate: **{predicted_class}**. Use this result with caution.",
            icon=None,
        )

    # ── Expandable full scores ─────────────────────────────────────────────
    with st.expander("🔍 See all pattern scores"):
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        for cls, score in sorted_scores:
            pct_score = score * 100
            bar_color = color if cls == predicted_class else "#475569"
            st.markdown(
                f"""
                <div style="margin-bottom:0.5rem;">
                    <div style="display:flex; justify-content:space-between;
                                margin-bottom:2px;">
                        <span style="color:#cbd5e1; font-size:0.85rem;">{cls}</span>
                        <span style="color:{bar_color}; font-size:0.85rem;
                                     font-weight:600;">{pct_score:.1f}%</span>
                    </div>
                    <div style="background:#1e293b; border-radius:4px; height:6px;">
                        <div style="background:{bar_color}; width:{pct_score:.1f}%;
                                    height:6px; border-radius:4px;
                                    transition: width 0.3s ease;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
