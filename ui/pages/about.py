"""
ui/pages/about.py -- Product-quality overview and trust page.
"""

from __future__ import annotations

import streamlit as st

from ui.theme import render_page_header, render_trust_strip


def render_about() -> None:
    """Render the About / trust page."""
    render_page_header(
        kicker="How it works",
        title="Model scope, methodology, and operating limits",
        subtitle=(
            "This page explains what the model does, what it does not do, and where caution is required."
        ),
        status_key="neutral",
        status_label="Model overview",
    )

    col_left, col_right = st.columns([1.05, 0.95], gap="large")

    with col_left:
        st.markdown(
            """
            <section class="tv-surface tv-about-card">
                <h3>Product purpose</h3>
                <p>
                    TradeVision classifies candlestick and OHLC chart screenshots into known technical pattern
                    categories. It is designed to surface a likely visual match, expose the full confidence
                    distribution, and flag uncertain outputs rather than overstate certainty.
                </p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <section class="tv-surface tv-about-card">
                <h3>Supported pattern families</h3>
                <ul>
                    <li>Head and Shoulders and Inverse Head and Shoulders</li>
                    <li>Double Top and Double Bottom</li>
                    <li>Cup and Handle</li>
                    <li>Ascending, Descending, and Symmetrical Triangles</li>
                    <li>Rising Wedge, Falling Wedge, Flag, Pennant, and Rounding Bottom</li>
                </ul>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <section class="tv-surface tv-about-card">
                <h3>Model approach</h3>
                <p>
                    The classifier uses transfer learning with an EfficientNetB0 backbone and a custom
                    classification head. Uploaded images are resized to 224 by 224, normalized, and scored
                    against the learned class set. The interface then applies threshold-aware display logic
                    so weak or ambiguous outputs are treated conservatively.
                </p>
            </section>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            """
            <section class="tv-surface tv-about-card">
                <h3>Technology footprint</h3>
                <table>
                    <thead>
                        <tr><th align="left">Layer</th><th align="left">Technology</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>Interface</td><td>Streamlit</td></tr>
                        <tr><td>Inference</td><td>TensorFlow and Keras</td></tr>
                        <tr><td>Backbone</td><td>EfficientNetB0</td></tr>
                        <tr><td>Image preprocessing</td><td>Pillow and NumPy</td></tr>
                        <tr><td>Visualization</td><td>Plotly</td></tr>
                    </tbody>
                </table>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <section class="tv-surface tv-about-card">
                <h3>Limits and failure modes</h3>
                <ul>
                    <li>TradeVision does not forecast price direction or future returns.</li>
                    <li>It does not issue buy, sell, hold, or risk-management signals.</li>
                    <li>Low-confidence outputs may indicate ambiguous structure, unsupported patterns, or weak image quality.</li>
                    <li>Results remain sensitive to dataset quality, labeling quality, and how closely the uploaded chart resembles the training distribution.</li>
                </ul>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <section class="tv-surface tv-about-card">
                <h3>Dataset and training posture</h3>
                <p>
                    The model is trained on a chart-pattern image dataset organized by class folder. The
                    standard pipeline uses a train, validation, and test split with augmentation applied
                    during training only.
                </p>
            </section>
            """,
            unsafe_allow_html=True,
        )

    render_trust_strip(
        [
            (
                "Not financial advice",
                "Pattern recognition output should never be treated as a substitute for regulated investment guidance.",
            ),
            (
                "Best use case",
                "TradeVision is most useful as a research companion for image classification, inspection, and model iteration.",
            ),
        ]
    )

    st.markdown(
        """
        <section class="tv-surface tv-about-card" style="margin-top:1rem;">
            <h3>Inference flow</h3>
            <p>
                Chart image input flows through resize and normalization, then into the frozen EfficientNetB0
                feature extractor and custom dense classification head. The application surfaces the leading
                class, confidence score, margin to the next candidate, and the ranked probability distribution.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
