"""
ui/pages/about.py — About page for TradeVision.

Describes the application, its limitations, dataset source, and disclaimers.
"""

import streamlit as st


def render_about() -> None:
    """Render the About / information page."""

    st.markdown(
        """
        <div style="text-align:center; padding:2rem 0 1rem 0;">
            <h1 style="font-size:2.5rem; font-weight:800; color:#f1f5f9;">
                About TradeVision
            </h1>
            <p style="color:#94a3b8; font-size:1.05rem;">
                AI-powered technical chart pattern recognition
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 🎯 What TradeVision Does")
        st.markdown(
            """
            TradeVision uses **Transfer Learning** (EfficientNetB0 backbone with custom
            classification head) to classify technical analysis patterns from candlestick
            / OHLC chart images.

            **Supported patterns include:**
            - Head and Shoulders / Inverse H&S
            - Double Top / Double Bottom
            - Cup and Handle
            - Ascending / Descending Triangle
            - Symmetrical Triangle
            - Rising / Falling Wedge
            - Flag and Pennant
            - Rounding Bottom

            The model outputs a probability distribution across all pattern classes
            and reports the most likely match along with a confidence score.
            """
        )

        st.markdown("### 🛠️ Technology Stack")
        st.markdown(
            """
            | Layer | Technology |
            |---|---|
            | UI | Streamlit |
            | Deep Learning | TensorFlow / Keras |
            | Backbone | EfficientNetB0 (ImageNet) |
            | Preprocessing | Pillow, NumPy |
            | Visualisation | Plotly |
            | Training | Keras callbacks + scikit-learn |
            """
        )

    with col2:
        st.markdown("### ❌ What TradeVision Does NOT Do")
        st.error(
            "TradeVision does **not** predict future price movements, "
            "generate buy/sell signals, or provide any form of financial advice.\n\n"
            "Pattern recognition ≠ price prediction.",
            icon="🚫",
        )

        st.markdown("### 📊 Dataset")
        st.markdown(
            """
            The model is trained on a Kaggle chart-pattern image dataset.
            See `data/download_dataset.py` for the dataset slug.

            **Dataset characteristics:**
            - Images: Candlestick / OHLC chart screenshots
            - Classes: Organised by pattern type in sub-folders
            - Split: 70% train / 15% val / 15% test
            - Augmentation: Flip, rotate, zoom, brightness (training only)
            """
        )

        st.markdown("### ⚠️ Disclaimer")
        st.warning(
            "This application is for **educational and research purposes only**.\n\n"
            "Do not use TradeVision outputs to make real financial decisions. "
            "Past chart patterns do not guarantee future performance. "
            "Always consult a licensed financial advisor.",
            icon="⚠️",
        )

    st.markdown("---")

    # ── How it works ─────────────────────────────────────────────────────────
    st.markdown("### ⚙️ How It Works")
    st.markdown(
        """
        ```
        Chart Image
            ↓ Resize to 224×224  ·  Normalise [0,1]  ·  RGB conversion
        EfficientNetB0 (frozen ImageNet weights)
            ↓
        GlobalAveragePooling2D → Dense(256, ReLU) → BatchNorm → Dropout(0.4)
            ↓
        Dense(num_classes, Softmax)
            ↓
        Pattern + Confidence Score
        ```

        The backbone weights are **frozen** during training so that only the
        classification head learns task-specific features, preventing catastrophic
        forgetting of ImageNet representations and reducing training time significantly.
        """
    )

    st.markdown("---")
    st.caption("TradeVision v0.1.0 · Built with Streamlit & TensorFlow")
