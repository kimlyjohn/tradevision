"""
ui/pages/home.py — Main prediction page.

Flow:
    1. Header
    2. Model availability check
    3. File uploader
    4. Image preview
    5. Spinner → inference
    6. Result card + confidence chart
"""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

import config
from tradevision.inference.predictor import (
    InferenceError,
    InvalidImageError,
    ModelMetadataError,
    ModelNotFoundError,
    Predictor,
)
from tradevision.utils.logger import get_logger
from ui.components.confidence_chart import render_confidence_chart
from ui.components.result_card import render_result_card
from ui.components.uploader import render_uploader

logger = get_logger(__name__)


def render_home(predictor: Predictor) -> None:
    """Render the Home / prediction page.

    Args:
        predictor: Shared :class:`Predictor` instance (loaded via cache_resource).
    """
    # ── Hero header ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding: 2.5rem 0 1.5rem 0;">
            <h1 style="font-size:3rem; font-weight:800;
                       background:linear-gradient(135deg,#6366f1,#8b5cf6,#a855f7);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                       margin-bottom:0.5rem;">
                TradeVision
            </h1>
            <p style="color:#94a3b8; font-size:1.15rem; max-width:540px;
                      margin:0 auto;">
                Deep learning–powered chart pattern recognition.
                Upload any candlestick chart and get an instant classification.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Model availability check ─────────────────────────────────────────────
    if not config.MODEL_PATH.exists():
        st.info(
            "🤖 **Model not trained yet.**\n\n"
            "No trained model was found. Train TradeVision first:\n\n"
            "```bash\n"
            "python data/download_dataset.py   # 1. Download dataset\n"
            "python -m tradevision.model.trainer  # 2. Train model\n"
            "```\n\n"
            "Once training is complete, refresh this page.",
            icon=None,
        )
        st.stop()

    # ── Upload section ───────────────────────────────────────────────────────
    col_upload, col_preview = st.columns([1, 1], gap="large")

    with col_upload:
        file_bytes = render_uploader()

    # ── Preview ──────────────────────────────────────────────────────────────
    if file_bytes is None:
        with col_preview:
            st.markdown(
                """
                <div style="
                    height:300px; border:2px dashed #334155;
                    border-radius:16px; display:flex; flex-direction:column;
                    align-items:center; justify-content:center;
                    color:#475569; font-size:0.9rem; text-align:center;
                    padding:1rem;
                ">
                    <span style="font-size:3rem;">📈</span>
                    <p style="margin:0.5rem 0 0 0;">
                        Your chart preview will appear here
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    # Show preview
    with col_preview:
        img = Image.open(io.BytesIO(file_bytes))
        st.image(img, caption="Uploaded chart", use_container_width=True)

    # ── Inference ────────────────────────────────────────────────────────────
    st.markdown("---")
    with st.spinner("🔍 Analysing pattern …"):
        try:
            result = predictor.predict(file_bytes)
        except ModelNotFoundError as exc:
            logger.error("ModelNotFoundError: %s", exc)
            st.error(
                "❌ **Model file missing.** "
                "Please train the model and place it in the `models/` directory."
            )
            return
        except ModelMetadataError:
            logger.exception("Model metadata missing or invalid")
            st.error(
                "❌ **Model metadata is missing or invalid.** "
                "Retrain the model so `class_names.json` is saved next to the checkpoint."
            )
            return
        except InvalidImageError:
            logger.exception("InvalidImageError during inference")
            st.error(
                "❌ **Could not process this image.** "
                "Please upload a valid chart screenshot in JPG, PNG, or WEBP format."
            )
            return
        except InferenceError:
            logger.exception("InferenceError during prediction")
            st.error(
                "❌ **Prediction failed.** "
                "The model could not complete inference for this image. Please try another chart."
            )
            return
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected inference error")
            st.error(
                "❌ **An unexpected error occurred.** "
                "Check the logs for details."
            )
            return

    # ── Results ──────────────────────────────────────────────────────────────
    st.markdown("## 🎯 Analysis Result")

    col_card, col_chart = st.columns([1, 1], gap="large")

    with col_card:
        render_result_card(result)

    with col_chart:
        render_confidence_chart(result)

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** TradeVision is a research tool for pattern recognition only. "
        "It does **not** predict future prices, generate trading signals, or constitute "
        "financial advice. Always consult a qualified financial advisor before trading."
    )
