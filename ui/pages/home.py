"""
ui/pages/home.py -- Prediction workspace.

Single-step workflow:
    1. Product header with current system state
    2. Upload workspace and preview pane
    3. Automatic inference for valid uploads
    4. Result summary and probability distribution
    5. Trust/compliance strip
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
from ui.theme import (
    icon_svg,
    render_empty_preview,
    render_page_header,
    render_section_intro,
    render_trust_strip,
)

logger = get_logger(__name__)


def _status_for_upload(upload_status: str) -> tuple[str, str]:
    """Map upload state to header treatment."""
    if upload_status == "invalid":
        return "warning", "Upload needs attention"
    if upload_status == "ready":
        return "processing", "Chart ready for inference"
    return "neutral", "Awaiting chart upload"


def render_home(predictor: Predictor) -> None:
    """Render the Home / prediction workspace page."""
    if not config.MODEL_PATH.exists():
        render_page_header(
            kicker="Classification workspace",
            title="Pattern intelligence for chart screenshots",
            subtitle=(
                "TradeVision turns a single uploaded chart into a model-driven pattern readout, "
                "but the workspace depends on a trained local model being available."
            ),
            status_key="critical",
            status_label="Model artifact missing",
        )
        render_section_intro(
            "Model required",
            (
                "The interface is ready, but inference is blocked until a trained checkpoint and "
                "its metadata are present in the local models directory."
            ),
            chips=[
                "Step 1: python data/download_dataset.py",
                "Step 2: python -m tradevision.model.trainer",
            ],
        )
        st.error(
            "No trained model was found. Train TradeVision locally, then refresh the workspace."
        )
        render_trust_strip(
            [
                (
                    "Research posture",
                    "This product surface is for pattern recognition only and does not produce trading advice.",
                ),
                (
                    "Deployment note",
                    "Model availability is a local runtime dependency rather than a hosted service dependency.",
                ),
            ]
        )
        st.stop()

    header_slot = st.container()

    upload_col, preview_col = st.columns([1.05, 1.15], gap="large")

    with upload_col:
        upload_copy = (
            "Use a chart screenshot with visible candles and minimal overlay clutter. "
            "The model performs best when the dominant visual content is the price structure itself."
        )
        render_section_intro(
            "Input workspace",
            upload_copy,
            chips=["Single image", "Automatic inference", "Max 10 MB"],
        )
        file_bytes, upload_meta = render_uploader()

    status_key, status_label = _status_for_upload(str(upload_meta["status"]))
    with header_slot:
        render_page_header(
            kicker="Classification workspace",
            title="Pattern intelligence for chart screenshots",
            subtitle=(
                "Upload one chart, review the leading class, and inspect the probability distribution."
            ),
            status_key=status_key,
            status_label=status_label,
        )

    with preview_col:
        render_section_intro(
            "Preview pane",
            (
                "Uploaded charts are shown here before prediction so the image quality and framing "
                "can be validated quickly."
            ),
            chips=["Live preview", "No preprocessing edits", "Inference-ready"],
        )

        if file_bytes is None:
            render_empty_preview()
        else:
            image = Image.open(io.BytesIO(file_bytes))
            st.image(image, use_container_width=True)
            file_name = upload_meta.get("file_name") or "Uploaded chart"
            size_mb = upload_meta.get("file_size_mb")
            size_label = (
                f"{float(size_mb):.1f} MB" if isinstance(size_mb, float) else "Validated image"
            )
            st.markdown(
                f"""
                <p class="tv-preview-caption">
                    {file_name} · {size_label}
                </p>
                """,
                unsafe_allow_html=True,
            )

    if file_bytes is None:
        render_trust_strip(
            [
                (
                    "Input expectations",
                    "Best results come from candlestick or OHLC screenshots with readable structure.",
                ),
                (
                    "Model scope",
                    "TradeVision classifies known pattern families and can reject ambiguous or unsupported images.",
                ),
            ]
        )
        return

    st.markdown("")

    result = None
    if upload_meta["status"] == "ready":
        with st.spinner("Analyzing chart image"):
            try:
                result = predictor.predict(file_bytes)
            except ModelNotFoundError as exc:
                logger.error("ModelNotFoundError: %s", exc)
                st.error(
                    "The model artifact is missing from the models directory. Retrain the model and reload the page."
                )
                render_trust_strip(
                    [
                        ("Inference state", "Runtime blocked because the model checkpoint could not be loaded."),
                        ("Recovery", "Retrain locally and confirm `class_names.json` exists beside the checkpoint."),
                    ]
                )
                return
            except ModelMetadataError:
                logger.exception("Model metadata missing or invalid")
                st.error(
                    "Model metadata is missing or inconsistent. Retrain the model so the saved checkpoint includes valid class labels."
                )
                render_trust_strip(
                    [
                        ("Inference state", "Prediction halted because class metadata is unavailable."),
                        ("Recovery", "Regenerate the model artifacts and retry the same upload."),
                    ]
                )
                return
            except InvalidImageError:
                logger.exception("InvalidImageError during inference")
                st.error(
                    "The uploaded image could not be processed for inference. Use a valid chart screenshot in a supported format."
                )
                render_trust_strip(
                    [
                        ("Image quality", "Corrupt files and non-chart images are rejected before or during preprocessing."),
                        ("Recommended action", "Try a different screenshot with visible price structure."),
                    ]
                )
                return
            except InferenceError:
                logger.exception("InferenceError during prediction")
                st.error(
                    "Prediction failed during model execution. Try another chart or inspect the application logs for runtime details."
                )
                render_trust_strip(
                    [
                        ("Failure mode", "The forward pass did not complete successfully."),
                        ("Recommended action", "Retry with a new chart and inspect logs if the issue persists."),
                    ]
                )
                return
            except Exception:  # noqa: BLE001
                logger.exception("Unexpected inference error")
                st.error(
                    "An unexpected application error interrupted the analysis. Check the logs for full diagnostic details."
                )
                render_trust_strip(
                    [
                        ("Failure mode", "An unhandled application error interrupted the prediction flow."),
                        ("Recommended action", "Review local logs before re-running the same input."),
                    ]
                )
                return

    if result is None:
        return

    status_key = "warning" if result["low_confidence"] else "success"
    status_label = (
        "Result available with caution"
        if result["low_confidence"]
        else "Result available"
    )
    render_page_header(
        kicker="Analysis output",
        title="Model verdict with ranked class confidence",
        subtitle=(
            "The leading class, confidence score, and ranked probabilities are shown below."
        ),
        status_key=status_key,
        status_label=status_label,
    )

    st.markdown(
        f"""
        <div class="tv-surface tv-surface-muted">
            <p class="tv-section-label">Result workspace</p>
            <p class="tv-helper">
                Primary classifications are summarized as a single decision card, while supporting
                class probabilities remain visible for comparison.
            </p>
            <div class="tv-meta-row">
                <span class="tv-meta-chip">{icon_svg("shield")} Threshold-aware output</span>
                <span class="tv-meta-chip">{icon_svg("chart")} Top {config.TOP_N_CLASSES} classes visible</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    card_col, chart_col = st.columns([1.02, 0.98], gap="large")
    with card_col:
        render_result_card(result)
    with chart_col:
        render_confidence_chart(result)

    render_trust_strip(
        [
            (
                "Interpretation limit",
                "Pattern recognition is not equivalent to forecasting, signaling, or portfolio advice.",
            ),
            (
                "Confidence handling",
                "Low-confidence outputs should be treated as ambiguous image classifications rather than actionable conclusions.",
            ),
        ]
    )
